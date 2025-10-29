"""
Intent Gatherer & Query Transformation Agent
Analyzes user intent and performs strategic retrieval based on query complexity
"""

import json
from typing import Dict, List, Optional
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from services.agent_schemas import IntentResult, KBDocument
from services.rag_scoring import score_and_pack
import services.services as services
from services.config_loader import get_config
from services.logger_setup import setup_logger
logger = setup_logger()
config = get_config()
# eaach aspect create should be a new search query and therefore self explanatory. Think more.
# we need to find the query topic and stick to it for question on each aspect. a new meta data needs to be created maybe.
class IntentGathererAgent:
    """
    Agent responsible for:
    1. Intent classification (query, complaint, service_request, feature_request)
    2. Multi-faceted query analysis (aspects extraction)
    3. Strategic retrieval based on complexity
    4. Scope gating (out_of_scope detection for non-BFSI queries)
    """
    
    def __init__(self, llm, tenant_id: str, user_role: str):
        """
        Initialize the Intent Gatherer Agent
        
        Args:
            llm: Language model instance (should support structured output)
            tenant_id: Unique identifier for the tenant
            user_role: User role for RBAC filtering
        """
        self.llm = llm
        # Create structured output LLM for intent analysis
        self.structured_llm = llm.with_structured_output(IntentResult)
        self.tenant_id = tenant_id
        self.user_role = user_role
        self.config = config
        
    def process(self, state: dict) -> dict:
        """
        Process user query to extract intent and gather relevant KB documents
        
        Args:
            state: Agent state containing messages and context
            
        Returns:
            Updated state with intent_result and kb_docs
        """
        logger.info("IntentGathererAgent: Processing user query")
        
        # Step 1: Get intent analysis from LLM
        intent_data = self._analyze_intent(state)
        intent_result = IntentResult(**intent_data)
        
        logger.info(f"Intent Analysis: {intent_result.intent}, aspects: {intent_result.aspects}, "
                   f"out_of_scope: {intent_result.out_of_scope}")
        
        if intent_result.out_of_scope:
            logger.info("Query is out of scope, skipping retrieval")
            return {
                'intent_result': intent_result.dict(),
                'kb_docs': []
            }
        
        # Step 3: Perform strategic retrieval based on complexity
        user_query = self._extract_query_text(state['messages'])
        kb_docs = self._gather_documents(user_query, intent_result.aspects)
        
        logger.info(f"Retrieved {len(kb_docs)} documents for user query")
        
        return {
            'intent_result': intent_result.dict(),
            'kb_docs': [doc.dict() for doc in kb_docs]
        }
    
    def _analyze_intent(self, state: dict) -> dict:
        system_prompt = """You are an intent analysis expert for BFSI (Banking, Financial Services, Insurance) domain.

Analyze the user query and classify their intent, urgency, sentiment.
Also perform query tranformation and generatre different self defining aspects for a retrieval (rag) service to find supporting docuemnts from knowledge base further.

Guidelines:
1. **Intent Classification:**
   - query: User asking for information, clarification, or help understanding policies
   - complaint: User expressing dissatisfaction or reporting an issue
   - service_request: User requesting a specific service/action (refund, policy change, etc.)
   - feature_request: User suggesting new features or improvements

2. **Aspects Extraction (IMPORTANT):**
   - For simple, single-topic queries: ["single aspect"]
   - For multi-faceted queries: break into distinct, least-granular aspects
   - Each aspect should be a searchable sub-topic
   - Examples:
     * "What are payment policies?" → ["payment policies"]
     * "What are payment policies and cancellation fees?" → ["payment policies", "cancellation fees"]
     * "How do I upgrade my policy and what happens to my current coverage?" → ["policy upgrade process", "coverage during upgrade"]

3. **Urgency Assessment:**
   - high: Explicit urgency words (urgent, critical, emergency, ASAP)
   - medium: Default for complaints and service requests
   - low: General queries, feature requests

4. **Sentiment:**
   - positive: Expressing satisfaction, gratitude, praise
   - neutral: Factual, informational tone (default)
   - negative: Expressing frustration, anger, dissatisfaction

5. **Out of Scope Detection (CRITICAL):**
   - Set true if query is NOT about BFSI (Banking, Financial Services, Insurance) domain
   - BFSI includes: banking accounts, loans, credit cards, insurance policies, claims, investments, financial products
   - Examples of OUT-OF-SCOPE: 
     * Furniture/appliance rental queries
     * E-commerce product questions
     * General world knowledge
     * Jokes or casual chat
     * Healthcare, travel, food services
   - Examples of IN-SCOPE: 
     * Insurance claims and policy questions
     * Banking account issues
     * Credit card disputes
     * Investment queries
     * Financial service complaints
   - Be STRICT: If it's not clearly BFSI-related, mark as out_of_scope=true
"""

        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        
        try:
            # Use structured output LLM - this will return an IntentResult object
            intent_result = self.structured_llm.invoke(messages)
            
            # Convert Pydantic model to dict for compatibility
            intent_data = intent_result.dict()
            
            # Ensure at least one aspect
            if not intent_data['aspects']:
                intent_data['aspects'] = [self._extract_query_text(state['messages'])]
            
            logger.debug(f"Intent analysis result: {intent_data}")
            return intent_data
            
        except Exception as e:
            logger.error(f"Error in structured intent analysis: {e}")
            # Fallback to default intent
            return self._default_intent(state)
    
    def _default_intent(self, state: dict) -> dict:
        query = self._extract_query_text(state['messages'])
        return {
            'intent': 'query',
            'urgency': 'medium',
            'sentiment': 'neutral',
            'aspects': [query],
            'out_of_scope': False
        }
    
    def _gather_documents(self, query: str, aspects: List[str]) -> List[KBDocument]:
        """
        Strategic multi-search with diversity (MMR) and scoring aggregation
        
        Strategy:
        - If len(aspects) <= 1: single retrieval
        - If len(aspects) > 1: parallel retrieval per aspect, union, score, dedupe, top-K
        - Use MMR for diversity when configured
        - Apply scoring and threshold filtering
        
        Args:
            query: Original user query
            aspects: List of query aspects/facets
            
        Returns:
            List of KBDocument objects, scored and deduplicated
        """
        threshold = self.config.get('retrieval.threshold', 0.25)
        max_results = self.config.get('chat.max_retrieval_results', 8)
        k_per_aspect = self.config.get('retrieval.k', 6)
        
        all_documents = []
        all_scores = []
        
        if len(aspects) <= 1:
            # Simple query: single retrieval
            search_query = aspects[0] if aspects else query
            logger.debug(f"Single-aspect retrieval for: {search_query}")
            
            docs, scores = services.retrieve_with_scores(
                query=search_query,
                tenant_id=self.tenant_id,
                user_role=self.user_role,
                k=k_per_aspect
            )
            
            all_documents.extend(docs)
            all_scores.extend(scores)
            
        else:
            # Complex query: multi-aspect retrieval
            logger.debug(f"Multi-aspect retrieval for {len(aspects)} aspects")
            
            for aspect in aspects:
                try:
                    docs, scores = services.retrieve_with_scores(
                        query=f"{query} {aspect}",  # Combine original query with aspect
                        tenant_id=self.tenant_id,
                        user_role=self.user_role,
                        k=k_per_aspect // len(aspects) + 1  # Distribute k across aspects
                    )
                    
                    all_documents.extend(docs)
                    all_scores.extend(scores)
                    logger.debug(f"Aspect '{aspect[:30]}...' retrieved {len(docs)} docs")
                    
                except Exception as e:
                    logger.warning(f"Failed to retrieve for aspect '{aspect}': {e}")
                    continue
        
        if not all_documents:
            logger.warning("No documents retrieved for any aspect")
            return []
        
        # Score, dedupe, and pack documents
        kb_docs = score_and_pack(
            query=query,
            documents=all_documents,
            similarity_scores=all_scores,
            threshold=threshold,
            max_results=max_results
        )
        
        logger.info(f"Final result: {len(kb_docs)} documents after scoring and deduplication")
        return kb_docs
    
    def _extract_query_text(self, messages: List) -> str:
        """
        Extract the text content from the last user message
        
        Args:
            messages: List of chat messages
            
        Returns:
            Text content of the last human message
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                # Handle both string content and structured content
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Extract text from structured content
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            return item.get('text', '')
                    return str(msg.content)
                else:
                    return str(msg.content)
        
        return ""

