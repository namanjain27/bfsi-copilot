"""
Answer Generator Agent
Composes grounded answers using KB documents and user messages
"""

from typing import Dict, List
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from services.agent_schemas import KBDocument
from services.logger_setup import setup_logger
logger = setup_logger()


class AnswerGeneratorAgent:
    """
    Agent responsible for generating grounded answers from KB documents.
    
    Key principles:
    - Use ONLY KB content, avoid assumptions
    - Cite sources minimally (filenames/sections)
    - Handle cases where KB has insufficient information or query is out of scope
    """
    
    def __init__(self, llm):
        """
        Initialize the Answer Generator Agent
        
        Args:
            llm: Language model instance (no tools needed, just generation)
        """
        self.llm = llm
        
    def process(self, state: dict) -> dict:
        """
        Generate grounded answer using KB documents
        
        Args:
            state: Agent state containing:
                - messages: Chat history
                - kb_docs: List of KBDocument objects
                - intent_result: Intent analysis result
                
        Returns:
            Updated state with final answer
        """
        logger.info("AnswerGeneratorAgent: Generating answer")
        
        kb_docs_data = state.get('kb_docs', [])
        kb_docs = [KBDocument(**doc) if isinstance(doc, dict) else doc for doc in kb_docs_data]
        intent_result = state.get('intent_result')
        
        # Build context from KB documents
        kb_context = self._format_kb_context(kb_docs)
        answer = self._generate_answer(state['messages'], kb_context, intent_result)
        
        logger.info(f"Generated answer of length {len(answer)}")
        return {
            'messages': [AIMessage(content=answer)],
            'final_answer': answer,
            'kb_docs_used': [doc.source for doc in kb_docs]
        }
    
    def _format_kb_context(self, kb_docs: List[KBDocument]) -> str:
        """
        Format KB documents into context string for the LLM
        
        Args:
            kb_docs: List of KBDocument objects
            
        Returns:
            Formatted context string with sources and relevance scores
        """
        if not kb_docs:
            return "No relevant documents found in knowledge base."
        
        context_parts = ["=== Knowledge Base Documents ===\n"]
        
        for i, doc in enumerate(kb_docs, 1):
            source_name = doc.source.split('/')[-1] if '/' in doc.source else doc.source
            context_parts.append(
                f"\n[Document {i}] (Source: {source_name}, Relevance: {doc.score:.2f})\n"
                f"{doc.content}\n"
            )
        
        context_parts.append("\n=== End of Knowledge Base Documents ===")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, messages: List, kb_context: str, intent_result) -> str:
        """
        Generate answer using LLM with KB context
        
        Args:
            messages: Chat history
            kb_context: Formatted KB documents
            intent_result: Intent analysis result
            
        Returns:
            Generated answer string
        """
        system_prompt = f"""You are a helpful AI assistant for a BFSI (Banking, Financial Services, Insurance) organization.

Your task is to answer the user's query using ONLY the information provided in the Knowledge Base documents below.

{kb_context}

**Guidelines:**
1. **Grounded Responses:** Base your answer STRICTLY on the KB documents provided. Do not use external knowledge or make assumptions.

2. **Source Citations:** When referencing information, briefly mention the source document (e.g., "According to the policy document...").

3. **Clarity & Completeness:** 
   - Provide clear, direct answers to the user's question
   - If multiple documents are relevant, synthesize the information coherently
   - Use bullet points or numbered lists for multi-part answers

4. **Handling Insufficient Information:**
   - If KB documents don't contain enough information: Clearly state what information IS available and what is missing
   - Offer to create a ticket for further assistance: "I don't have complete information on this in my knowledge base. Would you like me to create a ticket so our team can help you further?"
   - Do NOT guess or invent information

5. **Tone:**
   - Professional and helpful
   - Empathetic for complaints/issues
   - Clear and concise

6. **Avoiding Out-of-Scope:**
   - Stay focused on the organization's services and policies
   - Politely decline questions outside BFSI domain

Remember: If the information is not in the KB documents, say so clearly and offer alternatives."""

        # Combine system prompt with chat history
        conversation = [SystemMessage(content=system_prompt)] + list(messages)
        
        try:
            # Generate response using LLM (without tools)
            response = self.llm.invoke(conversation)
            answer = response.content
            
            # Ensure we have a valid response
            if not answer or not answer.strip():
                logger.warning("LLM returned empty response, using fallback")
                return self._fallback_answer(kb_context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._fallback_answer(kb_context)
    
    def _fallback_answer(self, kb_context: str) -> str:
        """
        Generate fallback answer when LLM fails
        
        Args:
            kb_context: Formatted KB context
            
        Returns:
            Fallback answer string
        """
        if "No relevant documents found" in kb_context:
            return ("I couldn't find relevant information in my knowledge base to answer your question. "
                   "Would you like me to create a ticket so our team can assist you further?")
        else:
            return ("I found some relevant information in my knowledge base, but I'm having trouble "
                   "processing it right now. Would you like me to create a ticket for further assistance?")

