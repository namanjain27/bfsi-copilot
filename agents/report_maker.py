"""
Report Maker Agent
Structures information into a report for claim verification
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from services.agent_schemas import Report, KBDocument
from services.logger_setup import setup_logger
logger = setup_logger()

# do not just pass in the kb docs received in the output as company_docs_about_issue. it needs to be rephrased and restructured in order to facilitate the user query for further evaluation.  
class ReportExtraction(BaseModel):
    """Schema for structured report extraction from LLM"""
    issue: str = Field(description="Clear, concise statement of the problem or topic (1-2 sentences)")
    user_demand: str = Field(description="What the user is specifically requesting or asking for")
    support_info_from_user: str = Field(description="Any evidence, context, details, or background information provided by the user")
    policy_refs: str = Field(description="Policy numbers or section references mentioned (if any)")
    company_docs_about_issue: str = Field(
        description="Rephrased and restructured relevant information from company documents. "
                    "Format as clear paragraphs or numbered points in a single string field."
    )

class ReportMakerAgent:
    """
    Agent responsible for structuring information into a standardized report.
    
    Key principles:
    - Extract and organize information from user messages and KB docs
    - Do NOT assume or invent information
    - Use ONLY provided input
    - Create clear, crisp report for claim verifier
    """
    
    def __init__(self, llm):
        """
        Initialize the Report Maker Agent
        
        Args:
            llm: Language model instance (no tools, just text-to-structure conversion)
        """
        self.llm = llm
        self.structured_llm = llm.with_structured_output(ReportExtraction)
        
    def process(self, state: dict) -> dict:
        """
        Structure input into a Report
        
        Args:
            state: Agent state containing:
                - messages: Chat history with user query
                - kb_docs: List of KBDocument objects from intent gatherer
                - intent_result: Intent analysis result
                
        Returns:
            Updated state with structured report
        """
        logger.info("ReportMakerAgent: Creating structured report")
        
        kb_docs_data = state.get('kb_docs', [])
        kb_docs = [KBDocument(**doc) if isinstance(doc, dict) else doc for doc in kb_docs_data]
        messages = state.get('messages', [])
        self.messages = messages
        
        # Build context from messages and KB documents
        context = self._format_context(messages, kb_docs)
        
        # Extract structured report from LLM
        report = self._extract_report(context, kb_docs)
        
        logger.info(f"Created report - Issue: {report.issue[:50]}...")
        
        return {
            'report': report.dict()
        }
    
    def _format_context(self, messages: List, kb_docs: List[KBDocument]) -> str:
        """
        Format user messages and KB documents into context for extraction
        
        Args:
            messages: Chat history
            kb_docs: Knowledge base documents
            intent_result: Intent classification result
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add user messages (extract only human messages for clarity)
        context_parts.append("=== User Messages ===")
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                context_parts.append(f"User: {msg.content}")
        context_parts.append("")
        
        # Add KB documents
        if kb_docs:
            context_parts.append("=== Company Documents (Knowledge Base) ===")
            for i, doc in enumerate(kb_docs, 1):
                source_name = doc.source.split('/')[-1] if '/' in doc.source else doc.source
                context_parts.append(
                    f"\n[Document {i}] (Source: {source_name}, Relevance: {doc.score:.2f})"
                )
                context_parts.append(doc.content)
                context_parts.append("")
        else:
            context_parts.append("=== Company Documents ===")
            context_parts.append("No relevant documents found in knowledge base.")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _extract_report(self, context: str, kb_docs: List[KBDocument]) -> Report:
        """
        Extract structured report using LLM with structured output
        
        Args:
            context: Formatted context string
            kb_docs: Original KB documents to include in report
            
        Returns:
            Report object
        """
        system_prompt = """You are a report structuring assistant for a BFSI organization.

Extract and structure the following from the provided context:

1. **issue**: Clear statement of the problem (1-2 sentences)
2. **user_demand**: What the user is requesting
3. **support_info_from_user**: Evidence/context provided by user
4. **company_docs_about_issue**: Rephrase ONLY the relevant parts of company documents that address the user's issue. Format as numbered points or paragraphs in a SINGLE STRING. Be concise.
5. **policy_refs**: Policy numbers or section references (if mentioned)

**Rules:**
- Use ONLY information from the provided context
- If info not available, use "Not specified"
- Be crisp - extract only what's relevant to the user's issue
- For company_docs_about_issue: Rephrase relevant KB content into clear, concise format (string, not list)"""

        user_prompt = f"""Please analyze the following information and create a structured report:

{context}"""

        try:
            extraction_result = self.structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Create Report with rephrased company docs
            report = Report(
                issue=extraction_result.issue,
                user_demand=extraction_result.user_demand,
                company_docs_about_issue=extraction_result.company_docs_about_issue, 
                support_info_from_user=extraction_result.support_info_from_user,
                policy_refs=extraction_result.policy_refs
            )
            
            # Log with character count and preview
            docs_preview = report.company_docs_about_issue[:100] if report.company_docs_about_issue else "Empty"
            logger.info(f"Successfully extracted report. Company docs: {len(report.company_docs_about_issue)} chars, preview: {docs_preview}...")
            return report
        except Exception as e:
            logger.error(f"Error in report extraction: {e}")
            return self._fallback_report(context, kb_docs)
    
    def _fallback_report(self, context: str, kb_docs: List[KBDocument]) -> Report:
        logger.warning("Using fallback report creation")
        issue = ""
        for msg in self.messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                issue = msg.content
        
        # Convert KB docs to single string for fallback
        if kb_docs:
            company_docs_parts = []
            for i, doc in enumerate(kb_docs, 1):
                content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                company_docs_parts.append(f"{i}. {content}")
            company_docs_text = "\n\n".join(company_docs_parts)
        else:
            company_docs_text = "No company documents available"
        
        return Report(
            issue=issue if issue else "Issue not specified",
            user_demand="As mentioned in the issue above",
            company_docs_about_issue=company_docs_text,  # Now returns string, not list
            support_info_from_user="As mentioned in the issue above",
            policy_refs="Not specified"
        )

