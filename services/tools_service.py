"""
Tools Service
Centralized tool definitions for use across agents
"""
from langchain_core.tools import tool
from services.config_loader import get_config
from services.logger_setup import setup_logger
from services.jira_tool import JiraTool
import services.services as services

logger = setup_logger()
config = get_config()

def get_retriever_tool(tenant_id: str, user_role: str):
    """
    Create tenant-aware retriever tool
    
    Args:
        tenant_id: Tenant identifier for filtering
        user_role: User role for RBAC
        
    Returns:
        Retriever tool instance
    """
    @tool
    def retriever_tool(query: str) -> str:
        """
        This tool searches and returns the information from the organization's knowledge base.
        Use this tool to fetch company policies, documents, and procedures relevant to claims, complaints, or service requests.
        """
        try:
            # Use retrieve_with_scores from services
            docs, scores = services.retrieve_with_scores(
                query=query,
                tenant_id=tenant_id,
                user_role=user_role,
                k=config.get('retrieval.k', 5)
            )
            
            if not docs:
                return "I found no relevant information in the knowledge base."
            
            # Format results with scores
            results = []
            for i, (doc, score) in enumerate(zip(docs, scores)):
                results.append(f"Document {i+1} (relevance: {score:.2f}):\n{doc.page_content}")
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Retriever tool failed: {e}")
            return f"Error retrieving documents: {str(e)}"
    
    return retriever_tool

@tool
def create_jira_ticket(summary: str, description: str, issue_type: str, urgency: str = "medium", sentiment: str = "neutral") -> str:
    """
    Creates a JIRA ticket for service requests, complaints, and feature requests.
    
    Args:
        summary: Short, self-defining subject of the ticket (max 100 chars)
        description: Detailed description with all necessary information to help resolve the issue
        issue_type: Type of issue - one of: 'service_request', 'complaint', 'feature_request'
        urgency: Priority level - 'high' (critical/urgent), 'medium' (default), 'low'
        sentiment: User sentiment - 'positive', 'neutral' (default), 'negative'
        
    Returns:
        Ticket ID as string if success, error message if failed
    """
    try:
        jira_tool = JiraTool()
        ticket_key = jira_tool.create_ticket(summary, description, issue_type, urgency, sentiment)
        return f"Successfully created JIRA ticket: {ticket_key}"
    except Exception as e:
        logger.error(f"JIRA ticket creation failed: {e}")
        return f"Failed to create JIRA ticket: {str(e)}"

def get_all_tools(tenant_id: str, user_role: str):
    """
    Get all available tools for agents
    
    Args:
        tenant_id: Tenant identifier
        user_role: User role for RBAC
        
    Returns:
        List of tool instances
    """
    from services.user_data_service import get_user_data, list_user_policies
    
    retriever = get_retriever_tool(tenant_id, user_role)
    
    return [
        retriever,
        create_jira_ticket,
        get_user_data,
        list_user_policies
    ]

