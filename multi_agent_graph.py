"""
Multi-Agent Graph Orchestration for BFSI Workflow
Uses LangGraph to coordinate Intent, Answer, Report, and Verification agents
"""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from operator import add as add_messages
from langchain.chat_models import init_chat_model

from services.config_loader import get_config
from services.tools_service import get_all_tools
from agents import IntentGathererAgent, AnswerGeneratorAgent, ReportMakerAgent, ClaimVerifierAgent
import os
import getpass
from services.logger_setup import setup_logger

load_dotenv()

logger = setup_logger()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


class MultiAgentState(TypedDict):
    """Shared state across all agents in the workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tenant_id: str
    user_role: str
    user_id: Optional[str]
    email: Optional[str]
    
    intent_result: Optional[dict]
    kb_docs: list
    report: Optional[dict]
    verification: Optional[dict]
    
    final_answer: Optional[str]


def create_multi_agent_graph(tenant_id: str = "default", user_role: str = "customer", user_id: Optional[str] = None):
    """
    Create and compile multi-agent graph for BFSI claim verification workflow
    
    Args:
        tenant_id: Tenant identifier for multi-tenancy
        user_role: User role for RBAC
        user_id: Optional user ID for data lookup
        
    Returns:
        Compiled LangGraph instance
    """
    config = get_config()
    
    # Initialize LLM
    model_config = config.get_section('model')
    base_llm = init_chat_model(
        model_config.get('name', 'gemini-2.0-flash'),
        model_provider=model_config.get('provider', 'google_genai')
    )
    
    # Initialize agents
    intent_agent = IntentGathererAgent(
        llm=base_llm,
        tenant_id=tenant_id,
        user_role=user_role
    )
    
    answer_agent = AnswerGeneratorAgent(llm=base_llm)
    report_agent = ReportMakerAgent(llm=base_llm)
    
    # Get all tools for claim verifier (retriever, jira, get_user_data, list_user_policies)
    tools = get_all_tools(tenant_id=tenant_id, user_role=user_role)
    tools_dict = {tool.name: tool for tool in tools}
    
    verifier_agent = ClaimVerifierAgent(
        llm=base_llm,
        tools=tools,
        tenant_id=tenant_id,
        user_role=user_role
    )
    
    # Node functions
    def intent_node(state: MultiAgentState) -> dict:
        logger.info("=== Intent Gatherer Node ===")
        result = intent_agent.process(state)
        return result
    
    def answer_node(state: MultiAgentState) -> dict:
        logger.info("=== Answer Generator Node ===")
        result = answer_agent.process(state)
        return result
    
    def report_node(state: MultiAgentState) -> dict:
        logger.info("=== Report Maker Node ===")
        result = report_agent.process(state)
        return result
    
    def verify_node(state: MultiAgentState) -> dict:
        logger.info("=== Claim Verifier Node ===")
        result = verifier_agent.process(state)
        return result
    
    def tool_execution_node(state: MultiAgentState) -> dict:
        """Execute tool calls from the verifier agent"""
        logger.info("=== Tool Execution Node ===")
        
        messages = state['messages']
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("No tool calls found in last message")
            return {'messages': []}
        
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            if tool_name not in tools_dict:
                error_msg = f"Tool {tool_name} not found in available tools"
                logger.error(error_msg)
                tool_results.append({
                    'tool_call_id': tool_id,
                    'name': tool_name,
                    'content': error_msg
                })
                continue
            
            try:
                result = tools_dict[tool_name].invoke(tool_args)
                logger.info(f"Tool {tool_name} executed successfully")
                tool_results.append({
                    'tool_call_id': tool_id,
                    'name': tool_name,
                    'content': str(result)
                })
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                logger.error(f"Tool {tool_name} failed: {e}")
                tool_results.append({
                    'tool_call_id': tool_id,
                    'name': tool_name,
                    'content': error_msg
                })
        
        # Import ToolMessage for tool results
        from langchain_core.messages import ToolMessage
        tool_messages = [ToolMessage(**tr) for tr in tool_results]
        
        return {'messages': tool_messages}
    
    def out_of_scope_node(state: MultiAgentState) -> dict:
        """Handle out-of-scope queries with hardcoded response"""
        logger.info("=== Out of Scope Handler ===")
        message = "I'm sorry, but your query appears to be outside the scope of BFSI (Banking, Financial Services, Insurance) services. I can only assist with questions related to insurance policies, claims, banking services, loans, credit cards, investments, and other financial products. Please try again with a query related to these topics."
        return {
            'final_answer': message
        }
    
    # Routing functions
    def route_after_intent(state: MultiAgentState) -> str:
        """Route based on intent classification"""
        intent_result = state.get('intent_result')
        
        if not intent_result:
            logger.warning("No intent result, sending to answer generator")
            return "answer"
        
        # Check for out-of-scope first - skip answer generator
        if intent_result.get('out_of_scope'):
            logger.info("Query is out of scope, routing to out-of-scope handler")
            return "out_of_scope"
        
        # Check for query intent
        if intent_result.get('intent') == 'query':
            logger.info("Routing query intent to answer generator")
            return "answer"
        
        # All other intents (complaint, service_request, feature_request) go to report maker
        logger.info(f"Intent: {intent_result.get('intent')}, routing to report maker")
        return "report"
    
    def route_after_verify(state: MultiAgentState) -> str:
        """Route based on whether verifier made tool calls"""
        messages = state.get('messages', [])
        
        if not messages:
            logger.warning("No messages in state, ending workflow")
            return "end"
        
        last_message = messages[-1]
        
        # Check if last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"Verifier made {len(last_message.tool_calls)} tool call(s), routing to tool execution")
            return "tools"
        
        # No tool calls, workflow complete
        logger.info("No tool calls from verifier, ending workflow")
        return "end"
    
    # Build graph
    graph = StateGraph(MultiAgentState)
    
    # Add nodes
    graph.add_node("intent", intent_node)
    graph.add_node("answer", answer_node)
    graph.add_node("report", report_node)
    graph.add_node("verify", verify_node)
    graph.add_node("tools", tool_execution_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    
    # Set entry point
    graph.set_entry_point("intent")
    
    # Add edges
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "answer": "answer",
            "report": "report",
            "out_of_scope": "out_of_scope",
            "end": END
        }
    )
    
    graph.add_edge("answer", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("report", "verify")
    
    # Conditional routing from verify: either execute tools or end
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tool execution, route back to verify for decision making
    graph.add_edge("tools", "verify")
    
    compiled_graph = graph.compile()
    logger.info("Multi-agent graph compiled successfully")
    return compiled_graph

def invoke_graph(query: str, tenant_id: str = "default", user_role: str = "customer", user_id: Optional[str] = None, email: Optional[str] = None) -> dict:
    """
    Invoke the multi-agent graph with a user query
    
    Args:
        query: User query string
        tenant_id: Tenant identifier
        user_role: User role
        user_id: Optional user ID
        email: Optional user email
        
    Returns:
        Final state dictionary
    """
    graph = create_multi_agent_graph(tenant_id, user_role, user_id)
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "tenant_id": tenant_id,
        "user_role": user_role,
        "user_id": user_id,
        "email": email,
        "intent_result": None,
        "kb_docs": [],
        "report": None,
        "verification": None,
        "final_answer": None
    }
    
    result = graph.invoke(initial_state)
    return result

