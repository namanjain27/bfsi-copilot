"""
Claim Verifier Agent
Verifies claims and makes resolution decisions for BFSI workflows
"""
import json
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from services.agent_schemas import VerificationDecision, ActionPlan, Report
from services.logger_setup import setup_logger

from services.config_loader import get_config

logger = setup_logger()
config = get_config()

class ClaimVerifierAgent:
    """
    Agent responsible for claim verification and resolution decision-making.
    
    Key principles:
    - Verify claims using company policies (from report + optional RAG)
    - Prioritize company policy over user demand
    - Call tools (RAG, get_user_data) as needed for informed decisions
    - Return structured decision (JSON) - execution happens separately
    - Low confidence (<0.7) triggers human escalation
    """
    
    def __init__(self, llm, tools, tenant_id: str, user_role: str):
        self.llm_with_tools = llm.bind_tools(tools)
        self.structured_llm = llm.with_structured_output(VerificationDecision)
        self.base_llm = llm
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.tenant_id = tenant_id
        self.user_role = user_role
        
        # Get thresholds from config
        self.min_confidence = config.get('multi_agent', {}).get('thresholds', {}).get('min_confidence', 0.7)
        self.max_refund_amount = config.get('multi_agent', {}).get('decision', {}).get('max_refund_amount', 50000)
        
    def process(self, state: dict) -> dict:
        """
        Verify claim and make resolution decision
        
        Args:
            state: Agent state containing:
                - report: Report object from report_maker
                - intent_result: Intent analysis
                - user_id: Optional user ID for data lookup
                - email: Optional user email
                - messages: Chat history (includes tool results if any)
                
        Returns:
            Updated state with verification decision or tool calls
        """
        logger.info("ClaimVerifierAgent: Starting claim verification")
        
        report_data = state.get('report')
        if not report_data:
            logger.error("No report found in state")
            return {'verification': self._create_fallback_decision(None, None).dict()}
        
        report = Report(**report_data) if isinstance(report_data, dict) else report_data
        intent_result = state.get('intent_result')
        user_id = state.get('user_id')
        email = state.get('email')
        messages = state.get('messages', [])
        
        # Check if we have tool results in messages (i.e., tools were just executed)
        has_tool_results = any(msg.__class__.__name__ == 'ToolMessage' for msg in messages)
        
        if has_tool_results:
            # Tools were executed, now make final decision with all information
            logger.info("Tool results available, making final decision")
            decision = self._make_final_decision(report, intent_result, messages)
            
            logger.info(f"Final Decision: valid={decision.is_valid}, confidence={decision.confidence:.2f}")
            
            return {
                'verification': decision.dict()
            }
        else:
            # First pass: check if we need tools
            logger.info("First pass: checking if tools are needed")
            response = self._check_and_call_tools(report, intent_result, user_id, email, messages)
            
            # If response contains tool_calls, return it for execution
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"Agent requesting {len(response.tool_calls)} tool call(s)")
                return {'messages': [response]}
            
            # No tools needed, make decision directly
            logger.info("No tools needed, making direct decision")
            decision = self._make_final_decision(report, intent_result, messages)
            
            logger.info(f"Decision: valid={decision.is_valid}, confidence={decision.confidence:.2f}")
            
            return {
                'verification': decision.dict()
            }
    
#     def _execute_tool_calls(self, report: Report, intent_result, user_id: str, messages: List) -> dict:
#         """
#         Execute tool calls if LLM decides it needs additional information
        
#         Args:
#             report: Structured report from report_maker
#             intent_result: Intent classification
#             user_id: User ID for data lookup
#             messages: Chat history
            
#         Returns:
#             State dict with tool call results
#         """
#         logger.info("Checking if additional information needed via tools")
        
#         # Build context for tool calling decision
#         context = self._format_verification_context(report, intent_result, user_id)
        
#         system_prompt = f"""You are a claim verification assistant for a BFSI organization.

# **Context:**
# {context}

# **Available Tools:**
# - get_user_data: Fetch user account information (user_id or email)
# - list_user_policies: Get user's active policies

# **Your Task:**
# Review the report above. Decide if you need additional information to verify this claim.
# - The report already contains policy information from the knowledge base
# - If you need to verify user eligibility, policy status, or account details, call get_user_data or list_user_policies
# - Only call tools if truly needed for verification

# If you have enough information, respond with "I have sufficient information to proceed with verification."
# """
        
#         tool_results = {}
#         conversation = [SystemMessage(content=system_prompt)] + list(messages) 
        
#         # Allow up to 3 tool call rounds
#         max_rounds = 3
#         for round_num in range(max_rounds):
#             try:
#                 response = self.llm_with_tools.invoke(conversation)
#                 conversation.append(response)
                
#                 # Check if LLM made tool calls
#                 if hasattr(response, 'tool_calls') and response.tool_calls:
#                     logger.info(f"Round {round_num + 1}: LLM making {len(response.tool_calls)} tool call(s)")
                    
#                     for tool_call in response.tool_calls:
#                         tool_name = tool_call['name']
#                         tool_args = tool_call['args']
#                         tool_id = tool_call['id']
                        
#                         logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                        
#                         # Execute the tool
#                         if tool_name in self.tools:
#                             try:
#                                 result = self.tools[tool_name].invoke(tool_args)
#                                 tool_results[tool_name] = result
                                
#                                 # Add tool result to conversation
#                                 conversation.append(ToolMessage(
#                                     content=str(result),
#                                     tool_call_id=tool_id
#                                 ))
                                
#                                 logger.info(f"Tool {tool_name} returned: {str(result)[:100]}...")
#                             except Exception as e:
#                                 logger.error(f"Tool {tool_name} failed: {e}")
#                                 tool_results[tool_name] = {"error": str(e)}
#                                 conversation.append(ToolMessage(
#                                     content=f"Error: {str(e)}",
#                                     tool_call_id=tool_id
#                                 ))
#                         else:
#                             logger.warning(f"Tool {tool_name} not found in available tools")
#                 else:
#                     # No tool calls, LLM has enough information
#                     logger.info("LLM has sufficient information, no more tool calls needed")
#                     break
                    
#             except Exception as e:
#                 logger.error(f"Error during tool calling round {round_num + 1}: {e}")
#                 break
        
#         return {
#             'tool_results': tool_results,
#             'final_context': conversation
#         }
    
    def _check_and_call_tools(self, report: Report, intent_result, user_id: str, email: str, messages: List):
        """
        Check if tools are needed and make tool calls if necessary
        
        Args:
            report: Structured report
            intent_result: Intent classification
            user_id: User ID
            email: User email
            messages: Chat history
            
        Returns:
            AIMessage with tool_calls if tools needed, or AIMessage without tool_calls
        """
        context = self._format_decision_context(report, intent_result)
        
        system_prompt = """You are a BFSI claim verification assistant.

Review the report and determine if you need additional information to make a proper verification decision.

**Available Tools:**
1. **retriever_tool** - Search organization's knowledge base for policies, procedures, and documents
2. **get_user_data** - Fetch user account information (requires user_id or email)
3. **list_user_policies** - Get user's active policies (requires user_id)
4. **create_jira_ticket** - Create ticket for human review (use only after decision is made)

**Context Available:**
- user_id: {user_id}
- email: {email}
- Report already contains policy information from intent gatherer's retrieval

**Your Task:**
Decide if you need MORE information to verify the claim:
- If report lacks sufficient policy details → call retriever_tool with specific query
- If need to verify user eligibility, account status, or policy details → call get_user_data or list_user_policies
- If information seems sufficient → respond with "I have enough information to make a decision"

Only call tools if truly needed. The report may already contain everything required.""".format(
            user_id=user_id or "Not provided",
            email=email or "Not provided"
        )
        
        user_prompt = f"""Please review this information and decide if you need additional tools:

{context}

Do you need to call any tools, or do you have sufficient information?"""
        
        try:
            response = self.llm_with_tools.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            return response
            
        except Exception as e:
            logger.error(f"Error checking for tool needs: {e}")
            # Return empty response, will proceed to direct decision
            return AIMessage(content="Proceeding with available information")
    
    def _make_final_decision(self, report: Report, intent_result, messages: List) -> VerificationDecision:
        logger.info("Making final verification decision")
        
        # Build comprehensive context including tool results
        context = self._format_decision_context(report, intent_result, messages)
        # TODO: in any case, make a db entry into the incident table for record keeping purposes. Once we have required fields from the response json of verifier agent then we will call this ourselves.
        system_prompt = """You are a BFSI claim verification decision maker.

Analyze the issue and all information provided. Make a final verification decision with a structured response.

**Decision Process:**
1. Verify if the issue is valid according to company policies and supporting documents
2. Calculate confidence score (0.0 to 1.0) based on:
   - Policy clarity and completeness
   - Information sufficiency
   - Alignment between user demand and policy
3. Determine appropriate resolution and action plan

**CRITICAL - Exact Field Values (Pydantic Schema):**

1. **is_valid** - MUST be EXACTLY one of these strings:
   - "Yes" - Issue is valid according to policies
   - "Low Confidence" - Uncertain, needs human review
   - "No" - Issue is not valid according to policies

2. **action_plan.ticket_type** - If creating ticket, MUST be EXACTLY one of:
   - "complaint" - For complaints
   - "service_request" - For service requests
   - "feature_request" - For feature requests
   DO NOT use descriptive strings like "Complaint - Insurance" or "PaymentDiscrepancy"
   Use the intent from the report to pick the right literal value

**Action Planning:**
- If confidence < 0.7: MUST set create_ticket=true, is_valid="Low Confidence"
- If valid AND confident: is_valid="Yes", provide clear resolution with policy citations
- If valid but no direct resolution available: is_valid="Yes", explain + create ticket
- If invalid: is_valid="No", explain reasoning with policy citations

**Important Rules:**
- NEVER call tools now - all needed information should already be gathered
- Resolution should directly address the user's issue
- Policy citations should reference specific sections when available
- If refund is appropriate, specify amount (if determinable)
- Ticket creation is for escalation/tracking, not data gathering

Return structured decision with: is_valid, confidence, resolution, policy_citations, and action_plan."""

        user_prompt = f"""Please analyze the following information and make a verification decision.

REMEMBER: Use exact literal values from the schema above:
- is_valid: "Yes", "Low Confidence", or "No"
- ticket_type: "complaint", "service_request", or "feature_request" (match the Intent from context)

{context}

Make your decision now with the correct literal string values."""

        try:
            # Use structured LLM (no tools) for final decision
            decision_extraction = self.structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # Apply safety checks
            confidence = min(max(decision_extraction.confidence, 0.0), 1.0)
            
            # Validate and fix is_valid if needed
            is_valid = decision_extraction.is_valid
            valid_is_valid_values = ['Yes', 'Low Confidence', 'No']
            
            if is_valid not in valid_is_valid_values:
                # Try to map from boolean or other values
                if isinstance(is_valid, bool):
                    is_valid = 'Yes' if is_valid else 'No'
                    logger.warning(f"Fixed boolean is_valid to string: '{is_valid}'")
                elif str(is_valid).lower() in ['true', 'yes', 'valid']:
                    is_valid = 'Yes'
                    logger.warning(f"Fixed invalid is_valid '{decision_extraction.is_valid}' to 'Yes'")
                elif str(is_valid).lower() in ['false', 'no', 'invalid']:
                    is_valid = 'No'
                    logger.warning(f"Fixed invalid is_valid '{decision_extraction.is_valid}' to 'No'")
                else:
                    is_valid = 'Low Confidence'
                    logger.warning(f"Fixed invalid is_valid '{decision_extraction.is_valid}' to 'Low Confidence'")
            
            # Create action plan with safety checks
            action_plan_data = decision_extraction.action_plan.model_dump()
            
            # Force ticket creation if confidence too low
            if confidence < self.min_confidence:
                action_plan_data['create_ticket'] = True
                logger.info(f"Confidence {confidence} below threshold {self.min_confidence}, forcing ticket creation")
            
            # Validate and fix ticket_type if needed
            if action_plan_data.get('create_ticket') and action_plan_data.get('ticket_type'):
                ticket_type = action_plan_data['ticket_type']
                valid_types = ['complaint', 'service_request', 'feature_request']
                
                if ticket_type not in valid_types:
                    # Try to map from intent or use best guess
                    intent = intent_result.get('intent') if intent_result else None
                    if intent in valid_types:
                        action_plan_data['ticket_type'] = intent
                        logger.warning(f"Fixed invalid ticket_type '{ticket_type}' to '{intent}' based on intent")
                    else:
                        action_plan_data['ticket_type'] = 'service_request'
                        logger.warning(f"Fixed invalid ticket_type '{ticket_type}' to 'service_request' (default)")
            
            # Ensure idempotency key exists
            # TODO: idempotency_key should be coded and not dependent on LLM
            if not action_plan_data.get('idempotency_key'):
                action_plan_data['idempotency_key'] = str(uuid.uuid4())
            
            action_plan = ActionPlan(**action_plan_data)
            
            decision = VerificationDecision(
                is_valid=is_valid,  # Use validated is_valid
                resolution=decision_extraction.resolution,
                confidence=confidence,
                policy_citations=decision_extraction.policy_citations,
                action_plan=action_plan
            )
            
            logger.info(f"Successfully created decision: valid={decision.is_valid}, confidence={decision.confidence:.2f}")
            return decision
            
        except Exception as e:
            logger.error(f"Error in verification decision: {e}")
            return self._create_fallback_decision(report, intent_result)
    
    def _format_decision_context(self, report: Report, intent_result, messages: List = None) -> str:
        """Format comprehensive context for final decision including tool results"""
        context_parts = []
        
        context_parts.append("=== INTENT & CLASSIFICATION ===")
        if intent_result:
            context_parts.append(f"Intent: {intent_result.get('intent', 'Unknown')}, Urgency: {intent_result.get('urgency', 'Unknown')}, Sentiment: {intent_result.get('sentiment', 'Unknown')}")
        
        context_parts.append(f"\n=== STRUCTURED REPORT ===\n{report.model_dump_json(indent=2)}")
        
        # Include tool results if available
        if messages:
            tool_results = [msg for msg in messages if msg.__class__.__name__ == 'ToolMessage']
            if tool_results:
                context_parts.append("\n=== TOOL RESULTS ===")
                for i, tool_msg in enumerate(tool_results, 1):
                    tool_name = getattr(tool_msg, 'name', 'Unknown Tool')
                    context_parts.append(f"\nTool {i} ({tool_name}):\n{tool_msg.content}")
        
        return "\n".join(context_parts)
    
    def _create_fallback_decision(self, report: Report, intent_result) -> VerificationDecision:
        """Create fallback decision when LLM decision fails"""
        logger.warning("Using fallback decision creation")
        
        # Map intent to valid ticket_type literal
        intent = intent_result.get('intent') if intent_result else 'service_request'
        if intent == 'query':
            ticket_type = 'service_request'
        elif intent in ['complaint', 'service_request', 'feature_request']:
            ticket_type = intent
        else:
            ticket_type = 'service_request'
        
        # Conservative fallback: low confidence, create ticket
        return VerificationDecision(
            is_valid='Low Confidence',  # Must be string literal: 'Yes', 'Low Confidence', or 'No'
            resolution="Unable to verify claim due to processing error. A ticket has been created for manual review.",
            confidence=0.3,
            policy_citations=[],
            action_plan=ActionPlan(
                create_ticket=True,
                ticket_type=ticket_type,
                make_db_entry=True,
                call_refund_api=False,
                refund_amount=None,
                idempotency_key=str(uuid.uuid4())
            )
        )

