"""
Agent State & Contract Schemas for Multi-Agent Workflow
Defines Pydantic models for type safety and validation across agents
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime


class IntentResult(BaseModel):
    """Result from intent classification and query analysis"""
    intent: Literal['query', 'complaint', 'service_request', 'feature_request']
    urgency: Literal['high', 'medium', 'low']
    sentiment: Literal['positive', 'neutral', 'negative']
    aspects: List[str] = Field(
        description="Sub-topics or facets of the query for multi-search. "
                    "Single element for simple queries, multiple for complex queries."
    )
    out_of_scope: bool = Field(
        default=False,
        description="True if query is not related to BFSI services/policies"
    )


class KBDocument(BaseModel):
    """Knowledge base document with scoring metadata"""
    content: str = Field(description="Document content/chunk text")
    score: float = Field(description="Combined relevance score (0-1)")
    source: str = Field(description="Source file path or identifier")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True


class Report(BaseModel):
    """Structured report from report maker agent"""
    issue: str = Field(description="Clear statement of the problem")
    user_demand: str = Field(description="What user is requesting")
    company_docs_about_issue: str = Field(
        description="Rephrased and restructured relevant information from company documents"
    )
    support_info_from_user: str = Field(
        description="Evidence/context provided by user"
    )
    policy_refs: str = Field(
        description="Referenced policy numbers or sections"
    )


class ActionPlan(BaseModel):
    """Action plan for claim verifier decisions"""
    create_ticket: bool = Field(
        default=False,
        description="Whether to create a Jira ticket for human review"
    )
    ticket_type: Optional[Literal['complaint', 'service_request', 'feature_request']] = Field(
        default=None,
        description="Type of ticket to create if create_ticket is True"
    )
    make_db_entry: bool = Field(
        default=True,
        description="Whether to create incident record in database"
    )
    call_refund_api: bool = Field(
        default=False,
        description="Whether to initiate refund process"
    )
    refund_amount: Optional[float] = Field(
        default=None,
        description="Refund amount if call_refund_api is True"
    )
    idempotency_key: str = Field(
        default="",
        description="Unique key to prevent duplicate operations"
    )


class VerificationDecision(BaseModel):
    """Decision from claim verifier agent"""
    is_valid: Literal['Yes', 'Low Confidence', 'No']
    resolution: str = Field(
        description="Explanation of decision with policy citations"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level in decision (0.0-1.0)"
    )
    policy_citations: List[str] = Field(
        default_factory=list,
        description="List of policy sections/documents cited"
    )
    action_plan: Optional[ActionPlan] = Field(
        default=None,
        description="Actions to execute if decision requires them"
    )

class MultiAgentState(BaseModel):
    """
    Shared state for multi-agent workflow (LangGraph compatible)
    Note: For actual LangGraph usage, convert to TypedDict
    """
    # Context
    tenant_id: str
    user_role: str
    user_id: Optional[str] = None
    
    # Agent outputs
    intent_result: Optional[IntentResult] = None
    kb_docs: List[KBDocument] = Field(default_factory=list)
    report: Optional[Report] = None
    verification: Optional[VerificationDecision] = None
    
    # Actions
    incident_id: Optional[str] = None
    ticket_id: Optional[str] = None
    actions_completed: dict = Field(default_factory=dict)
    
    # Final response
    final_answer: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
