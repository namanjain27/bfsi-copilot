"""
Pydantic Models for BFSI Multi-Agent Workflow
Data validation and serialization models for users, policies, and incidents
"""

from pydantic import BaseModel, EmailStr, Field, validator
from datetime import datetime
from typing import Optional, List, Dict, Any


# ========== USER MODELS ==========

class UserBase(BaseModel):
    """Base user model with common fields"""
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    customer_id: Optional[str] = Field(None, max_length=100)
    age: Optional[int] = Field(None, gt=0, lt=150)


class UserCreate(UserBase):
    """Model for creating a new user"""
    user_id: str = Field(..., min_length=1, max_length=100)
    date_registered: Optional[str] = None
    
    @validator('date_registered')
    def validate_date_format(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('date_registered must be a valid ISO format date string')
        return v


class UserInDB(UserBase):
    """User model as stored in database"""
    user_id: str
    date_registered: str
    
    class Config:
        from_attributes = True  # Allows SQLAlchemy model conversion


# ========== POLICY MODELS ==========

class PolicyBase(BaseModel):
    """Base policy model with common fields"""
    policy_name: str = Field(..., min_length=1, max_length=200)
    item_insured: Optional[str] = Field(None, max_length=500)
    start_date: str
    end_date: str
    billing_duration: Optional[str] = Field(None, max_length=50)
    last_payment_date: Optional[str] = None
    last_payment_amount: Optional[float] = Field(None, ge=0)
    
    @validator('start_date', 'end_date', 'last_payment_date')
    def validate_date_format(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Date must be a valid ISO format date string')
        return v


class PolicyCreate(PolicyBase):
    """Model for creating a new policy"""
    policy_number: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)


class PolicyInDB(PolicyBase):
    """Policy model as stored in database"""
    policy_number: str
    user_id: str
    
    class Config:
        from_attributes = True


# ========== INCIDENT MODELS ==========

class IncidentBase(BaseModel):
    """Base incident model with common fields"""
    user_id: Optional[str] = Field(None, max_length=100)
    tenant_id: Optional[str] = Field(None, max_length=100)
    intent: Optional[str] = Field(None, max_length=200)
    summary: Optional[str] = None
    issue: Optional[str] = None
    user_demand: Optional[str] = None
    is_valid: Optional[bool] = None
    resolution: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    status: str = Field(default="open", max_length=50)
    policy_refs: List[str] = Field(default_factory=list)
    decision_json: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['open', 'in_progress', 'resolved', 'closed']
        if v not in allowed_statuses:
            raise ValueError(f'Status must be one of {allowed_statuses}')
        return v


class IncidentCreate(IncidentBase):
    """Model for creating a new incident"""
    pass


class IncidentUpdate(BaseModel):
    """Model for updating an incident"""
    status: Optional[str] = Field(None, max_length=50)
    resolution: Optional[str] = None
    is_valid: Optional[bool] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    decision_json: Optional[Dict[str, Any]] = None
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            allowed_statuses = ['open', 'in_progress', 'resolved', 'closed']
            if v not in allowed_statuses:
                raise ValueError(f'Status must be one of {allowed_statuses}')
        return v


class IncidentInDB(IncidentBase):
    """Incident model as stored in database"""
    incident_id: str
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True

