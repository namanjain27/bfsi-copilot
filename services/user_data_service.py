"""
SQLite Database Service for BFSI Multi-Agent Workflow
Manages user data, policies, and incidents for claim verification
Using SQLAlchemy ORM and Pydantic for best practices
"""

import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from pydantic import ValidationError

# Import database models and session management
from services.database import get_db_session, User, Policy, Incident
from services.models import (
    UserCreate, UserInDB,
    PolicyCreate, PolicyInDB,
    IncidentCreate, IncidentInDB, IncidentUpdate
)


# ========== TOOL FUNCTIONS (LangChain decorated) ==========

@tool
def get_user_data(user_id: Optional[str] = None, email: Optional[str] = None) -> dict:
    """
    Retrieve user data from the database.
    
    Args:
        user_id: User ID to search for (optional)
        email: Email address to search for (optional)
    
    Returns:
        Dictionary containing user data with keys: user_id, name, customer_id, age, email, date_registered
        Returns empty dict if no user found
    """
    if not user_id and not email:
        return {"error": "Must provide either user_id or email"}
    
    with get_db_session() as session:
        if user_id:
            user = session.query(User).filter(User.user_id == user_id).first()
        else:
            user = session.query(User).filter(User.email == email).first()
        
        if user:
            return UserInDB.from_orm(user).dict()
        return {}


@tool
def list_user_policies(user_id: str) -> list:
    """
    List all policies for a given user.
    
    Args:
        user_id: User ID to get policies for
    
    Returns:
        List of dictionaries, each containing policy data with keys:
        policy_number, user_id, policy_name, item_insured, start_date, end_date,
        billing_duration, last_payment_date, last_payment_amount
    """
    with get_db_session() as session:
        policies = session.query(Policy).filter(Policy.user_id == user_id).all()
        return [PolicyInDB.from_orm(policy).dict() for policy in policies]


@tool
def create_incident_record(incident_data: dict) -> str:
    """
    Create a new incident record in the database.
    
    Args:
        incident_data: Dictionary containing incident information with keys:
            - user_id (optional)
            - tenant_id (optional)
            - intent (optional)
            - summary (optional)
            - issue (optional)
            - user_demand (optional)
            - is_valid (optional)
            - resolution (optional)
            - confidence (optional)
            - status (optional, defaults to 'open')
            - policy_refs (optional, list that will be stored as JSON)
            - decision_json (optional, dict that will be stored as JSON)
    
    Returns:
        String containing the newly created incident_id
    """
    try:
        # Validate input data with Pydantic
        incident_create = IncidentCreate(**incident_data)
    except ValidationError as e:
        raise ValueError(f"Invalid incident data: {e}")
    
    incident_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    with get_db_session() as session:
        # Create new incident
        new_incident = Incident(
            incident_id=incident_id,
            user_id=incident_create.user_id,
            tenant_id=incident_create.tenant_id,
            intent=incident_create.intent,
            summary=incident_create.summary,
            issue=incident_create.issue,
            user_demand=incident_create.user_demand,
            is_valid=incident_create.is_valid,
            resolution=incident_create.resolution,
            confidence=incident_create.confidence,
            status=incident_create.status,
            policy_refs=json.dumps(incident_create.policy_refs),
            decision_json=json.dumps(incident_create.decision_json),
            created_at=timestamp,
            updated_at=timestamp
        )
        
        session.add(new_incident)
        # Session commits automatically via context manager
    
    return incident_id


@tool
def update_incident_status(
    incident_id: str, 
    status: str, 
    resolution: Optional[str] = None
) -> bool:
    """
    Update the status and optionally the resolution of an incident.
    
    Args:
        incident_id: ID of the incident to update
        status: New status value (e.g., 'open', 'in_progress', 'resolved', 'closed')
        resolution: Optional resolution text to update
    
    Returns:
        True if update successful, False if incident not found
    """
    try:
        # Validate status with Pydantic
        update_data = {"status": status}
        if resolution:
            update_data["resolution"] = resolution
        IncidentUpdate(**update_data)
    except ValidationError as e:
        raise ValueError(f"Invalid update data: {e}")
    
    timestamp = datetime.now().isoformat()
    
    with get_db_session() as session:
        incident = session.query(Incident).filter(Incident.incident_id == incident_id).first()
        
        if not incident:
            return False
        
        incident.status = status
        if resolution:
            incident.resolution = resolution
        incident.updated_at = timestamp
        
        # Session commits automatically via context manager
    
    return True


# ========== CRUD HELPER FUNCTIONS (Non-tool) ==========

def insert_user(user_data: Dict[str, Any]) -> str:
    """
    Insert a new user into the database.
    
    Args:
        user_data: Dictionary with keys: user_id, name, customer_id, age, email, date_registered
    
    Returns:
        user_id of the inserted user
    """
    try:
        # Add default date_registered if not provided
        if 'date_registered' not in user_data:
            user_data['date_registered'] = datetime.now().strftime('%Y-%m-%d')
        
        # Validate with Pydantic
        user_create = UserCreate(**user_data)
    except ValidationError as e:
        raise ValueError(f"Invalid user data: {e}")
    
    with get_db_session() as session:
        new_user = User(
            user_id=user_create.user_id,
            name=user_create.name,
            customer_id=user_create.customer_id,
            age=user_create.age,
            email=user_create.email,
            date_registered=user_create.date_registered or datetime.now().strftime('%Y-%m-%d')
        )
        
        session.add(new_user)
        # Session commits automatically via context manager
    
    return user_create.user_id


def insert_policy(policy_data: Dict[str, Any]) -> str:
    """
    Insert a new policy into the database.
    
    Args:
        policy_data: Dictionary with keys: policy_number, user_id, policy_name, item_insured,
                     start_date, end_date, billing_duration, last_payment_date, last_payment_amount
    
    Returns:
        policy_number of the inserted policy
    """
    try:
        # Validate with Pydantic
        policy_create = PolicyCreate(**policy_data)
    except ValidationError as e:
        raise ValueError(f"Invalid policy data: {e}")
    
    with get_db_session() as session:
        new_policy = Policy(
            policy_number=policy_create.policy_number,
            user_id=policy_create.user_id,
            policy_name=policy_create.policy_name,
            item_insured=policy_create.item_insured,
            start_date=policy_create.start_date,
            end_date=policy_create.end_date,
            billing_duration=policy_create.billing_duration,
            last_payment_date=policy_create.last_payment_date,
            last_payment_amount=policy_create.last_payment_amount
        )
        
        session.add(new_policy)
        # Session commits automatically via context manager
    
    return policy_create.policy_number


def get_incident_by_id(incident_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific incident by ID.
    
    Args:
        incident_id: ID of the incident to retrieve
    
    Returns:
        Dictionary containing incident data, or None if not found
    """
    with get_db_session() as session:
        incident = session.query(Incident).filter(Incident.incident_id == incident_id).first()
        
        if incident:
            incident_dict = IncidentInDB.from_orm(incident).dict()
            # Parse JSON fields back to objects
            incident_dict['policy_refs'] = json.loads(incident.policy_refs) if incident.policy_refs else []
            incident_dict['decision_json'] = json.loads(incident.decision_json) if incident.decision_json else {}
            return incident_dict
        
        return None


def list_incidents(user_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List incidents with optional filtering.
    
    Args:
        user_id: Filter by user_id (optional)
        status: Filter by status (optional)
    
    Returns:
        List of incident dictionaries
    """
    with get_db_session() as session:
        query = session.query(Incident)
        
        if user_id:
            query = query.filter(Incident.user_id == user_id)
        
        if status:
            query = query.filter(Incident.status == status)
        
        query = query.order_by(Incident.created_at.desc())
        incidents = query.all()
        
        result = []
        for incident in incidents:
            incident_dict = IncidentInDB.from_orm(incident).dict()
            # Parse JSON fields
            incident_dict['policy_refs'] = json.loads(incident.policy_refs) if incident.policy_refs else []
            incident_dict['decision_json'] = json.loads(incident.decision_json) if incident.decision_json else {}
            result.append(incident_dict)
        
        return result


def delete_user(user_id: str) -> bool:
    """
    Delete a user from the database.
    Note: This will cascade delete all associated policies and incidents.
    
    Args:
        user_id: ID of user to delete
    
    Returns:
        True if deletion successful, False otherwise
    """
    with get_db_session() as session:
        user = session.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return False
        
        session.delete(user)
        # Session commits automatically via context manager
    
    return True


def delete_policy(policy_number: str) -> bool:
    """
    Delete a policy from the database.
    
    Args:
        policy_number: Policy number to delete
    
    Returns:
        True if deletion successful, False otherwise
    """
    with get_db_session() as session:
        policy = session.query(Policy).filter(Policy.policy_number == policy_number).first()
        
        if not policy:
            return False
        
        session.delete(policy)
        # Session commits automatically via context manager
    
    return True
