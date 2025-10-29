"""
SQLAlchemy Database Models and Session Management
ORM models for BFSI Multi-Agent Workflow
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pathlib import Path
from contextlib import contextmanager

# Database file path
DB_PATH = Path(__file__).parent.parent / "echopilot.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True for SQL query debugging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative models
Base = declarative_base()


# ========== ORM MODELS ==========

class User(Base):
    """User table - stores customer information"""
    __tablename__ = "users"
    
    user_id = Column(String(100), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    customer_id = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    date_registered = Column(String(50), nullable=False)
    
    # Relationships
    policies = relationship("Policy", back_populates="user", cascade="all, delete-orphan")
    incidents = relationship("Incident", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', name='{self.name}', email='{self.email}')>"


class Policy(Base):
    """Policy table - stores insurance policies"""
    __tablename__ = "policies"
    
    policy_number = Column(String(100), primary_key=True, index=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=False, index=True)
    policy_name = Column(String(200), nullable=False)
    item_insured = Column(String(500), nullable=True)
    start_date = Column(String(50), nullable=False)
    end_date = Column(String(50), nullable=False)
    billing_duration = Column(String(50), nullable=True)
    last_payment_date = Column(String(50), nullable=True)
    last_payment_amount = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="policies")
    
    def __repr__(self):
        return f"<Policy(policy_number='{self.policy_number}', user_id='{self.user_id}', policy_name='{self.policy_name}')>"


class Incident(Base):
    """Incident table - stores claim incidents and verifications"""
    __tablename__ = "incidents"
    
    incident_id = Column(String(100), primary_key=True, index=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=True, index=True)
    tenant_id = Column(String(100), nullable=True, index=True)
    intent = Column(String(200), nullable=True)
    summary = Column(Text, nullable=True)
    issue = Column(Text, nullable=True)
    user_demand = Column(Text, nullable=True)
    is_valid = Column(Boolean, nullable=True)
    resolution = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String(50), nullable=False, default='open', index=True)
    policy_refs = Column(Text, nullable=True)  # JSON string
    decision_json = Column(Text, nullable=True)  # JSON string
    created_at = Column(String(50), nullable=False)
    updated_at = Column(String(50), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="incidents")
    
    def __repr__(self):
        return f"<Incident(incident_id='{self.incident_id}', user_id='{self.user_id}', status='{self.status}')>"


# ========== DATABASE INITIALIZATION ==========

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


# ========== SESSION MANAGEMENT ==========

@contextmanager
def get_db_session() -> Session:
    """
    Context manager for database sessions.
    Ensures proper session cleanup and transaction handling.
    
    Usage:
        with get_db_session() as session:
            user = session.query(User).filter_by(user_id="123").first()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """
    Dependency function for getting database sessions.
    Used in FastAPI endpoints or other dependency injection scenarios.
    
    Usage:
        db = next(get_db())
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database tables on module import
init_db()

