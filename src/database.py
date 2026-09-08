from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

load_dotenv()  

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Configure engine based on database type
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL configuration with connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using them
        pool_size=5,         # Number of connections to maintain
        max_overflow=10      # Additional connections when pool is exhausted
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)  # user_id from main backend JWT
    created_at = Column(DateTime, default=datetime.utcnow)
    title = Column(String, default="New Chat")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ErrorLog(Base):
    """Stores API error logs for monitoring"""
    __tablename__ = "error_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, nullable=False)  # The API endpoint that caused the error
    error_type = Column(String, nullable=False)  # Exception type name
    error_message = Column(Text, nullable=False)  # Error details
    user_id = Column(String, nullable=True)  # User who triggered the error (if authenticated)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


def init_db():
    """Initialize database tables. Safe to call multiple times."""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization warning (may be race condition): {e}")
        # Tables might already exist from another worker, which is fine


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()