from sqlalchemy import create_engine, inspect, text, Column, Integer, String, DateTime, Text, JSON
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
    # Running summary of older messages, so long chats keep early context
    # without sending every message to the model. summary_upto_id is the id of
    # the last chat_messages row folded into the summary.
    summary = Column(Text, nullable=True)
    summary_upto_id = Column(Integer, nullable=True)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    # "message", "artists", "venues" or "offer" (see ResponseType in schema.py)
    response_type = Column(String, nullable=False, default="message", server_default="message")
    data = Column(JSON, nullable=True)  # Structured tool results shown with this message
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


def _migrate_chat_messages():
    """
    Add columns introduced after the chat tables were first created.

    create_all() only creates missing tables; it never alters existing ones, so
    databases created before these columns existed need them added here.
    Additive and idempotent.
    """
    inspector = inspect(engine)
    statements = []

    if inspector.has_table("chat_messages"):
        existing = {col["name"] for col in inspector.get_columns("chat_messages")}
        if "response_type" not in existing:
            statements.append(
                "ALTER TABLE chat_messages "
                "ADD COLUMN response_type VARCHAR NOT NULL DEFAULT 'message'"
            )
        if "data" not in existing:
            statements.append("ALTER TABLE chat_messages ADD COLUMN data JSON")

    if inspector.has_table("chat_sessions"):
        existing = {col["name"] for col in inspector.get_columns("chat_sessions")}
        if "summary" not in existing:
            statements.append("ALTER TABLE chat_sessions ADD COLUMN summary TEXT")
        if "summary_upto_id" not in existing:
            statements.append("ALTER TABLE chat_sessions ADD COLUMN summary_upto_id INTEGER")

    if statements:
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
        logger.info(f"Migrated chat tables: {statements}")


def init_db():
    """Initialize database tables. Safe to call multiple times."""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        _migrate_chat_messages()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization warning (may be race condition): {e}")
        # Tables/columns might already exist from another worker, which is fine


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()