import uuid
from datetime import datetime
from typing import Any, List, Dict, Optional
from sqlalchemy.orm import Session
from src.database import ChatSession, ChatMessage, ErrorLog

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def create_chat_session(db: Session, user_id: str, title: str = "New Chat") -> str:
    """Create a new chat session"""
    session_id = generate_session_id()
    db_session = ChatSession(
        session_id=session_id,
        user_id=user_id,
        title=title,
        created_at=datetime.utcnow()
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return session_id

def session_belongs_to_user(db: Session, session_id: str, user_id: str) -> bool:
    """Return True if the session exists and is owned by the given user."""
    return db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == user_id
    ).first() is not None

def save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    response_type: str = "message",
    data: Optional[Any] = None
):
    """Save a message (and any structured data shown with it) to the database"""
    db_message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        response_type=response_type,
        data=data,
        timestamp=datetime.utcnow()
    )
    db.add(db_message)
    db.commit()

def get_chat_history(db: Session, session_id: str, page: int = 1, limit: int = 20) -> tuple[List[Dict[str, str]], int]:
    """Get paginated chat history for a session"""
    # Get total count
    total = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).count()
    
    # Get paginated messages
    offset = (page - 1) * limit
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).offset(offset).limit(limit).all()
    
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "response_type": msg.response_type or "message",
            "data": msg.data,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ], total

def get_agent_context(db: Session, session_id: str, max_messages: int) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Conversation context for the agent: the session's running summary plus the
    most recent messages not yet folded into it (newest last, at most
    max_messages of them).
    """
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    summary = session.summary if session else None
    upto_id = session.summary_upto_id if session else None

    query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
    if upto_id is not None:
        query = query.filter(ChatMessage.id > upto_id)
    # Take the newest rows, then restore chronological order
    rows = query.order_by(ChatMessage.id.desc()).limit(max_messages).all()
    rows.reverse()

    return summary, [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "response_type": msg.response_type or "message",
            "data": msg.data,
        }
        for msg in rows
    ]

def update_session_summary(db: Session, session_id: str, summary: str, upto_id: int) -> None:
    """Store a new running summary covering messages up to and including upto_id."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if session:
        session.summary = summary
        session.summary_upto_id = upto_id
        db.commit()

def get_user_sessions(db: Session, user_id: str, search: str = None, page: int = 1, limit: int = 20) -> tuple:
    """Get paginated chat sessions for a user with optional search by title"""
    query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
    
    # Apply search filter if provided
    if search:
        query = query.filter(ChatSession.title.ilike(f"%{search}%"))
    
    # Get total count before pagination
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * limit
    sessions = query.order_by(ChatSession.created_at.desc()).offset(offset).limit(limit).all()
    
    return sessions, total

def delete_chat_session(db: Session, session_id: str, user_id: str) -> bool:
    """Delete a chat session and all its messages"""
    # First delete all messages
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    
    # Then delete the session
    session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == user_id
    ).first()
    
    if session:
        db.delete(session)
        db.commit()
        return True
    return False

def update_session_title(db: Session, session_id: str, user_id: str, title: str) -> bool:
    """Update the title of a chat session"""
    session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == user_id
    ).first()
    
    if session:
        session.title = title
        db.commit()
        return True
    return False

def log_error(db: Session, endpoint: str, error: Exception, user_id: str = None):
    """Log an error to the database"""
    try:
        error_log = ErrorLog(
            endpoint=endpoint,
            error_type=type(error).__name__,
            error_message=str(error),
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        db.add(error_log)
        db.commit()
    except Exception:
        # Don't let error logging fail the main request
        db.rollback()