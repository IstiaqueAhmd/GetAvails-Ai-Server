import uuid
from datetime import datetime
from typing import List, Dict
from sqlalchemy.orm import Session
from src.database import ChatSession, ChatMessage, User, ErrorLog

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

def save_message(db: Session, session_id: str, role: str, content: str):
    """Save a message to the database"""
    db_message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        timestamp=datetime.utcnow()
    )
    db.add(db_message)
    db.commit()

def get_chat_history(db: Session, session_id: str, page: int = 1, page_size: int = 20) -> tuple[List[Dict[str, str]], int]:
    """Get paginated chat history for a session"""
    # Get total count
    total = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).count()
    
    # Get paginated messages
    offset = (page - 1) * page_size
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).offset(offset).limit(page_size).all()
    
    return [{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp} for msg in messages], total

def get_user_sessions(db: Session, user_id: str) -> List[ChatSession]:
    """Get all chat sessions for a user"""
    return db.query(ChatSession).filter(
        ChatSession.user_id == user_id
    ).order_by(ChatSession.created_at.desc()).all()

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

def get_username(db: Session, user_id: int) -> str:
    """Get the username of a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        return user.username
    return "User"

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