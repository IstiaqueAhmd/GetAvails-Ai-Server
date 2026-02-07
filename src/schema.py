from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# ==================== Chat Schemas ====================
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    session_id: str
    timestamp: datetime

class ChatSession(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    page: int
    limit: int
    total_messages: int
    total_pages: int

class SessionList(BaseModel):
    sessions: List[ChatSession]
    page: int
    limit: int
    total: int
    total_pages: int

class TitleUpdateRequest(BaseModel):
    session_id: str
    title: str