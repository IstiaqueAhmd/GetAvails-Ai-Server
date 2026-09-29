from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime

# ==================== Chat Schemas ====================

# What an assistant reply carries. "message" is plain text only; the others
# mean `data` holds structured results for the client to render.
# Keep in sync with TOOL_RESPONSE_TYPES in src/tools.py.
ResponseType = Literal["message", "artists", "venues", "offer"]

# Structured payload: a list of results (artists/venues) or one object (offer)
ResponseData = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]

class ChatMessage(BaseModel):
    role: str
    content: str
    response_type: ResponseType = "message"
    data: ResponseData = None
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    response_type: ResponseType = "message"
    data: ResponseData = None
    session_id: str
    timestamp: datetime

class PublicChatResponse(BaseModel):
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