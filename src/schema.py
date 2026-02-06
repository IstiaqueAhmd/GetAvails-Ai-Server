from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# ==================== Authentication Schemas ====================

class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str

class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for token payload data"""
    email: Optional[str] = None

class User(BaseModel):
    """Schema for user information (without password)"""
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class UsernameUpdate(BaseModel):
    """Schema for updating username"""
    username: str

class ChangePassword(BaseModel):
    """Schema for changing password"""
    old_password: str
    new_password: str

# ==================== Chat Schemas ====================
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
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

class SessionList(BaseModel):
    sessions: List[ChatSession]

class TitleUpdateRequest(BaseModel):
    session_id: str
    title: str

# ==================== Admin Dashboard Schemas ====================

class MonthlyVolume(BaseModel):
    """Volume data for a single month"""
    month: int
    month_name: str
    chat_count: int

class MonthlyVolumeResponse(BaseModel):
    """Response for monthly chat volume endpoint"""
    year: int
    monthly_data: List[MonthlyVolume]
    total_chats: int

class ChatsTodayResponse(BaseModel):
    """Response for chats today endpoint"""
    chat_today: int

class AverageResponseTimeResponse(BaseModel):
    """Response for average response time endpoint"""
    average_response_time: float


class ErrorsTodayResponse(BaseModel):
    """Response for errors today endpoint"""
    errors_today: int

class ChatSettingsSchema(BaseModel):
    """Schema for chat settings"""
    model: str
    system_prompt: str
    max_tokens: int
    response_format: str
    
    class Config:
        from_attributes = True

class ChatSettingsUpdate(BaseModel):
    """Schema for updating chat settings"""
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None

class APIKeyResponse(BaseModel):
    """Schema for API key response (masked for security)"""
    openai_api_key: str  # Will be masked like "sk-...xxxx"
    has_key: bool

class APIKeyUpdate(BaseModel):
    """Schema for updating API key"""
    openai_api_key: str