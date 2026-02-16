from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

from src.database import get_db, ChatMessage as ChatMessageDB
from src.schema import (
    ChatRequest, ChatResponse, ChatHistory, SessionList, ChatSession,
    TitleUpdateRequest
)
from src.auth import get_current_user
from src.chat import Chat
from src.utils import (
    create_chat_session, 
    save_message, 
    get_chat_history, 
    get_user_sessions,
    delete_chat_session,
    update_session_title,
    log_error
)

# Check if running in production
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

# Create FastAPI app with docs disabled in production
app = FastAPI(
    title="Houseme Chat API",
    description="A chat API powered by AI",
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware for OAuth state management
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("JWT_SECRET_KEY", "change-this-secret-key"),
    same_site="lax",  # Required for OAuth redirects
    https_only=False  # Set to True in production with HTTPS
)

# Startup event to ensure database is ready
@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    from src.database import init_db
    init_db()

# Initialize chat service (lazy - will be created on first use)
chat = Chat()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Chat API is running!", "version": "1.0.0"}

# ==================== Health & Status Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# ==================== Chat Endpoints ====================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response (requires authentication)"""
    try:
        session_id = request.session_id

        # Get conversation history (all messages for AI context)
        history, _ = get_chat_history(db, session_id, page=1, limit=1000)
    
        # Generate title if this is the first message
        if len(history) == 0:
            title = chat.generate_title(request.message)
            update_session_title(db, session_id, user_id, title)

        # Save user message
        save_message(db, session_id, "user", request.message)
        
        # Generate AI response
        ai_response = chat.generate_response(request.message, history)
        
        # Save AI response
        save_message(db, session_id, "assistant", ai_response)
        
        return ChatResponse(
            role="assistant",
            content=ai_response,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        log_error(db, "/chat", e, user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

# Default system prompt for the public landing page agent
PUBLIC_CHAT_SYSTEM_PROMPT = os.getenv(
    "PUBLIC_CHAT_SYSTEM_PROMPT",
    "You are GetAvails AI, a friendly and helpful assistant on the GetAvails landing page. "
)

# Initialize a shared OpenAI client for the public chat endpoint
from openai import OpenAI as PublicOpenAI
_public_chat_client = PublicOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/public/chat", response_model=ChatResponse)
async def public_chat_endpoint(request: ChatRequest):
    """
    Send a message and get AI response without authentication.
    
    This is a generic landing page assistant. It does NOT use RAG or any tools.
    It does not require authentication and does not save chat history.
    """
    try:
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1000"))

        response = _public_chat_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PUBLIC_CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        return ChatResponse(
            role="assistant",
            content=ai_response,
            session_id=request.session_id or "public",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post("/create-session", response_model=dict)
async def create_session(
    title: str = "New Chat",
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new chat session (requires authentication)"""
    try:
        session_id = create_chat_session(db, user_id, title)
        return {"session_id": session_id, "message": "Session created successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )

@app.get("/sessions", response_model=SessionList)
async def get_sessions(
    search: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated chat sessions for the authenticated user.
    
    - **search**: Optional search query to filter sessions by title
    - **page**: Page number (default: 1)
    - **limit**: Number of sessions per page (default: 20, max: 100)
    """
    try:
        # Validate pagination parameters
        limit = min(limit, 100)  # Cap at 100
        page = max(page, 1)  # Ensure page >= 1
        
        sessions, total = get_user_sessions(db, user_id, search, page, limit)
        total_pages = (total + limit - 1) // limit  # Ceiling division
        
        return SessionList(
            sessions=sessions,
            page=page,
            limit=limit,
            total=total,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sessions: {str(e)}"
        )

@app.get("/chat/history", response_model=ChatHistory)
async def get_session_history(
    session_id: str,
    page: int = 1,
    limit: int = 20,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated chat history for a specific session.
    
    - **session_id**: The session ID to get history for
    - **page**: Page number (default: 1)
    - **limit**: Number of messages per page (default: 20, max: 100)
    """
    try:
        # Validate limit
        limit = min(limit, 100)  # Cap at 100
        page = max(page, 1)  # Ensure page >= 1
        
        messages, total = get_chat_history(db, session_id, page, limit)
        total_pages = (total + limit - 1) // limit  # Ceiling division
        
        chat_messages = [
            {"role": msg["role"], "content": msg["content"], "timestamp": msg.get("timestamp", datetime.utcnow())}
            for msg in messages
        ]
        return ChatHistory(
            session_id=session_id,
            messages=chat_messages,
            page=page,
            limit=limit,
            total_messages=total,
            total_pages=total_pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@app.delete("/delete-session")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session (requires authentication and ownership)"""
    try:
        success = delete_chat_session(db, session_id, user_id)
        if success:
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session: {str(e)}"
        )

@app.put("/sessions/update-title")
async def update_title(
    request: TitleUpdateRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update session title (requires authentication and ownership)"""
    try:
        success = update_session_title(db, request.session_id, user_id, request.title)
        if success:
            return {"message": "Title updated successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating title: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)