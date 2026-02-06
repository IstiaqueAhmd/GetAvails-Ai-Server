from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from datetime import datetime, timedelta
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

from src.database import get_db, User, ChatMessage as ChatMessageDB, ErrorLog, ChatSettings
from src.schema import (
    ChatRequest, ChatResponse, ChatHistory, SessionList, ChatSession,
    TitleUpdateRequest,
    UserCreate, UserLogin, Token, User as UserSchema,
    UsernameUpdate, ChangePassword,
    MonthlyVolumeResponse, MonthlyVolume, ChatsTodayResponse,
    AverageResponseTimeResponse, ErrorsTodayResponse,
    ChatSettingsSchema, ChatSettingsUpdate, APIKeyResponse, APIKeyUpdate
)
from src.auth import (
    get_password_hash, verify_password, authenticate_user, create_access_token,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_admin_user
)
from src.chat import Chat
from src.utils import (
    create_chat_session, 
    save_message, 
    get_chat_history, 
    get_user_sessions,
    delete_chat_session,
    update_session_title,
    get_username,
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

# Initialize chat service
chat = Chat()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Chat API is running!", "version": "1.0.0"}

# ==================== Authentication Endpoints ====================

# -------------------- Email/Password Endpoints --------------------

@app.post("/register", response_model=UserSchema)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    - **email**: Valid email address (must be unique)
    - **username**: Display name for the user
    - **password**: Password (will be hashed before storage)
    """
    try:
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@app.post("/login", response_model=Token)
async def login(form_data: UserLogin, db: Session = Depends(get_db)):
    """
    Login with email and password to receive a JWT token.
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns an access token that should be included in the Authorization header
    for subsequent requests: `Authorization: Bearer <token>`
    """
    user = authenticate_user(db, form_data.email, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserSchema)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user's information.
    
    Requires valid JWT token in Authorization header.
    """
    return current_user

@app.put("/users/update-username", response_model=UserSchema)
async def update_username(
    request: UsernameUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update the current user's username.
    
    Requires valid JWT token in Authorization header.
    """
    try:
        # Update the username
        current_user.username = request.username
        db.commit()
        db.refresh(current_user)
        
        return current_user
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating username: {str(e)}"
        )

@app.put("/users/change-password")
async def change_password(
    request: ChangePassword,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change the current user's password.
    
    Requires valid JWT token in Authorization header.
    
    - **old_password**: Current password for verification
    - **new_password**: New password to set
    """
    try:
        # Verify old password
        if not current_user.hashed_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change not available for OAuth users"
            )
        
        if not verify_password(request.old_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect old password"
            )
        
        # Update password
        current_user.hashed_password = get_password_hash(request.new_password)
        db.commit()
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error changing password: {str(e)}"
        )

# ==================== Health & Status Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# ==================== Health & Status Endpoints ====================

@app.get("/user-name")
async def get_user_name(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Get user name endpoint"""
    return {"username": get_username(db, current_user.id)}

@app.get("/user-email")
async def get_user_email(current_user: User = Depends(get_current_active_user)):
    """Get user email endpoint"""
    return {"email": current_user.email}

# ==================== Chat Endpoints ====================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response (requires authentication)"""
    try:
        session_id = request.session_id
        user_id = str(current_user.id)

        # Get conversation history
        history = get_chat_history(db, session_id)
    
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
            response=ai_response,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        log_error(db, "/chat", e, user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post("/public/chat", response_model=ChatResponse)
async def public_chat_endpoint(request: ChatRequest):
    """
    Send a message and get AI response without authentication.
    
    This endpoint does not require authentication and does not save chat history.
    """
    try:
        # Generate AI response
        ai_response = chat.generate_response(request.message, None)
        
        return ChatResponse(
            response=ai_response,
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new chat session (requires authentication)"""
    try:
        user_id = str(current_user.id)
        session_id = create_chat_session(db, user_id, title)
        return {"session_id": session_id, "message": "Session created successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )

@app.get("/sessions", response_model=SessionList)
async def get_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all chat sessions for the authenticated user"""
    try:
        user_id = str(current_user.id)
        sessions = get_user_sessions(db, user_id)
        return SessionList(sessions=sessions)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sessions: {str(e)}"
        )

@app.get("/chat/history", response_model=ChatHistory)
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get chat history for a specific session (requires authentication and ownership)"""
    try:
        messages = get_chat_history(db, session_id)
        chat_messages = [
            {"role": msg["role"], "content": msg["content"], "timestamp": datetime.utcnow()}
            for msg in messages
        ]
        return ChatHistory(session_id=session_id, messages=chat_messages)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@app.delete("/delete-session")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session (requires authentication and ownership)"""
    try:
        user_id = str(current_user.id)
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update session title (requires authentication and ownership)"""
    try:
        # Verify the user owns this session
        user_id = str(current_user.id)
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

# ==================== Admin Dashboard Endpoints ====================

@app.get("/admin/chat-volume", response_model=MonthlyVolumeResponse)
async def get_monthly_chat_volume(
    year: Optional[int] = None,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get month-wise chat volume for a given year (admin only).
    
    - **year**: Optional. The year to get chat volume for. Defaults to current year.
    
    Returns the number of chat messages per month for the specified year.
    """
    try:
        # Default to current year if not provided
        if year is None:
            year = datetime.utcnow().year
        
        # Query to get chat count grouped by month
        monthly_stats = db.query(
            extract('month', ChatMessageDB.timestamp).label('month'),
            func.count(ChatMessageDB.id).label('count')
        ).filter(
            extract('year', ChatMessageDB.timestamp) == year
        ).group_by(
            extract('month', ChatMessageDB.timestamp)
        ).all()
        
        # Create a dictionary for quick lookup
        month_counts = {int(stat.month): stat.count for stat in monthly_stats}
        
        # Month names for response
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        
        # Build complete monthly data (including months with 0 chats)
        monthly_data = []
        total_chats = 0
        
        for month_num in range(1, 13):
            count = month_counts.get(month_num, 0)
            total_chats += count
            monthly_data.append(MonthlyVolume(
                month=month_num,
                month_name=month_names[month_num - 1],
                chat_count=count
            ))
        
        return MonthlyVolumeResponse(
            year=year,
            monthly_data=monthly_data,
            total_chats=total_chats
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat volume: {str(e)}"
        )

@app.get("/admin/chats-today", response_model=ChatsTodayResponse)
async def get_chats_today(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get the number of chat messages for today (admin only).
    
    Returns the count of all chat messages created today.
    """
    try:
        # Get today's date range (start of day to now)
        today = datetime.utcnow().date()
        
        # Count chat messages created today
        chat_count = db.query(func.count(ChatMessageDB.id)).filter(
            func.date(ChatMessageDB.timestamp) == today
        ).scalar()
        
        return ChatsTodayResponse(chat_today=chat_count or 0)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving today's chat count: {str(e)}"
        )

@app.get("/admin/average-response-time", response_model=AverageResponseTimeResponse)
async def get_average_response_time(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get average response time in seconds (admin only).
    
    Calculates the average time between user messages and assistant responses.
    """
    try:
        # Get all messages ordered by session and timestamp
        messages = db.query(ChatMessageDB).order_by(
            ChatMessageDB.session_id,
            ChatMessageDB.timestamp
        ).all()
        
        response_times = []
        
        # Group by session and calculate response times
        i = 0
        while i < len(messages) - 1:
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            # Check if this is a user message followed by assistant response in same session
            if (current_msg.session_id == next_msg.session_id and
                current_msg.role == "user" and 
                next_msg.role == "assistant"):
                
                time_diff = (next_msg.timestamp - current_msg.timestamp).total_seconds()
                if time_diff >= 0:  # Only count valid positive times
                    response_times.append(time_diff)
            
            i += 1
        
        # Calculate average
        if response_times:
            avg_time = sum(response_times) / len(response_times)
        else:
            avg_time = 0.0
        
        return AverageResponseTimeResponse(average_response_time=round(avg_time, 2))
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating average response time: {str(e)}"
        )

@app.get("/admin/errors-today", response_model=ErrorsTodayResponse)
async def get_errors_today(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get the number of errors logged today (admin only).
    
    Returns the count of all errors that occurred today.
    """
    try:
        # Get today's date
        today = datetime.utcnow().date()
        
        # Count errors logged today
        error_count = db.query(func.count(ErrorLog.id)).filter(
            func.date(ErrorLog.timestamp) == today
        ).scalar() or 0
        
        return ErrorsTodayResponse(errors_today=error_count)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving error count: {str(e)}"
        )

@app.get("/admin/chat-settings", response_model=ChatSettingsSchema)
async def get_chat_settings(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get current chat settings (admin only).
    
    Returns the current model, system prompt, max tokens, and response format.
    """
    try:
        settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
        
        if not settings:
            # Create default settings if not exist
            settings = ChatSettings(
                id=1,
                model="gpt-4o-mini",
                system_prompt="You are a helpful assistant.",
                max_tokens=1000,
                response_format="Short and concise"
            )
            db.add(settings)
            db.commit()
            db.refresh(settings)
        
        return ChatSettingsSchema(
            model=settings.model,
            system_prompt=settings.system_prompt,
            max_tokens=settings.max_tokens,
            response_format=settings.response_format
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat settings: {str(e)}"
        )

@app.put("/admin/chat-settings", response_model=ChatSettingsSchema)
async def update_chat_settings(
    request: ChatSettingsUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update chat settings (admin only).
    
    Only provided fields will be updated.
    
    - **model**: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o")
    - **system_prompt**: System prompt for the AI
    - **max_tokens**: Maximum tokens for response
    - **response_format**: Response format instruction
    """
    try:
        settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
        
        if not settings:
            settings = ChatSettings(id=1)
            db.add(settings)
        
        # Update only provided fields
        if request.model is not None:
            settings.model = request.model
        if request.system_prompt is not None:
            settings.system_prompt = request.system_prompt
        if request.max_tokens is not None:
            settings.max_tokens = request.max_tokens
        if request.response_format is not None:
            settings.response_format = request.response_format
        
        db.commit()
        db.refresh(settings)
        
        return ChatSettingsSchema(
            model=settings.model,
            system_prompt=settings.system_prompt,
            max_tokens=settings.max_tokens,
            response_format=settings.response_format
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating chat settings: {str(e)}"
        )

def mask_api_key(key: str) -> str:
    """Mask API key for display, showing only first 3 and last 4 characters"""
    if not key or len(key) < 10:
        return "***"
    return f"{key[:3]}...{key[-4:]}"

@app.get("/admin/api-key", response_model=APIKeyResponse)
async def get_api_key(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get current OpenAI API key (masked for security) (admin only).
    
    Returns the API key with most characters hidden.
    """
    try:
        settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
        
        if not settings or not settings.openai_api_key:
            # Check if env var has a key
            env_key = os.getenv("OPENAI_API_KEY")
            return APIKeyResponse(
                openai_api_key=mask_api_key(env_key) if env_key else "Not set",
                has_key=bool(env_key)
            )
        
        return APIKeyResponse(
            openai_api_key=mask_api_key(settings.openai_api_key),
            has_key=True
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving API key: {str(e)}"
        )

@app.put("/admin/api-key", response_model=APIKeyResponse)
async def update_api_key(
    request: APIKeyUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update OpenAI API key (admin only).
    
    After updating, the new key will be used for all subsequent chat requests.
    
    - **openai_api_key**: The new OpenAI API key
    """
    try:
        settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
        
        if not settings:
            settings = ChatSettings(id=1)
            db.add(settings)
        
        settings.openai_api_key = request.openai_api_key
        db.commit()
        
        # Reload the API key in the Chat class
        chat.reload_api_key()
        
        return APIKeyResponse(
            openai_api_key=mask_api_key(request.openai_api_key),
            has_key=True
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating API key: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)