from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from src.database import get_db, SessionLocal, ChatMessage as ChatMessageDB
from src.schema import (
    ChatRequest, ChatResponse, PublicChatResponse, ChatHistory, SessionList,
    ChatSession, TitleUpdateRequest
)
from src.auth import get_current_user
from src.chat import Chat
from src.utils import (
    create_chat_session,
    save_message,
    get_chat_history,
    get_agent_context,
    update_session_summary,
    get_user_sessions,
    delete_chat_session,
    update_session_title,
    session_belongs_to_user,
    log_error
)

logger = logging.getLogger("tour_guide.api")

# Check if running in production
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

# Secret used for both JWT validation and the session middleware. Required.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set")

# Allowed CORS origins. Comma-separated list in ALLOWED_ORIGINS, e.g.
# "https://app.getavails.com,https://getavails.com". Defaults to localhost for
# local development. "*" cannot be combined with credentials, so it is not used.
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    from src.database import init_db
    init_db()
    yield


# Create FastAPI app with docs disabled in production
app = FastAPI(
    title="Getavails AI API",
    description="An API server powered by AI",
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware (explicit origins so credentialed requests work)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware for OAuth state management
app.add_middleware(
    SessionMiddleware,
    secret_key=JWT_SECRET_KEY,
    same_site="lax",  # Required for OAuth redirects
    https_only=IS_PRODUCTION  # Secure cookies in production (HTTPS)
)

# Initialize the chat service (fails fast if OPENAI_API_KEY is missing)
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

# Messages sent to the model in full; once a chat has more unsummarized
# messages than this, older ones are folded into the session summary.
CHAT_MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_MAX_HISTORY_MESSAGES", "20"))
# Most recent messages kept in full when older ones are summarized
CHAT_SUMMARY_KEEP_RECENT = int(os.getenv("CHAT_SUMMARY_KEEP_RECENT", "8"))


def _require_owned_session(db: Session, session_id: Optional[str], user_id: str) -> str:
    """Validate the session id and ownership before doing any work."""
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required"
        )
    if not session_belongs_to_user(db, session_id, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied"
        )
    return session_id


async def _start_turn(db: Session, session_id: str, user_id: str, content: str):
    """
    Load the agent's context for a new user message, title the session on its
    first message, and save the message. Returns (summary, history).
    """
    summary, history = get_agent_context(db, session_id, CHAT_MAX_HISTORY_MESSAGES)
    if summary is None and not history:
        title = await run_in_threadpool(chat.generate_title, content)
        update_session_title(db, session_id, user_id, title)
    save_message(db, session_id, "user", content)
    return summary, history


def _refresh_summary(session_id: str) -> None:
    """
    Fold older messages into the session summary once the unsummarized part
    of the chat grows past CHAT_MAX_HISTORY_MESSAGES. Runs after the reply is
    sent, so it never delays a response.
    """
    db = SessionLocal()
    try:
        summary, messages = get_agent_context(db, session_id, max_messages=200)
        if len(messages) <= CHAT_MAX_HISTORY_MESSAGES:
            return
        to_fold = messages[:-CHAT_SUMMARY_KEEP_RECENT]
        new_summary = chat.summarize_conversation(summary, to_fold)
        if new_summary:
            update_session_summary(db, session_id, new_summary, to_fold[-1]["id"])
            logger.info(f"Summarized {len(to_fold)} messages for session {session_id}")
    except Exception as e:
        logger.warning(f"Summary refresh failed for session {session_id}: {e}")
    finally:
        db.close()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response (requires authentication)"""
    session_id = _require_owned_session(db, request.session_id, user_id)

    try:
        summary, history = await _start_turn(db, session_id, user_id, request.content)

        # Generate AI response (blocking network I/O -> run off the event loop)
        result = await run_in_threadpool(
            chat.generate_response, request.content, history, summary
        )

        # Save AI response along with any structured tool data
        save_message(
            db, session_id, "assistant", result.content,
            response_type=result.response_type, data=result.data
        )
        background_tasks.add_task(_refresh_summary, session_id)

        return ChatResponse(
            role="assistant",
            content=result.content,
            response_type=result.response_type,
            data=result.data,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        log_error(db, "/chat", e, user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat request"
        )


def _sse(event: str, payload: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Same as /chat, but streams the reply as Server-Sent Events (requires authentication).

    Events, in order:
        status  {"tool", "message"}  a tool started, e.g. "Searching venues on GetAvails…"
        token   {"text"}             a piece of the reply as it's written
        done    ChatResponse fields  the final reply; always the last event

    Tokens are a live preview: replace the streamed text with `done.content`,
    which is the saved, authoritative reply (with any cards in `data`).
    """
    session_id = _require_owned_session(db, request.session_id, user_id)
    try:
        summary, history = await _start_turn(db, session_id, user_id, request.content)
    except Exception as e:
        log_error(db, "/chat/stream", e, user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat request"
        )

    def events():
        # Runs in a worker thread after the endpoint returns, so it uses its
        # own DB session rather than the request-scoped one.
        for event in chat.stream_response(request.content, history, summary):
            if event["type"] != "done":
                yield _sse(event["type"], {k: v for k, v in event.items() if k != "type"})
                continue

            result = event["result"]
            stream_db = SessionLocal()
            try:
                save_message(
                    stream_db, session_id, "assistant", result.content,
                    response_type=result.response_type, data=result.data
                )
            except Exception as e:
                log_error(stream_db, "/chat/stream", e, user_id=user_id)
            finally:
                stream_db.close()

            yield _sse("done", ChatResponse(
                role="assistant",
                content=result.content,
                response_type=result.response_type,
                data=result.data,
                session_id=session_id,
                timestamp=datetime.utcnow()
            ).model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Keep proxies (e.g. nginx) from buffering the stream
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(_refresh_summary, session_id),
    )

# Default system prompt for the public landing page agent (used when the env var is unset or empty)
PUBLIC_CHAT_SYSTEM_PROMPT = os.getenv("PUBLIC_CHAT_SYSTEM_PROMPT") or (
    "You are Ava, the friendly AI assistant for GetAvails, speaking with visitors on the GetAvails landing page. "
    "GetAvails helps artists, venues, and other roles on the platform connect: finding artists and venues, "
    "sending and understanding offers, and more. Help visitors understand what GetAvails does and how it can help them. "
)

# In-memory (non-persisted) conversation history for the public endpoint.
from src.public_history import InMemoryHistoryStore
from src.utils import generate_session_id

_public_history = InMemoryHistoryStore(
    max_sessions=int(os.getenv("PUBLIC_CHAT_MAX_SESSIONS", "1000")),
    max_messages=int(os.getenv("PUBLIC_CHAT_MAX_HISTORY_MESSAGES", "20")),
    ttl_seconds=int(os.getenv("PUBLIC_CHAT_SESSION_TTL", "3600")),
)

@app.post("/public/chat", response_model=PublicChatResponse)
async def public_chat_endpoint(request: ChatRequest):
    """
    Send a message and get AI response without authentication.
    
    This is a generic landing page assistant. It does NOT use RAG or any tools.
    It does not persist chat history to the database, but it does keep a short,
    in-memory history per session so the conversation stays coherent.
    """
    # Use the provided session id, or mint a new one so history can be tracked.
    # (Never fall back to a shared bucket - that would mix visitors' chats.)
    session_id = request.session_id or generate_session_id()

    try:
        settings = chat.settings

        # Prior turns + new message; the system prompt goes in `instructions`
        messages = list(_public_history.get(session_id))
        messages.append({"role": "user", "content": request.content})

        response = await run_in_threadpool(
            lambda: chat.client.responses.create(
                model=settings.model,
                instructions=PUBLIC_CHAT_SYSTEM_PROMPT,
                input=messages,
                max_output_tokens=settings.max_tokens,
                store=False,
                **({"reasoning": {"effort": settings.reasoning_effort}} if settings.reasoning_effort else {})
            )
        )

        ai_response = response.output_text.strip()
        if not ai_response:
            raise RuntimeError(f"Empty public chat response (status={response.status})")

        # Persist this turn to the in-memory history only after a successful call.
        _public_history.append(session_id, "user", request.content)
        _public_history.append(session_id, "assistant", ai_response)

        return PublicChatResponse(
            role="assistant",
            content=ai_response,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat request"
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
            detail="Error creating session"
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
            detail="Error retrieving sessions"
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
    if not session_belongs_to_user(db, session_id, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied"
        )

    try:
        # Validate limit
        limit = min(limit, 100)  # Cap at 100
        page = max(page, 1)  # Ensure page >= 1

        messages, total = get_chat_history(db, session_id, page, limit)
        total_pages = (total + limit - 1) // limit  # Ceiling division

        chat_messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "response_type": msg["response_type"],
                "data": msg["data"],
                "timestamp": msg.get("timestamp", datetime.utcnow()),
            }
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
            detail="Error retrieving chat history"
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
            detail="Error deleting session"
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
            detail="Error updating title"
        )