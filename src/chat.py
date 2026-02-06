from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv
from src.database import SessionLocal, ChatSettings

load_dotenv()

def get_or_create_chat_settings():
    """Get chat settings from database, create default if not exists"""
    db = SessionLocal()
    try:
        settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
        if not settings:
            settings = ChatSettings(
                id=1,
                openai_api_key=None,  # Will fall back to env var
                model="gpt-4o-mini",
                system_prompt="You are a helpful assistant.",
                max_tokens=1000,
                response_format="Short and concise"
            )
            db.add(settings)
            db.commit()
            db.refresh(settings)
        return {
            "openai_api_key": settings.openai_api_key,
            "model": settings.model,
            "system_prompt": settings.system_prompt,
            "max_tokens": settings.max_tokens,
            "response_format": settings.response_format
        }
    finally:
        db.close()

class Chat:
    # Class-level cache for API key (loaded once at startup)
    _cached_api_key = None
    _client = None
    
    def __init__(self):
        self._initialize_client()
        self.load_settings()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key from database or env"""
        if Chat._client is None:
            settings = get_or_create_chat_settings()
            api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            Chat._cached_api_key = api_key
            Chat._client = OpenAI(api_key=api_key)
        self.client = Chat._client
    
    @classmethod
    def reload_api_key(cls):
        """Reload API key from database (call after updating the key)"""
        settings = get_or_create_chat_settings()
        api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        cls._cached_api_key = api_key
        cls._client = OpenAI(api_key=api_key)
    
    def load_settings(self):
        """Load settings from database (except API key which is cached)"""
        settings = get_or_create_chat_settings()
        self.model = settings["model"]
        self.system_prompt = settings["system_prompt"]
        self.max_tokens = settings["max_tokens"]
        self.response_format = settings["response_format"]

    def generate_response(
        self, 
        message: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate a response using OpenAI API"""
        try:
            # Reload settings to get latest from database
            self.load_settings()
            
            # Build messages array for OpenAI chat completion
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"]:
                        messages.append({"role": role, "content": content})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Return user-friendly error messages without exposing API details
            error_str = str(e).lower()
            
            if "insufficient_quota" in error_str or "429" in error_str or "exceeded" in error_str:
                return "Sorry, the service is currently unavailable due to high demand. Please try again later."
            elif "invalid_api_key" in error_str or "401" in error_str or "api_key" in error_str:
                return "Sorry, there's a configuration issue with the service. Please contact support."
            elif "rate_limit" in error_str:
                return "Sorry, the service is experiencing high traffic. Please wait a moment and try again."
            elif "timeout" in error_str or "connection" in error_str:
                return "Sorry, the service is temporarily unavailable. Please try again in a moment."
            else:
                # Generic error message that doesn't expose internals
                return "Sorry, something went wrong. Please try again later."
    
    def get_conversation_context(self, messages: List[Dict[str, str]], max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context for API calls"""
        return messages[-max_messages:] if len(messages) > max_messages else messages
    
    def generate_title(self, message) -> str:
        """Generate a title for the chat session based on initial user message"""
        try:
            prompt = f"Generate a concise and descriptive title (max 5 words) for a chat that starts with: {message}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You generate short, concise titles for chat conversations. Respond with only the title, no quotes or extra text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.5
            )
            
            title = response.choices[0].message.content.strip().strip('"')
            return title if title else "New Chat"
            
        except Exception as e:
            return "New Chat"