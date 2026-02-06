from openai import OpenAI
from typing import List, Dict
import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy.exc import IntegrityError

from src.database import SessionLocal, ChatSettings
from src.graphs import create_agent_graph

load_dotenv()

# Configure logging for the tour guide agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tour_guide.chat")

def get_or_create_chat_settings():
    """Get chat settings from database, create default if not exists.
    
    This function is safe for concurrent access from multiple workers.
    If multiple workers try to create the settings simultaneously,
    we catch the IntegrityError and fetch the existing record.
    """
    db = SessionLocal()
    try:
        settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
        if not settings:
            try:
                settings = ChatSettings(
                    id=1,
                    openai_api_key=None,  # Will fall back to env var
                    model="gpt-4o-mini",
                    system_prompt="""You are a helpful and friendly AI tour guide assistant called GetAvails. You help users plan trips, find flights, discover destinations, and provide travel advice.

IMPORTANT GUIDELINES:
1. For general conversation, greetings, or unclear messages - respond directly WITHOUT using any tools. Just be friendly and ask how you can help.
2. Only use the search_flights tool when the user explicitly asks to search for flights AND provides: origin, destination, and travel dates.
3. Only use get_airport_info when the user specifically asks about airport codes or airports for a city.
4. Only use get_destination_info when the user asks about travel information for a specific destination.

If the user's message is vague or unclear, ask clarifying questions instead of using tools. Be conversational and helpful!""",
                    max_tokens=1000,
                    response_format="Short and concise"
                )
                db.add(settings)
                db.commit()
                db.refresh(settings)
            except IntegrityError:
                # Another worker created the settings first - rollback and fetch it
                db.rollback()
                settings = db.query(ChatSettings).filter(ChatSettings.id == 1).first()
                if not settings:
                    raise RuntimeError("Failed to get or create chat settings")
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
    _agent_graph = None
    
    def __init__(self):
        self._initialize_client()
        self.load_settings()
        self._initialize_agent_graph()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key from database or env"""
        if Chat._client is None:
            settings = get_or_create_chat_settings()
            api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            Chat._cached_api_key = api_key
            Chat._client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        self.client = Chat._client
    
    def _initialize_agent_graph(self):
        """Initialize the LangGraph agent"""
        if Chat._agent_graph is None:
            logger.info("Initializing LangGraph agent")
            Chat._agent_graph = create_agent_graph(
                model=self.model,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                api_key=Chat._cached_api_key
            )
        self.agent_graph = Chat._agent_graph
    
    @classmethod
    def reload_api_key(cls):
        """Reload API key from database (call after updating the key)"""
        logger.info("Reloading API key and recreating agent graph")
        settings = get_or_create_chat_settings()
        api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        cls._cached_api_key = api_key
        cls._client = OpenAI(api_key=api_key)
        # Also recreate the agent graph with new API key
        cls._agent_graph = create_agent_graph(
            model=settings.get("model", "gpt-4o-mini"),
            system_prompt=settings.get("system_prompt", "You are a helpful assistant."),
            max_tokens=settings.get("max_tokens", 1000),
            api_key=api_key
        )
    
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
        """Generate a response using LangGraph agent"""
        try:
            # Reload settings to get latest from database
            self.load_settings()
            logger.info(f"Processing chat request: {message[:50]}..." if len(message) > 50 else f"Processing chat request: {message}")
            
            # Recreate agent graph if settings changed
            self.agent_graph = create_agent_graph(
                model=self.model,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                api_key=Chat._cached_api_key
            )
            
            # Convert conversation history to LangChain messages
            messages = []
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            logger.info(f"Invoking agent graph with {len(messages)} messages")
            
            # Invoke the agent graph with recursion limit to prevent infinite loops
            # Max 10 tool calls per request (5 round trips of agent -> tool -> agent)
            config = {"recursion_limit": 10}
            result = self.agent_graph.invoke({"messages": messages}, config=config)
            
            # Extract the final response
            final_message = result["messages"][-1]
            logger.info(f"Agent response received ({len(final_message.content)} chars)")
            return final_message.content
            
        except Exception as e:
            # Return user-friendly error messages without exposing API details
            error_str = str(e).lower()
            logger.error(f"Error in generate_response: {error_str}")
            
            if "recursion" in error_str or "limit" in error_str:
                logger.warning("Agent hit recursion limit - possible tool loop detected")
                return "I got a bit confused processing your request. Could you please rephrase or provide more specific details about what you'd like help with?"
            elif "insufficient_quota" in error_str or "429" in error_str or "exceeded" in error_str:
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