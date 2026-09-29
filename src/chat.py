from openai import OpenAI
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple
import json
import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from src.graphs import create_agent_graph
from src.tools import TOOL_RESPONSE_TYPES, summarize_tool_data

load_dotenv()

# Configure logging for the tour guide agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tour_guide.chat")

# Default system prompt for the Ava agent
DEFAULT_SYSTEM_PROMPT = """You are Ava, the friendly AI assistant for GetAvails. You help artists, venues, and everyone else on the platform get around GetAvails: navigating the platform, finding artists and venues, understanding offers, and anything else they need to get the most out of it.

IMPORTANT GUIDELINES:
1. For general conversation, greetings, or unclear messages - respond directly WITHOUT using any tools. Just be friendly and ask how you can help.
2. Only use a tool when it is directly relevant to what the user asked.
3. When a tool returns artists, venues, or an offer, the app shows the full details to the user as cards. Keep your reply to a brief summary or next step instead of repeating every field.

If the user's message is vague or unclear, ask clarifying questions instead of using tools. Be conversational and helpful!"""


@dataclass
class ChatResult:
    """An assistant reply plus any structured data produced by tools this turn."""
    content: str
    response_type: str = "message"
    data: Optional[Any] = None


def get_chat_settings():
    """Get chat settings from environment variables."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "system_prompt": os.getenv("CHAT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        "max_tokens": int(os.getenv("CHAT_MAX_TOKENS", "1000")),
        "response_format": os.getenv("CHAT_RESPONSE_FORMAT", "Short and concise"),
        # Cap how many prior messages are sent to the model to bound latency/cost
        "max_history_messages": int(os.getenv("CHAT_MAX_HISTORY_MESSAGES", "20")),
    }


class Chat:
    # Class-level cache for API key (loaded once at startup)
    _cached_api_key = None
    _client = None
    _agent_graph = None
    # Signature of the settings the cached graph was built with; used to
    # avoid recompiling the graph on every request.
    _graph_signature = None

    def __init__(self):
        self._initialize_client()
        self.load_settings()
        self._ensure_agent_graph()

    def _initialize_client(self):
        """Initialize OpenAI client with API key from env"""
        if Chat._client is None:
            settings = get_chat_settings()
            api_key = settings.get("openai_api_key")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable is not set"
                )
            Chat._cached_api_key = api_key
            Chat._client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        self.client = Chat._client

    def _settings_signature(self):
        """A hashable snapshot of the settings that affect the compiled graph."""
        return (self.model, self.system_prompt, self.max_tokens)

    def _ensure_agent_graph(self):
        """Build the LangGraph agent once, rebuilding only when settings change."""
        signature = self._settings_signature()
        if Chat._agent_graph is None or Chat._graph_signature != signature:
            logger.info("Building LangGraph agent (first run or settings changed)")
            Chat._agent_graph = create_agent_graph(
                model=self.model,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                api_key=Chat._cached_api_key
            )
            Chat._graph_signature = signature
        self.agent_graph = Chat._agent_graph

    def load_settings(self):
        """Load settings from environment variables"""
        settings = get_chat_settings()
        self.model = settings["model"]
        self.system_prompt = settings["system_prompt"]
        self.max_tokens = settings["max_tokens"]
        self.response_format = settings["response_format"]
        self.max_history_messages = settings["max_history_messages"]

    def generate_response(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]] = None
    ) -> ChatResult:
        """Generate a response (and any structured tool data) using the LangGraph agent"""
        try:
            # Reload settings to get latest from environment, rebuilding the
            # compiled graph only when a setting that affects it changed.
            self.load_settings()
            self._ensure_agent_graph()
            logger.info(f"Processing chat request: {message[:50]}..." if len(message) > 50 else f"Processing chat request: {message}")

            # Only send the most recent messages to the model to bound latency,
            # token cost, and the risk of exceeding the context window.
            if conversation_history:
                conversation_history = self.get_conversation_context(
                    conversation_history, max_messages=self.max_history_messages
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
                        # Include structured data shown with earlier replies so the
                        # model can refer back to it (e.g. "offer the first artist").
                        # Use the compact summary; full rows are for the client.
                        data = msg.get("data")
                        if data is not None:
                            response_type = msg.get("response_type")
                            summary = summarize_tool_data(response_type, data)
                            content = (
                                f"{content}\n\n(Context: {response_type} data shown "
                                f"to the user with this reply: {json.dumps(summary)})"
                            )
                        messages.append(AIMessage(content=content))

            # Add current message
            messages.append(HumanMessage(content=message))

            logger.info(f"Invoking agent graph with {len(messages)} messages")

            # Invoke the agent graph with recursion limit to prevent infinite loops
            # Max 10 tool calls per request (5 round trips of agent -> tool -> agent)
            config = {"recursion_limit": 10}
            result = self.agent_graph.invoke({"messages": messages}, config=config)

            # Extract the final response and any structured tool data from this turn
            final_message = result["messages"][-1]
            response_type, data = self._extract_structured_result(result["messages"])
            logger.info(
                f"Agent response received ({len(final_message.content)} chars, "
                f"response_type={response_type})"
            )
            return ChatResult(
                content=final_message.content,
                response_type=response_type,
                data=data
            )

        except Exception as e:
            # Return user-friendly error messages without exposing API details
            error_str = str(e).lower()
            logger.error(f"Error in generate_response: {error_str}")
            
            if "recursion" in error_str or "limit" in error_str:
                logger.warning("Agent hit recursion limit - possible tool loop detected")
                return ChatResult("I got a bit confused processing your request. Could you please rephrase or provide more specific details about what you'd like help with?")
            elif "insufficient_quota" in error_str or "429" in error_str or "exceeded" in error_str:
                return ChatResult("Sorry, the service is currently unavailable due to high demand. Please try again later.")
            elif "invalid_api_key" in error_str or "401" in error_str or "api_key" in error_str:
                return ChatResult("Sorry, there's a configuration issue with the service. Please contact support.")
            elif "rate_limit" in error_str:
                return ChatResult("Sorry, the service is experiencing high traffic. Please wait a moment and try again.")
            elif "timeout" in error_str or "connection" in error_str:
                return ChatResult("Sorry, the service is temporarily unavailable. Please try again in a moment.")
            else:
                # Generic error message that doesn't expose internals
                return ChatResult("Sorry, something went wrong. Please try again later.")

    @staticmethod
    def _extract_structured_result(messages: List[BaseMessage]) -> Tuple[str, Optional[Any]]:
        """
        Find the structured tool result produced during the current turn.

        Only messages after the latest user message are considered, so results
        from earlier turns never leak into this reply. If several mapped tools
        ran this turn, the last successful one determines the response type.
        """
        last_human_index = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
            default=-1
        )

        response_type, data = "message", None
        for m in messages[last_human_index + 1:]:
            if not isinstance(m, ToolMessage):
                continue
            if getattr(m, "status", "success") == "error" or m.artifact is None:
                continue
            mapped_type = TOOL_RESPONSE_TYPES.get(m.name)
            if mapped_type:
                response_type, data = mapped_type, m.artifact
        return response_type, data

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