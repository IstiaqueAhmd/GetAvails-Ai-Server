from openai import OpenAI
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple
import json
import logging
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from src.agent import AgentContext, AgentSettings, build_agent, load_agent_settings
from src.tools import TOOL_RESPONSE_TYPES, TOOL_STATUS_MESSAGES, summarize_tool_data

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
4. GetAvails data comes first. For questions about a specific artist or venue, search the platform (search_artists / search_venues) first. Use web_search only when the platform results are missing or don't cover what the user asked (e.g. parking, box office hours, age policy, recent news). Never use web_search to find artists or venues to book.
5. When you answer from web_search results, say the information comes from the web rather than GetAvails, name or link the source, and suggest the user confirm important details directly with the venue or artist.
6. When the user wants to create or send an offer, gather the offer details and then call generate_offer. Reuse what the conversation and search results already give you, and ask the user for the rest a few related details at a time (the event and venue, then the money, then the production contact) instead of all at once or one by one. Never guess or invent a detail. The app opens the draft for the user to review, add the recipient, sign, and send, and fills in the buyer's and signatory's details from their profile, so don't ask for those, for who to send it to, or for a signature, and never say an offer has been sent.

If the user's message is vague or unclear, ask clarifying questions instead of using tools. Be conversational and helpful!"""

SUMMARY_INSTRUCTIONS = """You maintain a running summary of a conversation between a user and Ava, the GetAvails assistant.
Update the existing summary with the new messages. Keep what later replies may depend on: the user's role and goals, artists/venues discussed (with their ids and sources), dates, locations, fees, offers drafted, preferences, and open questions. While the user is giving details for an offer that has not been drafted yet, keep every detail given so far exactly as stated (names, addresses, phone numbers, amounts, times).
Drop small talk. Write plain, compact notes (no more than ~250 words). Respond with only the updated summary."""

# Reply used when the agent stops without producing a usable answer
FALLBACK_REPLY = (
    "I got a bit confused processing your request. Could you please rephrase or "
    "provide more specific details about what you'd like help with?"
)


@dataclass
class ChatResult:
    """An assistant reply plus any structured data produced by tools this turn."""
    content: str
    response_type: str = "message"
    data: Optional[Any] = None


def _friendly_error(e: Exception) -> str:
    """User-facing message for an agent failure, without exposing API details."""
    error_str = str(e).lower()
    if "insufficient_quota" in error_str or "429" in error_str or "exceeded" in error_str:
        return "Sorry, the service is currently unavailable due to high demand. Please try again later."
    if "invalid_api_key" in error_str or "401" in error_str or "api_key" in error_str:
        return "Sorry, there's a configuration issue with the service. Please contact support."
    if "rate_limit" in error_str:
        return "Sorry, the service is experiencing high traffic. Please wait a moment and try again."
    if "timeout" in error_str or "connection" in error_str:
        return "Sorry, the service is temporarily unavailable. Please try again in a moment."
    return "Sorry, something went wrong. Please try again later."


class Chat:
    # Agent compiled once per process, rebuilt only when its settings change
    _agent = None
    _agent_settings: Optional[AgentSettings] = None

    def __init__(self):
        self.settings = load_agent_settings(DEFAULT_SYSTEM_PROMPT)
        self.client = OpenAI(api_key=self.settings.api_key)
        self._ensure_agent()

    def _ensure_agent(self):
        """Reload settings from the environment and rebuild the agent if they changed."""
        self.settings = load_agent_settings(DEFAULT_SYSTEM_PROMPT)
        if Chat._agent is None or Chat._agent_settings != self.settings:
            Chat._agent = build_agent(self.settings)
            Chat._agent_settings = self.settings
        self.agent = Chat._agent

    # ==================== Agent turns ====================

    @staticmethod
    def _build_messages(message: str, history: Optional[List[Dict[str, Any]]]) -> List[BaseMessage]:
        """Convert stored history plus the new user message into LangChain messages."""
        messages: List[BaseMessage] = []
        for msg in history or []:
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
        messages.append(HumanMessage(content=message))
        return messages

    def _result_from_messages(self, messages: List[BaseMessage]) -> ChatResult:
        """Final reply text plus this turn's structured tool data."""
        final = messages[-1] if messages else None
        content = final.text.strip() if isinstance(final, AIMessage) and not final.tool_calls else ""
        response_type, data = self._extract_structured_result(messages)
        if not content:
            # e.g. a call limit stopped the agent mid tool loop
            logger.warning("Agent ended without a final text reply")
            content = FALLBACK_REPLY
        logger.info(f"Agent response ready ({len(content)} chars, response_type={response_type})")
        return ChatResult(content=content, response_type=response_type, data=data)

    def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_summary: Optional[str] = None,
    ) -> ChatResult:
        """Generate a response (and any structured tool data) using the agent."""
        try:
            self._ensure_agent()
            logger.info(f"Processing chat request: {message[:50]}")
            messages = self._build_messages(message, conversation_history)
            result = self.agent.invoke(
                {"messages": messages},
                context=AgentContext(conversation_summary=conversation_summary),
            )
            return self._result_from_messages(result["messages"])
        except Exception as e:
            logger.exception(f"Error in generate_response: {e}")
            return ChatResult(_friendly_error(e))

    def stream_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_summary: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Run the agent and yield events as it works:

            {"type": "status", "tool": name, "message": text}   a tool started
            {"type": "token", "text": delta}                    reply text as it is written
            {"type": "done", "result": ChatResult}              always last

        Tokens are a live preview. Text written before a tool call, or by a
        failed model before a fallback, can show up in them, so clients should
        replace the streamed text with the final `done` content.
        """
        try:
            self._ensure_agent()
            logger.info(f"Processing streamed chat request: {message[:50]}")
            messages = self._build_messages(message, conversation_history)
            for mode, chunk in self.agent.stream(
                {"messages": messages},
                context=AgentContext(conversation_summary=conversation_summary),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg, metadata = chunk
                    if metadata.get("langgraph_node") == "model" and msg.text:
                        yield {"type": "token", "text": msg.text}
                    continue

                # "updates": new messages from each step, used to build the final state
                for update in chunk.values():
                    new_messages = (update or {}).get("messages") if isinstance(update, dict) else None
                    for m in new_messages or []:
                        messages.append(m)
                        for call in getattr(m, "tool_calls", None) or []:
                            yield {
                                "type": "status",
                                "tool": call["name"],
                                "message": TOOL_STATUS_MESSAGES.get(call["name"], "Working on it…"),
                            }
            yield {"type": "done", "result": self._result_from_messages(messages)}
        except Exception as e:
            logger.exception(f"Error in stream_response: {e}")
            yield {"type": "done", "result": ChatResult(_friendly_error(e))}

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

    # ==================== Side tasks (small model) ====================

    def _complete(self, instructions: str, prompt: str, max_output_tokens: int) -> str:
        """One-shot text completion on the fallback (small, fast) model."""
        response = self.client.responses.create(
            model=self.settings.fallback_model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            store=False,
        )
        return response.output_text.strip()

    def generate_title(self, message) -> str:
        """Generate a title for the chat session based on initial user message"""
        try:
            title = self._complete(
                "You generate short, concise titles for chat conversations. "
                "Respond with only the title, no quotes or extra text.",
                f"Generate a concise and descriptive title (max 5 words) for a chat that starts with: {message}",
                max_output_tokens=200,
            ).strip('"')
            return title or "New Chat"
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return "New Chat"

    def summarize_conversation(
        self, previous_summary: Optional[str], messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Fold messages into the running conversation summary. None on failure."""
        lines = []
        for msg in messages:
            content = msg.get("content") or ""
            if msg.get("data") is not None:
                data = summarize_tool_data(msg.get("response_type"), msg["data"])
                content += f" [{msg.get('response_type')} shown: {json.dumps(data)[:1500]}]"
            lines.append(f"{msg.get('role')}: {content}")
        prompt = (
            f"Existing summary:\n{previous_summary or '(none)'}\n\n"
            "New messages:\n" + "\n".join(lines)
        )
        try:
            return self._complete(SUMMARY_INSTRUCTIONS, prompt, max_output_tokens=800) or None
        except Exception as e:
            logger.warning(f"Conversation summary failed: {e}")
            return None
