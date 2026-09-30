"""
Ava agent definition.

Builds the tool-calling agent with LangChain's `create_agent` on top of the
OpenAI Responses API (required for tool use on reasoning models such as
gpt-5.5 / gpt-6.x). Cross-cutting behaviour lives in middleware rather than
hand-written graph nodes:

    - dynamic system prompt   current date/time + running conversation summary
    - model fallback          switch to the fallback model if the primary errors
    - model retry             retry transient model errors with backoff
    - call limits             stop runaway tool loops with a friendly reply
    - platform first          web_search only after GetAvails has been searched
    - tool errors             unexpected tool exceptions go back to the model
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.tools import TOOLS, search_artists, search_venues, web_search

logger = logging.getLogger("tour_guide.agent")

# Model used when CHAT_MODEL is not set
DEFAULT_CHAT_MODEL = "gpt-5.5"
# Smaller model used as the fallback and for cheap side tasks (titles, summaries)
DEFAULT_FALLBACK_MODEL = "gpt-5.4-mini"

# Max model calls per user message (each tool round trip is one model call)
MODEL_CALL_LIMIT = 8
# Max tool calls per user message
TOOL_CALL_LIMIT = 12


@dataclass(frozen=True)
class AgentSettings:
    model: str
    fallback_model: str
    system_prompt: str
    max_tokens: int
    reasoning_effort: Optional[str]
    api_key: str


@dataclass
class AgentContext:
    """Per-request context passed to the agent at invoke time."""
    # Summary of conversation turns older than the messages sent in full
    conversation_summary: Optional[str] = None


def build_chat_model(
    model: str,
    api_key: str,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
) -> ChatOpenAI:
    """
    ChatOpenAI configured for the Responses API.

    store=False keeps conversations out of OpenAI's response storage; the
    encrypted reasoning is returned instead so it can be passed back between
    tool calls within a turn.
    """
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,  # sent as max_output_tokens on the Responses API
        use_responses_api=True,
        output_version="responses/v1",
        store=False,
        include=["reasoning.encrypted_content"],
        **kwargs,
    )


# ==================== Middleware ====================

def _platform_searched_this_turn(messages: Sequence[BaseMessage]) -> bool:
    """True if search_artists or search_venues ran since the latest user message."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return False
        if isinstance(m, ToolMessage) and m.name in (search_artists.name, search_venues.name):
            return True
    return False


@wrap_tool_call
def platform_first(request, handler):
    """Defer web_search until GetAvails itself has been searched this turn."""
    if request.tool_call["name"] == web_search.name and not _platform_searched_this_turn(
        request.state["messages"]
    ):
        logger.info("web_search deferred: no platform search yet this turn")
        return ToolMessage(
            content=(
                "Check GetAvails first: call search_venues or search_artists for the "
                "venue or artist in question. Call web_search again only if the "
                "platform results don't answer the user's question."
            ),
            tool_call_id=request.tool_call["id"],
            name=web_search.name,
        )
    return handler(request)


@before_model(can_jump_to=["end"])
def limit_model_calls(state, runtime):
    """
    End the turn with a friendly reply once the model has been called
    MODEL_CALL_LIMIT times for the current user message (a runaway tool loop).
    """
    calls = 0
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage):
            calls += 1
    if calls < MODEL_CALL_LIMIT:
        return None
    logger.warning(f"Model call limit ({MODEL_CALL_LIMIT}) reached; ending turn")
    return {
        "messages": [AIMessage(content=(
            "I wasn't able to finish that one. Could you rephrase or give me a "
            "bit more detail about what you're looking for?"
        ))],
        "jump_to": "end",
    }


def _tool_error_message(exc: Exception, request) -> str:
    """Hand unexpected tool exceptions back to the model instead of failing the turn."""
    logger.exception(f"Tool raised an exception: {exc}")
    return "This tool failed unexpectedly. Tell the user it's unavailable right now, or try another approach."


def _make_system_prompt(base_prompt: str):
    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        prompt = base_prompt
        context = request.runtime.context
        summary = getattr(context, "conversation_summary", None) if context else None
        if summary:
            prompt += (
                "\n\nSummary of earlier parts of this conversation (older messages "
                f"are not shown in full):\n{summary}"
            )
        # Kept last so the stable part of the prompt stays cacheable
        return f"{prompt}\n\nCurrent date and time: {now}"

    return system_prompt


# ==================== Agent ====================

def build_agent(settings: AgentSettings):
    """Create the compiled agent for the given settings."""
    logger.info(
        f"Building agent: model={settings.model}, fallback={settings.fallback_model}, "
        f"reasoning_effort={settings.reasoning_effort}, max_tokens={settings.max_tokens}"
    )
    primary = build_chat_model(
        settings.model, settings.api_key, settings.max_tokens, settings.reasoning_effort
    )
    middleware = [
        _make_system_prompt(settings.system_prompt),
        limit_model_calls,
        ToolCallLimitMiddleware(run_limit=TOOL_CALL_LIMIT),
    ]
    if settings.fallback_model and settings.fallback_model != settings.model:
        # Listed before the retry so the primary is retried before falling back
        middleware.append(ModelFallbackMiddleware(
            build_chat_model(settings.fallback_model, settings.api_key, settings.max_tokens)
        ))
    middleware += [
        ModelRetryMiddleware(max_retries=2, initial_delay=1.0),
        platform_first,
        ToolErrorMiddleware(_tool_error_message),
    ]
    return create_agent(
        primary,
        TOOLS,
        middleware=middleware,
        context_schema=AgentContext,
        name="ava",
    )


def load_agent_settings(default_system_prompt: str) -> AgentSettings:
    """Read agent settings from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return AgentSettings(
        model=os.getenv("CHAT_MODEL") or DEFAULT_CHAT_MODEL,
        fallback_model=os.getenv("CHAT_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL,
        system_prompt=os.getenv("CHAT_SYSTEM_PROMPT") or default_system_prompt,
        max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "1000")),
        # Empty string means "use the model's default"
        reasoning_effort=os.getenv("CHAT_REASONING_EFFORT", "low") or None,
        api_key=api_key,
    )
