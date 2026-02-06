"""
Graph Edges for the Tour Guide Agent

This module defines the conditional routing logic for the LangGraph workflow.
Edges determine how the graph flows between nodes.
"""

from typing import Literal
from langchain_core.messages import AIMessage

from src.states import AgentState


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Determine whether the agent should continue to tools or end.
    
    This function checks if the last message from the LLM contains tool calls.
    If it does, we route to the tools node. Otherwise, we end the conversation turn.
    
    Args:
        state: The current agent state containing messages
    
    Returns:
        "tools" if the agent wants to use tools, "__end__" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message is an AI message with tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # No tool calls, end the conversation turn
    return "__end__"
