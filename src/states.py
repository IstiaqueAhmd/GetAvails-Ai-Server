"""
Agent State Definition for LangGraph

This module defines the state schema used by the LangGraph agent.
The state is passed between nodes and maintains conversation context.
"""

from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the tour guide agent.
    
    Attributes:
        messages: List of conversation messages (managed by LangGraph's add_messages reducer)
                  This includes HumanMessage, AIMessage, and ToolMessage objects.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
