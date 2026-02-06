"""
Agent Graph for the Tour Guide

This module builds and compiles the LangGraph workflow that powers the agentic chatbot.
The graph connects the agent node and tool node with conditional routing.
"""

from langgraph.graph import StateGraph, END

from src.states import AgentState
from src.nodes import create_agent_node, create_tool_node
from src.edges import should_continue


def create_agent_graph(model: str, system_prompt: str, max_tokens: int, api_key: str):
    """
    Create and compile the agent graph with the given configuration.
    
    Args:
        model: The OpenAI model to use (e.g., "gpt-4o-mini")
        system_prompt: The system prompt for the agent
        max_tokens: Maximum tokens for the response
        api_key: OpenAI API key
    
    Returns:
        A compiled LangGraph workflow ready for invocation
    """
    # Create nodes
    agent_node = create_agent_node(model, system_prompt, max_tokens, api_key)
    tool_node = create_tool_node()
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    
    # Tools always return to agent
    workflow.add_edge("tools", "agent")
    
    # Compile and return the graph
    return workflow.compile()
