"""
Graph Nodes for the Tour Guide Agent

This module defines the nodes (processing steps) in the LangGraph workflow.
Each node is a function that takes the current state and returns an update.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from src.states import AgentState
from src.tools import TOOLS


def create_agent_node(model: str, system_prompt: str, max_tokens: int, api_key: str):
    """
    Factory function to create an agent node with the given configuration.
    
    Args:
        model: The OpenAI model to use (e.g., "gpt-4o-mini")
        system_prompt: The system prompt for the agent
        max_tokens: Maximum tokens for the response
        api_key: OpenAI API key
    
    Returns:
        A function that processes the agent state
    """
    # Create the LLM with tools bound
    llm = ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        api_key=api_key
    ).bind_tools(TOOLS)
    
    def agent_node(state: AgentState) -> dict:
        """
        The main agent node that processes messages and decides on actions.
        
        Args:
            state: The current agent state containing messages
        
        Returns:
            A dict with the updated messages
        """
        messages = state["messages"]
        
        # Add system message at the beginning if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        # Invoke the LLM
        response = llm.invoke(messages)
        
        return {"messages": [response]}
    
    return agent_node


def create_tool_node():
    """
    Create a tool node that executes tools when called by the agent.
    
    Returns:
        A ToolNode configured with the available tools
    """
    return ToolNode(TOOLS)
