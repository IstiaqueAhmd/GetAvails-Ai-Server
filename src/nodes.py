"""
Graph Nodes for the Tour Guide Agent

This module defines the nodes (processing steps) in the LangGraph workflow.
Each node is a function that takes the current state and returns an update.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
import logging
from datetime import datetime

from src.states import AgentState
from src.tools import TOOLS

# Configure logger for nodes
logger = logging.getLogger("tour_guide.nodes")


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
    # Create the LLM, binding tools only if any exist (OpenAI rejects an empty tools list)
    llm = ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        api_key=api_key
    )
    if TOOLS:
        llm = llm.bind_tools(TOOLS)
    
    def agent_node(state: AgentState) -> dict:
        """
        The main agent node that processes messages and decides on actions.
        
        Args:
            state: The current agent state containing messages
        
        Returns:
            A dict with the updated messages
        """
        messages = state["messages"]
        logger.info(f"Agent node processing {len(messages)} messages")
        
        # Get current date and time
        now = datetime.now()
        current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")
        
        # Create system message with current date/time injected
        full_system_prompt = f"{system_prompt}\n\nCurrent date and time: {current_datetime}"
        
        # Add system message at the beginning if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=full_system_prompt)] + list(messages)
        else:
            # Update the existing system message with current time
            messages = [SystemMessage(content=full_system_prompt)] + list(messages[1:])
        
        # Invoke the LLM
        logger.debug("Invoking LLM for response")
        response = llm.invoke(messages)
        
        # Log if tool calls are present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc['name'] for tc in response.tool_calls]
            logger.info(f"Agent decided to use tools: {tool_names}")
        else:
            logger.info("Agent generated final response (no tools)")
        
        return {"messages": [response]}
    
    return agent_node


def create_tool_node():
    """
    Create a tool node that executes tools when called by the agent.
    
    Returns:
        A ToolNode configured with the available tools
    """
    logger.info(f"Tool node created with {len(TOOLS)} tools: {[t.name for t in TOOLS]}")
    return ToolNode(TOOLS)
