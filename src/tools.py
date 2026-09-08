"""
Tools for the Tour Guide Agent

This module defines tools that the agent can use to help users.
Tools are decorated functions that the LLM can invoke when needed.
"""

from langchain_core.tools import tool
import logging

# Configure logger for tools
logger = logging.getLogger("tour_guide.tools")


@tool
def get_destination_info(destination: str) -> str:
    """
    Get travel tips for a destination. ONLY use this when the user asks specifically
    about travel tips, things to do, or information about visiting a specific place.

    DO NOT use this tool for:
    - General greetings or unclear messages
    - Airport code lookups

    Args:
        destination: The specific city or country the user asked about

    Returns:
        Travel tips and information about the destination.
    """
    logger.info(f"get_destination_info called for: {destination}")

    return (
        f"🌍 **Travel Guide: {destination}**\n\n"
        f"As your AI tour guide, I'd be happy to help you explore {destination}!\n\n"
        f"**Things to Consider:**\n"
        f"📍 Popular attractions and landmarks\n"
        f"🍽️ Local cuisine and restaurants\n"
        f"🗓️ Best times to visit\n"
        f"🎭 Cultural customs and etiquette\n"
        f"🚗 Transportation options\n"
        f"💰 Budget considerations\n\n"
        f"Feel free to ask me any specific questions about {destination}!"
    )


# List of all available tools for the agent
TOOLS = [get_destination_info]
