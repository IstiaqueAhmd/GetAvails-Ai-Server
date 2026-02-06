"""
Tools for the Tour Guide Agent

This module defines tools that the agent can use to help users.
Tools are decorated functions that the LLM can invoke when needed.
"""

from langchain_core.tools import tool
from typing import Optional
import os


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1
) -> str:
    """
    Search for available flights between two cities.
    
    Use this tool when users ask about flight availability, prices, or booking flights.
    
    Args:
        origin: Origin city or airport code (e.g., "New York" or "JFK")
        destination: Destination city or airport code (e.g., "Paris" or "CDG")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Optional return date in YYYY-MM-DD format for round trips
        adults: Number of adult passengers (default: 1)
    
    Returns:
        A string containing flight search results or instructions to configure the API.
    """
    # Check if Amadeus API credentials are configured
    amadeus_client_id = os.getenv("AMADEUS_CLIENT_ID")
    amadeus_client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    
    if not amadeus_client_id or not amadeus_client_secret:
        return (
            f"I found your flight search request:\n"
            f"- From: {origin}\n"
            f"- To: {destination}\n"
            f"- Departure: {departure_date}\n"
            f"- Return: {return_date or 'One-way'}\n"
            f"- Passengers: {adults} adult(s)\n\n"
            f"⚠️ The flight search API (Amadeus) is not yet configured. "
            f"Please contact the administrator to enable live flight searches. "
            f"In the meantime, I can help you with general travel advice, destination information, "
            f"or answer other questions about your trip!"
        )
    
    # TODO: Implement actual Amadeus API call when credentials are available
    # Example implementation:
    # from amadeus import Client, ResponseError
    # amadeus = Client(client_id=amadeus_client_id, client_secret=amadeus_client_secret)
    # response = amadeus.shopping.flight_offers_search.get(
    #     originLocationCode=origin,
    #     destinationLocationCode=destination,
    #     departureDate=departure_date,
    #     adults=adults
    # )
    
    return (
        f"Flight search results for {origin} → {destination}:\n"
        f"- Departure: {departure_date}\n"
        f"- Return: {return_date or 'One-way'}\n"
        f"- Passengers: {adults} adult(s)\n\n"
        f"[Amadeus API integration pending - placeholder response]"
    )


@tool
def get_destination_info(destination: str) -> str:
    """
    Get travel information about a destination.
    
    Use this tool when users ask about places to visit, travel tips,
    local customs, best times to visit, or general destination information.
    
    Args:
        destination: The city, country, or region to get information about
    
    Returns:
        A string with helpful travel information about the destination.
    """
    # This is a placeholder that returns generic advice
    # In the future, this could integrate with travel APIs or a knowledge base
    return (
        f"Here's some travel information about {destination}:\n\n"
        f"As your AI tour guide, I can help you explore {destination}! "
        f"I recommend researching:\n"
        f"- Popular attractions and landmarks\n"
        f"- Local cuisine and restaurants\n"
        f"- Best times to visit\n"
        f"- Cultural customs and etiquette\n"
        f"- Transportation options\n\n"
        f"Feel free to ask me specific questions about {destination} "
        f"and I'll do my best to help you plan an amazing trip!"
    )


# List of all available tools for the agent
TOOLS = [search_flights, get_destination_info]
