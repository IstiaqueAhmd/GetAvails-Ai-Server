"""
Tools for the Tour Guide Agent

This module defines tools that the agent can use to help users.
Tools are decorated functions that the LLM can invoke when needed.
"""

from langchain_core.tools import tool
from typing import Optional
import os
import logging
from datetime import datetime

# Configure logger for tools
logger = logging.getLogger("tour_guide.tools")


def get_amadeus_client():
    """
    Get an authenticated Amadeus client.
    
    Returns:
        Amadeus client instance or None if credentials are not configured.
    """
    from amadeus import Client
    
    api_key = os.getenv("AMADEUS_API_KEY")
    api_secret = os.getenv("AMADEUS_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Amadeus API credentials not configured")
        return None
    
    logger.info("Amadeus client initialized successfully")
    return Client(
        client_id=api_key,
        client_secret=api_secret
    )


def format_flight_offer(offer: dict, index: int) -> str:
    """
    Format a single flight offer into a readable string.
    
    Args:
        offer: The flight offer dictionary from Amadeus API
        index: The offer number for display
    
    Returns:
        A formatted string representation of the flight offer
    """
    try:
        price = offer.get("price", {})
        total_price = price.get("total", "N/A")
        currency = price.get("currency", "USD")
        
        itineraries = offer.get("itineraries", [])
        
        result = f"\n✈️ **Option {index}** - {currency} {total_price}\n"
        
        for i, itinerary in enumerate(itineraries):
            trip_type = "Outbound" if i == 0 else "Return"
            duration = itinerary.get("duration", "N/A").replace("PT", "").lower()
            segments = itinerary.get("segments", [])
            
            result += f"\n  📍 {trip_type} ({duration}):\n"
            
            for seg in segments:
                departure = seg.get("departure", {})
                arrival = seg.get("arrival", {})
                carrier = seg.get("carrierCode", "")
                flight_num = seg.get("number", "")
                
                dep_airport = departure.get("iataCode", "")
                dep_time = departure.get("at", "")
                arr_airport = arrival.get("iataCode", "")
                arr_time = arrival.get("at", "")
                
                # Format times nicely
                if dep_time:
                    dep_dt = datetime.fromisoformat(dep_time.replace("Z", "+00:00"))
                    dep_time = dep_dt.strftime("%H:%M")
                if arr_time:
                    arr_dt = datetime.fromisoformat(arr_time.replace("Z", "+00:00"))
                    arr_time = arr_dt.strftime("%H:%M")
                
                result += f"     {carrier}{flight_num}: {dep_airport} ({dep_time}) → {arr_airport} ({arr_time})\n"
        
        return result
        
    except Exception as e:
        return f"\n✈️ Option {index}: Error formatting flight details\n"


@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    max_results: int = 5
) -> str:
    """
    Search for flights. ONLY use this when the user EXPLICITLY asks to find/search flights 
    AND provides origin city, destination city, and departure date.
    
    DO NOT use this tool for:
    - General greetings or unclear messages
    - Questions about destinations without flight search intent
    - When the user hasn't provided specific travel details
    
    Args:
        origin: Origin airport IATA code (e.g., "JFK", "LAX", "LHR")
        destination: Destination airport IATA code (e.g., "CDG", "NRT", "SYD")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Optional return date in YYYY-MM-DD format for round trips
        adults: Number of adult passengers (default: 1)
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        A string containing flight search results with prices and schedules.
    """
    logger.info(f"search_flights called: {origin} -> {destination} on {departure_date}")
    
    try:
        amadeus = get_amadeus_client()
        
        if not amadeus:
            return (
                f"🔍 Flight Search Request:\n"
                f"- From: {origin}\n"
                f"- To: {destination}\n"
                f"- Departure: {departure_date}\n"
                f"- Return: {return_date or 'One-way'}\n"
                f"- Passengers: {adults} adult(s)\n\n"
                f"⚠️ **Amadeus API not configured**\n"
                f"The administrator needs to set AMADEUS_API_KEY and AMADEUS_API_SECRET "
                f"environment variables to enable live flight searches.\n\n"
                f"In the meantime, I can help you with:\n"
                f"- General travel advice for your destination\n"
                f"- Visa requirements and travel documents\n"
                f"- Packing tips and travel recommendations"
            )
        
        # Build search parameters
        search_params = {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "max": max_results,
            "currencyCode": "USD"
        }
        
        if return_date:
            search_params["returnDate"] = return_date
        
        # Make API call
        response = amadeus.shopping.flight_offers_search.get(**search_params)
        
        if not response.data:
            return (
                f"🔍 No flights found for:\n"
                f"- {origin} → {destination}\n"
                f"- Date: {departure_date}\n\n"
                f"Try adjusting your dates or check nearby airports."
            )
        
        # Format results
        trip_type = "Round-trip" if return_date else "One-way"
        result = (
            f"🛫 **Flight Search Results**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Route: {origin.upper()} → {destination.upper()}\n"
            f"Departure: {departure_date}\n"
        )
        
        if return_date:
            result += f"Return: {return_date}\n"
        
        result += f"Passengers: {adults} adult(s) | {trip_type}\n"
        result += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Add each flight offer
        for i, offer in enumerate(response.data[:max_results], 1):
            result += format_flight_offer(offer, i)
        
        result += (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 Prices shown are estimates and may vary. "
            f"Book directly with airlines for final pricing."
        )
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Flight search error: {error_msg}")
        
        # Handle common Amadeus API errors
        if "INVALID" in error_msg.upper() or "NOT FOUND" in error_msg.upper():
            return (
                f"❌ Invalid airport code or date format.\n\n"
                f"Please use:\n"
                f"- 3-letter IATA airport codes (e.g., JFK, LAX, LHR, CDG)\n"
                f"- Date format: YYYY-MM-DD (e.g., 2026-03-15)\n\n"
                f"Your request:\n"
                f"- Origin: {origin}\n"
                f"- Destination: {destination}\n"
                f"- Date: {departure_date}"
            )
        elif "RATE LIMIT" in error_msg.upper() or "429" in error_msg:
            return (
                "⏳ Flight search is temporarily busy. Please try again in a few moments."
            )
        else:
            return (
                f"❌ Unable to search flights at this time.\n"
                f"Please try again later or contact support if the issue persists."
            )


@tool
def get_airport_info(city_name: str) -> str:
    """
    Look up airport codes for a city. ONLY use this when the user EXPLICITLY asks 
    "what is the airport code for X" or "what airports are in X".
    
    DO NOT use this tool for:
    - General greetings or unclear messages
    - Flight searches (use search_flights instead)
    - General questions about a city
    
    Args:
        city_name: The city name to look up airports for (e.g., "Tokyo", "New York")
    
    Returns:
        Information about airports serving the given city.
    """
    logger.info(f"get_airport_info called for: {city_name}")
    
    try:
        amadeus = get_amadeus_client()
        
        if not amadeus:
            return (
                f"🔍 Looking for airports in/near: {city_name}\n\n"
                f"⚠️ **Amadeus API not configured**\n"
                f"Here are some common airport codes:\n"
                f"- New York: JFK, LGA, EWR\n"
                f"- Los Angeles: LAX\n"
                f"- London: LHR, LGW, STN\n"
                f"- Paris: CDG, ORY\n"
                f"- Tokyo: NRT, HND\n"
                f"- Dubai: DXB\n"
                f"- Singapore: SIN\n\n"
                f"Please provide the IATA code when searching for flights."
            )
        
        # Search for airports/cities
        response = amadeus.reference_data.locations.get(
            keyword=city_name,
            subType="AIRPORT,CITY"
        )
        
        if not response.data:
            return f"No airports found for '{city_name}'. Try a different spelling or nearby city."
        
        result = f"🛫 **Airports for '{city_name}'**\n\n"
        
        for location in response.data[:5]:
            name = location.get("name", "Unknown")
            iata = location.get("iataCode", "N/A")
            loc_type = location.get("subType", "").replace("_", " ").title()
            city = location.get("address", {}).get("cityName", "")
            country = location.get("address", {}).get("countryName", "")
            
            result += f"✈️ **{iata}** - {name}\n"
            result += f"   {city}, {country} ({loc_type})\n\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Airport info error: {e}")
        return f"Unable to look up airport information. Please try using common 3-letter airport codes like JFK, LAX, LHR."


@tool
def get_destination_info(destination: str) -> str:
    """
    Get travel tips for a destination. ONLY use this when the user asks specifically 
    about travel tips, things to do, or information about visiting a specific place.
    
    DO NOT use this tool for:
    - General greetings or unclear messages  
    - Flight searches
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
        f"**Need Flights?**\n"
        f"I can search for flights to {destination}! Just tell me:\n"
        f"- Your departure city\n"
        f"- Preferred travel dates\n"
        f"- Number of passengers\n\n"
        f"Feel free to ask me any specific questions about {destination}!"
    )


# List of all available tools for the agent
TOOLS = [search_flights, get_airport_info, get_destination_info]
