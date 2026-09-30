"""
Tools for the Ava Agent

This module defines tools that the agent can use to help users.
Tools are decorated functions that the LLM can invoke when needed.

Tools that produce structured results for the client use
response_format="content_and_artifact" and return a (content, artifact) pair:
    - content:  text the model reads to write its reply
    - artifact: structured data returned to the client in the /chat `data` field
Register each such tool's response type in TOOL_RESPONSE_TYPES at the bottom.

search_artists and search_venues call the GetAvails backend. web_search calls
Tavily for external information and returns text only (no artifact). generate_offer
still returns dummy data; replace its body with a real call and keep the
(content, artifact) return shape.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.tools import BaseTool, tool

# Configure logger for tools
logger = logging.getLogger("tour_guide.tools")


# ==================== GetAvails backend client ====================

_http: Optional[httpx.Client] = None
_http_lock = threading.Lock()


def _backend() -> httpx.Client:
    """Shared HTTP client for the GetAvails backend (created lazily, thread-safe)."""
    global _http
    if _http is None:
        with _http_lock:
            if _http is None:
                _http = httpx.Client(
                    base_url=os.getenv("GETAVAILS_API_BASE_URL", "https://backend.getavails.com/api/v1"),
                    timeout=float(os.getenv("GETAVAILS_API_TIMEOUT", "15")),
                    headers={"Accept": "application/json"},
                )
    return _http


# Genre names/slugs -> slug, fetched from the backend and cached.
_GENRE_CACHE_TTL_SECONDS = 3600
_genre_lookup: Optional[Dict[str, str]] = None
_genre_fetched_at = 0.0
_genre_lock = threading.Lock()


def _get_genre_lookup() -> Optional[Dict[str, str]]:
    """
    Map lowercase genre names and slugs to their slug, e.g. "classic rock"
    (name "Classic Rock") and "classic-rock" -> "classic-rock". Cached for an
    hour. Returns the last known lookup (or None) if the backend can't be reached.
    """
    global _genre_lookup, _genre_fetched_at
    with _genre_lock:
        if _genre_lookup is not None and time.time() - _genre_fetched_at < _GENRE_CACHE_TTL_SECONDS:
            return _genre_lookup

    try:
        response = _backend().get("/catalog/genres/")
        response.raise_for_status()
        results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"Could not fetch genres from backend: {e}")
        return _genre_lookup

    lookup: Dict[str, str] = {}
    for genre in results:
        slug = genre.get("slug")
        if not slug:
            continue
        lookup[slug.lower()] = slug
        if genre.get("name"):
            lookup[genre["name"].lower()] = slug

    with _genre_lock:
        _genre_lookup, _genre_fetched_at = lookup, time.time()
    return lookup


def _resolve_genres(genres: List[str]) -> Tuple[List[str], List[str], Optional[Dict[str, str]]]:
    """Resolve free-form genre names to backend slugs. Returns (slugs, unknown, lookup)."""
    lookup = _get_genre_lookup()
    slugs, unknown = [], []
    for raw in genres:
        key = raw.strip().lower()
        if not key:
            continue
        candidates = (key, re.sub(r"[\s_]+", "-", key))  # "hip hop" -> "hip-hop"
        if lookup is None:
            # Genre list unavailable: pass through a best-effort slug
            slugs.append(candidates[1])
            continue
        slug = next((lookup[c] for c in candidates if c in lookup), None)
        if slug:
            slugs.append(slug)
        else:
            unknown.append(raw)
    return slugs, unknown, lookup


def _summarize_artist(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact view of an artist row for the model. Full rows carry a year of
    availability data, which is for the client to render, not for the model.
    """
    if row.get("source") == "seatgeek":
        return {
            "id": row.get("id"),
            "source": "seatgeek",
            "name": row.get("name"),
            "genres": row.get("genres") or [],
            "popularity_score": row.get("score"),
        }
    user = row.get("user") or {}
    return {
        "id": row.get("id"),
        "source": "internal",
        "name": user.get("name"),
        "location": row.get("location"),
        "genres": [g.get("name") for g in row.get("genres") or [] if isinstance(g, dict)],
        "base_price_cents": row.get("base_price_cents"),
        "experience_years": row.get("experience_years"),
    }


def _summarize_venue(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of a venue row for the model (see _summarize_artist)."""
    # SeatGeek reports capacity 0 when it is unknown; don't present that as 0 seats
    capacity = row.get("capacity") or None
    if row.get("source") == "seatgeek":
        return {
            "id": row.get("id"),
            "source": "seatgeek",
            "name": row.get("name"),
            "address": row.get("address"),
            "city": row.get("city"),
            "state": row.get("state"),
            "capacity": capacity,
            "popularity_score": row.get("score"),
        }
    user = row.get("user") or {}
    return {
        "id": row.get("id"),
        "source": "internal",
        "name": user.get("name"),
        "address": row.get("address"),
        "location": row.get("location"),
        "capacity": capacity,
    }


def summarize_tool_data(response_type: Optional[str], data: Any) -> Any:
    """
    Compact form of a stored tool artifact, used when replaying chat history
    to the model. Falls back to the data as-is for types without a summary.
    """
    if response_type == "artists" and isinstance(data, list):
        return [_summarize_artist(row) for row in data if isinstance(row, dict)]
    if response_type == "venues" and isinstance(data, list):
        return [_summarize_venue(row) for row in data if isinstance(row, dict)]
    return data


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, 20))


# ==================== Tavily web search client ====================

TAVILY_API_URL = "https://api.tavily.com/search"
# Cap each page excerpt so a handful of results doesn't flood the model's context
_WEB_SNIPPET_MAX_CHARS = 800

_tavily_http: Optional[httpx.Client] = None


def _tavily() -> httpx.Client:
    """Shared HTTP client for Tavily (created lazily, thread-safe)."""
    global _tavily_http
    if _tavily_http is None:
        with _http_lock:
            if _tavily_http is None:
                _tavily_http = httpx.Client(timeout=float(os.getenv("TAVILY_API_TIMEOUT", "20")))
    return _tavily_http


# ==================== Tools ====================

@tool(response_format="content_and_artifact")
def search_artists(
    query: Optional[str] = None,
    genres: Optional[List[str]] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_miles: Optional[float] = None,
    available_on: Optional[str] = None,
    available_from: Optional[str] = None,
    available_to: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Search for artists on GetAvails. Results include GetAvails artists
    (source "internal") first, then SeatGeek performers (source "seatgeek").
    Use this when the user wants to find, browse, or get recommendations for
    artists, e.g. "find jazz artists near Chicago free on March 14".

    Args:
        query: Free-text search, such as an artist name or keywords
        genres: Genres to filter by, as slugs such as "rock", "hip-hop", "classic-rock"
        latitude: Latitude of the search center. If the user names a place, pass
            its approximate coordinates. Only applies together with longitude.
        longitude: Longitude of the search center. Only applies together with latitude.
        radius_miles: Search radius around the coordinates (defaults to 50 when omitted)
        available_on: A single date (YYYY-MM-DD) the artist must be free
        available_from: Start (YYYY-MM-DD) of a date range the artist must be free
        available_to: End (YYYY-MM-DD) of a date range the artist must be free
        limit: Number of artists to return (default 10, max 20)
        offset: Number of results to skip, used to show more results
    """
    logger.info(
        f"search_artists called: query={query}, genres={genres}, lat={latitude}, "
        f"lng={longitude}, radius={radius_miles}, on={available_on}, "
        f"from={available_from}, to={available_to}, limit={limit}, offset={offset}"
    )

    # The backend silently ignores unparseable dates, which would quietly widen
    # the search, so reject them here where the model can correct them.
    for label, value in (
        ("available_on", available_on),
        ("available_from", available_from),
        ("available_to", available_to),
    ):
        if value:
            try:
                date.fromisoformat(value)
            except ValueError:
                return f"Invalid {label} '{value}'. Use the YYYY-MM-DD format.", None

    params: Dict[str, Any] = {
        "limit": _clamp_limit(limit),
        "offset": max(0, offset),
    }
    if query:
        params["q"] = query
    if genres:
        slugs, unknown, lookup = _resolve_genres(genres)
        if unknown and lookup:
            valid = ", ".join(sorted(set(lookup.values())))
            return (
                f"Unknown genre(s): {', '.join(unknown)}. Valid genre slugs are: {valid}. "
                "Retry with valid slugs, or omit genres."
            ), None
        if slugs:
            params["genres"] = ",".join(slugs)
    if latitude is not None and longitude is not None:
        params["latitude"] = latitude
        params["longitude"] = longitude
        if radius_miles is not None:
            params["radius_miles"] = radius_miles
    for key, value in (
        ("available_on", available_on),
        ("available_from", available_from),
        ("available_to", available_to),
    ):
        if value:
            params[key] = value

    try:
        response = _backend().get("/catalog/artists/", params=params)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"Artist search request failed: {e}")
        return "Artist search is temporarily unavailable. Please try again shortly.", None

    artists = body.get("results") or []
    if not artists:
        # No artifact: the reply is a plain message rather than an empty card list
        return "No artists matched these filters.", None

    content = json.dumps({
        "total_matches": body.get("count"),
        "offset": params["offset"],
        "returned": len(artists),
        "artists": [_summarize_artist(row) for row in artists],
    })
    return content, artists


@tool(response_format="content_and_artifact")
def search_venues(
    query: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_miles: Optional[float] = None,
    limit: int = 10,
    offset: int = 0,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Search for venues on GetAvails. Results include GetAvails venues
    (source "internal") first, then SeatGeek venues (source "seatgeek").
    Use this when the user wants to find, browse, or get recommendations for
    venues, e.g. "venues in Austin" or "find the Paramount Theatre".

    This search cannot filter by date availability, genre, or capacity. If the
    user asks for those, search by name/location and say which results fit
    based on the details returned, without claiming the search filtered them.

    Args:
        query: Free-text search, such as a venue name or keywords
        latitude: Latitude of the search center. If the user names a place, pass
            its approximate coordinates. Only applies together with longitude.
        longitude: Longitude of the search center. Only applies together with latitude.
        radius_miles: Search radius around the coordinates (defaults to 50 when omitted)
        limit: Number of venues to return (default 10, max 20)
        offset: Number of results to skip, used to show more results
    """
    logger.info(
        f"search_venues called: query={query}, lat={latitude}, lng={longitude}, "
        f"radius={radius_miles}, limit={limit}, offset={offset}"
    )

    params: Dict[str, Any] = {
        "limit": _clamp_limit(limit),
        "offset": max(0, offset),
    }
    if query:
        params["q"] = query
    if latitude is not None and longitude is not None:
        params["latitude"] = latitude
        params["longitude"] = longitude
        if radius_miles is not None:
            params["radius_miles"] = radius_miles

    try:
        response = _backend().get("/catalog/venues/", params=params)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"Venue search request failed: {e}")
        return "Venue search is temporarily unavailable. Please try again shortly.", None

    venues = body.get("results") or []
    if not venues:
        # No artifact: the reply is a plain message rather than an empty card list
        return "No venues matched these filters.", None

    content = json.dumps({
        "total_matches": body.get("count"),
        "offset": params["offset"],
        "returned": len(venues),
        "venues": [_summarize_venue(row) for row in venues],
    })
    return content, venues


@tool(response_format="content_and_artifact")
def generate_offer(
    artist_id: str,
    venue_id: str,
    event_date: str,
    fee: float,
    currency: str = "USD",
    notes: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Draft a booking offer between an artist and a venue. This only creates a
    draft for the user to review; it does not send anything.

    Only use this once you know the artist, the venue, the event date, and the
    fee. If any of these are missing, ask the user instead of guessing. Use the
    `id` values returned by search_artists and search_venues.

    Args:
        artist_id: ID of the artist the offer is for (the `id` from search_artists)
        venue_id: ID of the venue hosting the event (the `id` from search_venues)
        event_date: Event date in YYYY-MM-DD format
        fee: Offered performance fee
        currency: Currency code for the fee (default: "USD")
        notes: Optional extra terms or details for the offer
    """
    logger.info(f"generate_offer called: artist={artist_id}, venue={venue_id}, date={event_date}, fee={fee}")

    offer = {
        "offer_id": f"offer_{uuid.uuid4().hex[:8]}",
        "status": "draft",
        "artist_id": artist_id,
        "venue_id": venue_id,
        "event_date": event_date,
        "fee": fee,
        "currency": currency.upper(),
        "notes": notes,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    return json.dumps(offer), offer


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the public web for information that GetAvails does not have, such
    as a venue's parking, box office hours, age policy, seating chart, history,
    or an artist's recent news, discography, or social links.

    Always check the platform first: for questions about a specific artist or
    venue, call search_artists or search_venues before this tool, and only use
    web_search when those results are missing or don't contain what the user
    asked for. Never use it to find or recommend artists or venues to book.

    Args:
        query: A specific search query. Include the full name and city of the
            artist or venue plus the detail wanted, e.g.
            "Paramount Theatre Austin TX parking and box office hours"
        max_results: Number of web results to return (default 5, max 10)
    """
    # Platform-first ordering is enforced by the platform_first middleware in src/agent.py
    logger.info(f"web_search called: query={query}, max_results={max_results}")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("web_search called but TAVILY_API_KEY is not set")
        return "Web search is not configured, so external information is unavailable."

    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": True,
    }
    try:
        response = _tavily().post(
            TAVILY_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"Web search request failed: {e}")
        return "Web search is temporarily unavailable. Please try again shortly."

    results = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "content": (r.get("content") or "")[:_WEB_SNIPPET_MAX_CHARS],
        }
        for r in body.get("results") or []
    ]
    if not results and not body.get("answer"):
        return "The web search found nothing relevant for this query."

    return json.dumps({
        "source": "web",
        "note": "External web results, not GetAvails data. Mention the source when using them.",
        "answer": body.get("answer"),
        "results": results,
    })


# List of all available tools for the agent
TOOLS: list[BaseTool] = [search_artists, search_venues, generate_offer, web_search]

# Maps a tool name to the /chat `response_type` used when that tool produced
# the reply's structured data. Tools not listed here yield "message".
# Values must match the ResponseType literal in src/schema.py.
TOOL_RESPONSE_TYPES: Dict[str, str] = {
    search_artists.name: "artists",
    search_venues.name: "venues",
    generate_offer.name: "offer",
}

# Progress text streamed to the client (/chat/stream) while a tool runs
TOOL_STATUS_MESSAGES: Dict[str, str] = {
    search_artists.name: "Searching artists on GetAvails…",
    search_venues.name: "Searching venues on GetAvails…",
    generate_offer.name: "Drafting the offer…",
    web_search.name: "Searching the web…",
}
