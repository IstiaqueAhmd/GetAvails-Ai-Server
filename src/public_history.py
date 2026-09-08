"""
In-memory conversation history for the public (unauthenticated) chat endpoint.

The public landing-page assistant intentionally does NOT persist chat history to
the database. To still hold a coherent conversation within a session, we keep a
short, bounded history in process memory keyed by session id.

Bounds are important here because the endpoint is unauthenticated:
    - max_messages     caps history length per session (latency/token cost)
    - max_sessions     caps how many sessions we retain (memory ceiling, LRU)
    - ttl_seconds      drops sessions that have been idle too long

Note: this store lives in a single process. With multiple workers/replicas a
session's history is only guaranteed on the worker that served it, so requests
must be sticky per session for continuity. For a landing-page assistant that
tradeoff is acceptable; use a shared store (e.g. Redis) if that changes.
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, List


class InMemoryHistoryStore:
    def __init__(self, max_sessions: int = 1000, max_messages: int = 20, ttl_seconds: int = 3600):
        # session_id -> {"messages": [{"role", "content"}], "last_access": float}
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> List[Dict[str, str]]:
        """Return a copy of the message history for a session (may be empty)."""
        with self._lock:
            self._evict_expired()
            entry = self._store.get(session_id)
            if entry is None:
                return []
            entry["last_access"] = time.time()
            self._store.move_to_end(session_id)  # mark as most-recently used
            return list(entry["messages"])

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's history, enforcing all bounds."""
        with self._lock:
            self._evict_expired()
            entry = self._store.get(session_id)
            if entry is None:
                entry = {"messages": [], "last_access": time.time()}
                self._store[session_id] = entry

            entry["messages"].append({"role": role, "content": content})
            # Keep only the most recent messages
            if len(entry["messages"]) > self.max_messages:
                entry["messages"] = entry["messages"][-self.max_messages:]
            entry["last_access"] = time.time()
            self._store.move_to_end(session_id)

            # Evict least-recently-used sessions if over capacity
            while len(self._store) > self.max_sessions:
                self._store.popitem(last=False)

    def _evict_expired(self) -> None:
        """Drop sessions idle for longer than ttl_seconds. Caller holds the lock."""
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired = [
            sid for sid, entry in self._store.items()
            if now - entry["last_access"] > self.ttl_seconds
        ]
        for sid in expired:
            del self._store[sid]
