"""
session.py — Conversation memory manager for Advocate AI
Maintains per-session chat history for follow-up questions.
"""

from typing import List, Dict
from datetime import datetime, timedelta
import uuid

# In-memory session store (use Redis for production scaling)
_sessions: Dict[str, Dict] = {}

SESSION_TIMEOUT_MINUTES = 30


def create_session() -> str:
    """Create a new session and return session ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "history": [],
        "created_at": datetime.utcnow(),
        "last_active": datetime.utcnow(),
    }
    return session_id


def get_history(session_id: str, max_turns: int = 5) -> List[Dict]:
    """Get conversation history for a session (last N turns)."""
    _cleanup_expired_sessions()
    session = _sessions.get(session_id)
    if not session:
        return []
    # Return last max_turns exchanges (each turn = user + assistant message)
    history = session["history"]
    return history[-(max_turns * 2):]


def add_turn(session_id: str, user_message: str, assistant_message: str):
    """Add a conversation turn to session history."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow(),
        }

    session = _sessions[session_id]
    session["history"].append({"role": "user", "content": user_message})
    session["history"].append({"role": "assistant", "content": assistant_message})
    session["last_active"] = datetime.utcnow()


def clear_session(session_id: str):
    """Clear a session's history."""
    if session_id in _sessions:
        _sessions[session_id]["history"] = []


def _cleanup_expired_sessions():
    """Remove sessions that have been inactive too long."""
    cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    expired = [
        sid
        for sid, data in _sessions.items()
        if data["last_active"] < cutoff
    ]
    for sid in expired:
        del _sessions[sid]


def format_history_for_ollama(history: List[Dict]) -> List[Dict]:
    """Format history into Ollama message format."""
    return [{"role": msg["role"], "content": msg["content"]} for msg in history]
