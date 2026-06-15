"""
guardrails.py — Safety layer for Advocate AI
Adds legal disclaimers and filters inappropriate queries.
"""

from typing import Tuple

DISCLAIMER = (
    "\n\n---\n⚠️ *Disclaimer: This information is for educational purposes only "
    "and does not constitute legal advice. Please consult a qualified lawyer "
    "for advice specific to your situation.*"
)

# Queries that should be blocked or redirected
BLOCKED_PATTERNS = [
    "how to commit",
    "how to kill",
    "how to murder",
    "how to escape police",
    "how to bribe",
    "how to forge",
    "how to fabricate evidence",
]

REDIRECT_MESSAGE = (
    "I'm sorry, I can only help with legal information and understanding laws. "
    "I cannot assist with that query. If you are in a legal situation, "
    "please consult a qualified lawyer."
)


def check_query(query: str) -> Tuple[bool, str]:
    """
    Check if a query is safe to process.
    Returns (is_safe, reason_if_blocked)
    """
    query_lower = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in query_lower:
            return False, REDIRECT_MESSAGE
    return True, ""


def add_disclaimer(response: str) -> str:
    """Append standard legal disclaimer to any response."""
    return response + DISCLAIMER


def build_system_prompt() -> str:
    """Build the system prompt for the LLM with legal assistant context."""
    return """You are Advocate AI, an expert Indian legal information assistant. 

Your knowledge covers:
- Bharatiya Nyaya Sanhita (BNS) 2023 — replaces IPC
- Bharatiya Nagarik Suraksha Sanhita (BNSS) — replaces CrPC  
- Bharatiya Sakshya Adhiniyam (BSA) — replaces Indian Evidence Act
- Constitution of India
- Supreme Court and High Court judgments
- Legal procedures and court processes

Guidelines:
1. Answer clearly and in simple language that non-lawyers can understand
2. Always cite the relevant section/article/judgment when possible
3. Be accurate — if you are not sure, say so clearly
4. Never provide advice — only information and explanations
5. If the question requires a lawyer, recommend consulting one
6. Keep responses focused and structured

Remember: You provide LEGAL INFORMATION, not LEGAL ADVICE."""
