"""
llm.py — Ollama LLM service for Advocate AI
Supports switchable Llama3 / Mistral models via Ollama.
"""

import httpx
import json
import logging
from typing import List, Dict, AsyncGenerator

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["llama3", "mistral"]


async def generate_response(
    prompt: str,
    history: List[Dict] = None,
    model: str = "llama3",
    system_prompt: str = "",
    ollama_url: str = "http://localhost:11434",
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """
    Generate a response using Ollama.
    Supports conversation history for follow-up questions.
    """
    if model not in AVAILABLE_MODELS:
        logger.warning(f"Unknown model '{model}', defaulting to llama3")
        model = "llama3"

    # Build messages list
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add conversation history
    if history:
        messages.extend(history)

    # Add current user query
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama. Is it running?")
        raise ConnectionError(
            "Cannot connect to Ollama. Please run 'ollama serve' and make sure "
            "your model is pulled (ollama pull llama3 / ollama pull mistral)."
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise


async def stream_response(
    prompt: str,
    history: List[Dict] = None,
    model: str = "llama3",
    system_prompt: str = "",
    ollama_url: str = "http://localhost:11434",
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """
    Stream a response token-by-token using Ollama.
    Used for real-time UI streaming.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{ollama_url}/api/chat", json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if not data.get("done", False):
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        continue


async def check_ollama_connection(ollama_url: str = "http://localhost:11434") -> Dict:
    """Check if Ollama is running and which models are available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            available = [m["name"] for m in data.get("models", [])]
            return {
                "connected": True,
                "available_models": available,
            }
    except Exception as e:
        return {"connected": False, "error": str(e)}
