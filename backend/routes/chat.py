"""
chat.py — Text chat API routes
Handles text-based legal queries with session memory.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import logging

from services.rag import process_legal_query
from services.llm import check_ollama_connection, stream_response, AVAILABLE_MODELS
from services.vector_store import get_collection_stats
from utils.session import create_session, clear_session
from utils.guardrails import build_system_prompt, check_query
from config import settings

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


# ── Request / Response Models ─────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model: Optional[str] = "llama3"

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is Section 420 BNS?",
                "session_id": "abc-123",
                "model": "llama3",
            }
        }


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    model_used: str
    documents_retrieved: int
    session_id: str
    blocked: bool


class SessionResponse(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main legal chat endpoint.
    Runs the full RAG pipeline and returns answer + sources.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if len(request.query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars).")

    # Use provided session or create new
    session_id = request.session_id or create_session()

    # Validate model
    model = request.model if request.model in AVAILABLE_MODELS else settings.llm_model

    try:
        result = await process_legal_query(
            query=request.query,
            session_id=session_id,
            model=model,
            ollama_url=settings.ollama_base_url,
            top_k=settings.top_k_results,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            collection_name=settings.chroma_collection_name,
            persist_dir=settings.chroma_persist_dir,
        )
        return ChatResponse(
            **result,
            session_id=session_id,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/session/new", response_model=SessionResponse)
async def new_session():
    """Create a new conversation session."""
    session_id = create_session()
    return SessionResponse(session_id=session_id)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a session's conversation history."""
    clear_session(session_id)
    return {"message": "Session cleared.", "session_id": session_id}


@router.get("/models")
async def list_models():
    """List available LLM models and Ollama connection status."""
    ollama_status = await check_ollama_connection(settings.ollama_base_url)
    return {
        "supported_models": AVAILABLE_MODELS,
        "default_model": settings.llm_model,
        "ollama_status": ollama_status,
    }


@router.get("/stats")
async def get_stats():
    """Get knowledge base statistics."""
    stats = get_collection_stats(
        collection_name=settings.chroma_collection_name,
        persist_dir=settings.chroma_persist_dir,
    )
    return stats


@router.get("/health")
async def health_check():
    """System health check."""
    ollama_status = await check_ollama_connection(settings.ollama_base_url)
    kb_stats = get_collection_stats(
        collection_name=settings.chroma_collection_name,
        persist_dir=settings.chroma_persist_dir,
    )
    return {
        "status": "ok",
        "ollama": ollama_status,
        "knowledge_base": kb_stats,
    }
