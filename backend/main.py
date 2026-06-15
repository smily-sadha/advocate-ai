"""
main.py — Advocate AI FastAPI Application Entry Point
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from routes.chat import router as chat_router
from routes.voice import router as voice_router
from routes.documents import router as documents_router

# ── Logging Setup ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/advocate_ai.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── App Lifecycle ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 50)
    logger.info(f"  {settings.app_name} Starting Up")
    logger.info("=" * 50)
    logger.info(f"  LLM Model     : {settings.llm_model}")
    logger.info(f"  Ollama URL    : {settings.ollama_base_url}")
    logger.info(f"  Whisper Model : {settings.whisper_model}")
    logger.info(f"  ChromaDB Dir  : {settings.chroma_persist_dir}")
    logger.info("=" * 50)

    # Pre-load embedding model on startup for faster first query
    try:
        from services.embeddings import get_embedding_model
        get_embedding_model(settings.embedding_model)
        logger.info("Embedding model pre-loaded.")
    except Exception as e:
        logger.warning(f"Could not pre-load embedding model: {e}")

    yield

    logger.info("Advocate AI shutting down.")


# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="Advocate AI",
    description="Voice-enabled Indian Legal Information Assistant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(voice_router)
app.include_router(documents_router)

# ── Serve Frontend ─────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "app": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/health",
        }


# ── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
