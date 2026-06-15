"""
voice.py — Voice API routes (STT + TTS)
Handles voice input (Whisper) and voice output (Coqui TTS).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from typing import Optional
import logging
import uuid
import os

from services.stt import transcribe_audio, save_uploaded_audio
from services.tts import synthesize_speech
from services.rag import process_legal_query
from utils.session import create_session
from config import settings

router = APIRouter(prefix="/api/voice", tags=["voice"])
logger = logging.getLogger(__name__)


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, m4a, webm)"),
):
    """
    Transcribe uploaded audio to text using Whisper.
    Returns the transcribed text.
    """
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg"]

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    if len(audio_bytes) > 25 * 1024 * 1024:  # 25MB limit
        raise HTTPException(status_code=400, detail="Audio file too large (max 25MB).")

    filename = f"upload_{uuid.uuid4().hex[:8]}_{audio.filename}"
    audio_path = save_uploaded_audio(audio_bytes, settings.audio_upload_dir, filename)

    try:
        result = transcribe_audio(
            audio_path=audio_path,
            model_size=settings.whisper_model,
        )
        return {
            "text": result["text"],
            "language": result["language"],
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(audio_path):
            os.remove(audio_path)


@router.post("/ask")
async def voice_ask(
    audio: UploadFile = File(..., description="Audio file with your legal question"),
    session_id: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default="llama3"),
):
    """
    Full voice pipeline:
    1. Transcribe audio → text (Whisper)
    2. Process legal query (RAG + LLM)
    3. Convert answer → speech (Coqui TTS)
    Returns: JSON with text answer + audio file path
    """
    # Step 1: Transcribe
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    filename = f"voice_{uuid.uuid4().hex[:8]}_{audio.filename}"
    audio_path = save_uploaded_audio(audio_bytes, settings.audio_upload_dir, filename)

    try:
        transcription = transcribe_audio(
            audio_path=audio_path,
            model_size=settings.whisper_model,
        )
        query_text = transcription["text"]
        logger.info(f"Voice query transcribed: {query_text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # Step 2: RAG pipeline
    session_id = session_id or create_session()
    model = model if model in ["llama3", "mistral"] else "llama3"

    try:
        rag_result = await process_legal_query(
            query=query_text,
            session_id=session_id,
            model=model,
            ollama_url=settings.ollama_base_url,
            top_k=settings.top_k_results,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            collection_name=settings.chroma_collection_name,
            persist_dir=settings.chroma_persist_dir,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Step 3: TTS
    try:
        audio_output_path = synthesize_speech(
            text=rag_result["answer"],
            output_dir=settings.audio_output_dir,
            model_name=settings.tts_model,
            engine=settings.tts_engine,
            lang=settings.tts_lang,
        )
        audio_filename = os.path.basename(audio_output_path)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        audio_filename = None  # Return text only if TTS fails

    return {
        "transcribed_query": query_text,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "model_used": rag_result["model_used"],
        "session_id": session_id,
        "audio_url": f"/api/voice/audio/{audio_filename}" if audio_filename else None,
    }


@router.post("/speak")
async def text_to_speech(text: str = Form(...)):
    """
    Convert text to speech.
    Returns path to the generated audio file.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_path = synthesize_speech(
            text=text,
            output_dir=settings.audio_output_dir,
            model_name=settings.tts_model,
            engine=settings.tts_engine,
            lang=settings.tts_lang,
        )
        filename = os.path.basename(audio_path)
        return {"audio_url": f"/api/voice/audio/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve a generated audio file."""
    # Security: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = os.path.join(settings.audio_output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )
