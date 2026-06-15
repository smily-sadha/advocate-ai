"""
stt.py — Speech-to-Text service using OpenAI Whisper
Converts user voice audio to text for the RAG pipeline.
"""

import whisper
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_whisper_model = None


def get_whisper_model(model_size: str = "large-v3"):
    """Singleton loader for Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {model_size}")
        _whisper_model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully.")
    return _whisper_model


def transcribe_audio(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = None,  # None = auto-detect (supports Hindi, English, etc.)
) -> dict:
    """
    Transcribe an audio file to text.
    
    Args:
        audio_path: Path to the audio file (wav, mp3, m4a, etc.)
        model_size: Whisper model size (tiny/base/small/medium/large)
        language: Language code (e.g., "en", "hi") or None for auto-detect
    
    Returns:
        dict with 'text' (transcription) and 'language' (detected language)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model(model_size)

    logger.info(f"Transcribing: {audio_path}")

    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    transcription = result["text"].strip()
    detected_language = result.get("language", "unknown")

    logger.info(f"Transcription complete. Language: {detected_language}")
    logger.info(f"Text: {transcription[:100]}...")

    return {
        "text": transcription,
        "language": detected_language,
    }


def save_uploaded_audio(audio_bytes: bytes, upload_dir: str, filename: str) -> str:
    """Save uploaded audio bytes to disk and return the file path."""
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path
