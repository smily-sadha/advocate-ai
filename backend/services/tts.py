"""
tts.py — Text-to-Speech service
Supports two engines:
  - "gtts"  : Google Text-to-Speech (online, multilingual, fast, outputs .mp3)
  - "coqui" : Coqui TTS (fully local/offline, outputs .wav)
Converts LLM responses to spoken audio for voice interaction.
"""

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_tts_model = None  # cached Coqui model (only used for the "coqui" engine)


def get_tts_model(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
    """Singleton loader for Coqui TTS model."""
    global _tts_model
    if _tts_model is None:
        try:
            from TTS.api import TTS
            logger.info(f"Loading Coqui TTS model: {model_name}")
            _tts_model = TTS(model_name=model_name, progress_bar=False)
            logger.info("Coqui TTS model loaded successfully.")
        except ImportError:
            logger.error("Coqui TTS not installed. Run: pip install TTS")
            raise
    return _tts_model


def synthesize_speech(
    text: str,
    output_dir: str = "./data/audio_outputs",
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
    filename: str = None,
    engine: str = "gtts",
    lang: str = "en",
) -> str:
    """
    Convert text to speech and save to disk.

    Args:
        text: Text to convert to speech
        output_dir: Directory to save audio
        model_name: Coqui TTS model name (only used when engine="coqui")
        filename: Output filename (auto-generated if None)
        engine: "gtts" (online, mp3) or "coqui" (local, wav)
        lang: Language code for gTTS (e.g. "en", "hi")

    Returns:
        Path to the generated audio file (.mp3 for gtts, .wav for coqui)
    """
    clean_text = _clean_text_for_speech(text)

    if not clean_text:
        raise ValueError("Text is empty after cleaning.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    engine = (engine or "gtts").lower()

    if engine == "gtts":
        return _synthesize_gtts(clean_text, output_dir, filename, lang)
    elif engine == "coqui":
        return _synthesize_coqui(clean_text, output_dir, filename, model_name)
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r} (use 'gtts' or 'coqui')")


def _synthesize_gtts(text: str, output_dir: str, filename: str, lang: str) -> str:
    """Synthesize speech with Google TTS (gTTS). Outputs an .mp3 file."""
    try:
        from gtts import gTTS
    except ImportError:
        logger.error("gTTS not installed. Run: pip install gTTS")
        raise

    if not filename:
        filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
    if not filename.lower().endswith(".mp3"):
        filename += ".mp3"

    output_path = os.path.join(output_dir, filename)

    tts = gTTS(text=text, lang=lang or "en")
    tts.save(output_path)

    logger.info(f"Speech synthesized (gTTS, lang={lang}): {output_path}")
    return output_path


def _synthesize_coqui(text: str, output_dir: str, filename: str, model_name: str) -> str:
    """Synthesize speech with a local Coqui TTS model. Outputs a .wav file."""
    if not filename:
        filename = f"response_{uuid.uuid4().hex[:8]}.wav"
    if not filename.lower().endswith(".wav"):
        filename += ".wav"

    output_path = os.path.join(output_dir, filename)

    tts = get_tts_model(model_name)
    tts.tts_to_file(text=text, file_path=output_path)

    logger.info(f"Speech synthesized (Coqui): {output_path}")
    return output_path


def _clean_text_for_speech(text: str, max_chars: int = 1000) -> str:
    """
    Clean text for TTS:
    - Remove markdown formatting
    - Remove disclaimer (too long for speech)
    - Truncate to manageable length
    """
    import re

    # Remove disclaimer block
    text = re.sub(r"\n---\n⚠️.*", "", text, flags=re.DOTALL)

    # Remove markdown bold/italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bullet points
    text = re.sub(r"^[-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "... For the complete answer, please read the text response."

    return text.strip()
