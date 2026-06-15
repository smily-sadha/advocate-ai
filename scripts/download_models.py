"""
download_models.py — Download all required ML models
Run ONCE before starting the server.

Downloads:
- Whisper STT model (base by default)
- BGE embedding model
- Coqui TTS model

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --whisper-size small --skip-tts
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_whisper(model_size: str = "base"):
    """Download Whisper STT model."""
    logger.info(f"Downloading Whisper '{model_size}' model...")
    try:
        import whisper
        model = whisper.load_model(model_size)
        logger.info(f"Whisper '{model_size}' downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Whisper download failed: {e}")
        return False


def download_embeddings(model_name: str = "BAAI/bge-base-en-v1.5"):
    """Download BGE embedding model from HuggingFace."""
    logger.info(f"Downloading embedding model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        # Test it
        test_embedding = model.encode(["test legal query"])
        logger.info(
            f"Embedding model downloaded. Dimension: {len(test_embedding[0])}"
        )
        return True
    except Exception as e:
        logger.error(f"Embedding model download failed: {e}")
        return False


def download_tts(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
    """Download Coqui TTS model."""
    logger.info(f"Downloading Coqui TTS model: {model_name}")
    try:
        from TTS.api import TTS
        tts = TTS(model_name=model_name, progress_bar=True)
        logger.info("Coqui TTS model downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"TTS model download failed: {e}")
        logger.warning("TTS is optional. Text responses will still work.")
        return False


def check_ollama():
    """Check if Ollama is installed and models are available."""
    import subprocess
    logger.info("Checking Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("Ollama is installed and running.")
            output = result.stdout
            has_llama3 = "llama3" in output.lower()
            has_mistral = "mistral" in output.lower()

            if not has_llama3:
                logger.warning("llama3 not found. Run: ollama pull llama3")
            else:
                logger.info("llama3 model is available.")

            if not has_mistral:
                logger.warning("mistral not found. Run: ollama pull mistral")
            else:
                logger.info("mistral model is available.")
        else:
            logger.warning("Ollama not running. Start it with: ollama serve")
    except FileNotFoundError:
        logger.error("Ollama not installed. Download from: https://ollama.ai")
    except Exception as e:
        logger.warning(f"Could not check Ollama: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download required ML models for Advocate AI"
    )
    parser.add_argument(
        "--whisper-size",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip Coqui TTS download",
    )
    args = parser.parse_args()

    logger.info("=" * 40)
    logger.info("  Advocate AI — Model Downloader")
    logger.info("=" * 40)

    results = {}

    # 1. Check Ollama
    check_ollama()

    # 2. Whisper STT
    results["whisper"] = download_whisper(args.whisper_size)

    # 3. BGE Embeddings
    results["embeddings"] = download_embeddings()

    # 4. Coqui TTS
    if not args.skip_tts:
        results["tts"] = download_tts()
    else:
        logger.info("Skipping TTS download (--skip-tts flag).")

    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("  DOWNLOAD SUMMARY")
    logger.info("=" * 40)
    for component, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        logger.info(f"  {component.upper():<15} {status}")
    logger.info("=" * 40)
    logger.info("\nNext step: python scripts/ingest.py")
