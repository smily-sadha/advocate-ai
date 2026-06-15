"""
config.py — Central configuration for Advocate AI
Loads all settings from .env file
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os


class Settings(BaseSettings):
    # LLM
    llm_model: str = Field(default="llama3", env="LLM_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")

    # Embeddings
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5", env="EMBEDDING_MODEL"
    )

    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="advocate_legal_docs", env="CHROMA_COLLECTION_NAME"
    )

    # RAG
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # Speech
    whisper_model: str = Field(default="large-v3", env="WHISPER_MODEL")
    # TTS engine: "gtts" (Google TTS, online, multilingual, mp3) or "coqui" (local, wav)
    tts_engine: str = Field(default="gtts", env="TTS_ENGINE")
    tts_lang: str = Field(default="en", env="TTS_LANG")
    tts_model: str = Field(
        default="tts_models/en/ljspeech/tacotron2-DDC", env="TTS_MODEL"
    )
    audio_upload_dir: str = Field(default="./data/audio_uploads", env="AUDIO_UPLOAD_DIR")
    audio_output_dir: str = Field(default="./data/audio_outputs", env="AUDIO_OUTPUT_DIR")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")

    # Session
    max_history_turns: int = Field(default=5, env="MAX_HISTORY_TURNS")

    # App
    app_name: str = Field(default="Advocate AI", env="APP_NAME")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self):
        """Create required directories if they don't exist"""
        dirs = [
            self.chroma_persist_dir,
            self.audio_upload_dir,
            self.audio_output_dir,
            "./data/raw",
            "./data/processed",
            "./logs",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_dirs()
