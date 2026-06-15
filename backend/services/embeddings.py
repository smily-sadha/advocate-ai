"""
embeddings.py — Embedding service using sentence-transformers (BGE model)
BGE is optimized for retrieval tasks and works well for legal text.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

_model_instance = None


def get_embedding_model(model_name: str = "BAAI/bge-base-en-v1.5") -> SentenceTransformer:
    """Singleton loader for the embedding model."""
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading embedding model: {model_name}")
        _model_instance = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    return _model_instance


def embed_texts(texts: List[str], model_name: str = "BAAI/bge-base-en-v1.5") -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    BGE models work best with a query prefix for retrieval tasks.
    """
    model = get_embedding_model(model_name)
    # BGE recommendation: prefix "Represent this sentence:" for passages
    prefixed = [f"Represent this sentence: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str, model_name: str = "BAAI/bge-base-en-v1.5") -> List[float]:
    """
    Generate embedding for a single query.
    BGE recommendation: prefix "query: " for search queries.
    """
    model = get_embedding_model(model_name)
    prefixed = f"query: {query}"
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return embedding[0].tolist()
