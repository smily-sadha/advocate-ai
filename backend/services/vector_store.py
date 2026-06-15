"""
vector_store.py — ChromaDB vector store service
Handles storing and retrieving legal document embeddings.
Includes hybrid search: vector similarity + keyword (BM25-style) filtering.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Tuple
import logging
import re

from services.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_chroma_client(persist_dir: str = "./data/chroma_db"):
    """Get or create ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client initialized at: {persist_dir}")
    return _client


def get_collection(
    collection_name: str = "advocate_legal_docs",
    persist_dir: str = "./data/chroma_db",
):
    """Get or create the legal documents collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client(persist_dir)
        _collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Collection '{collection_name}' ready. "
            f"Total documents: {_collection.count()}"
        )
    return _collection


def add_documents(
    chunks: List[Dict],  # [{"content": str, "metadata": dict}]
    collection_name: str = "advocate_legal_docs",
    persist_dir: str = "./data/chroma_db",
    batch_size: int = 50,
):
    """Add document chunks to ChromaDB with their embeddings."""
    collection = get_collection(collection_name, persist_dir)

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Generate IDs based on source + index
    ids = [
        f"{m.get('source', 'doc')}_{m.get('chunk_index', i)}"
        for i, m in enumerate(metadatas)
    ]

    # Batch insert to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        embeddings = embed_texts(batch_texts)

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        logger.info(f"Ingested batch {i // batch_size + 1} ({len(batch_texts)} chunks)")

    logger.info(f"Total documents in collection: {collection.count()}")


def hybrid_search(
    query: str,
    top_k: int = 5,
    collection_name: str = "advocate_legal_docs",
    persist_dir: str = "./data/chroma_db",
) -> List[Dict]:
    """
    Hybrid search: vector similarity + keyword boosting.
    Legal queries often have exact section numbers that need keyword matching.
    """
    collection = get_collection(collection_name, persist_dir)

    if collection.count() == 0:
        logger.warning("Vector store is empty. Please run ingest.py first.")
        return []

    # Vector search
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, collection.count()),  # Over-fetch for reranking
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # Keyword boost: boost docs that contain exact terms from query
    keywords = extract_legal_keywords(query)
    scored_results = []

    for doc, meta, dist in zip(docs, metas, distances):
        vector_score = 1 - dist  # Convert distance to similarity
        keyword_score = sum(
            1 for kw in keywords if kw.lower() in doc.lower()
        ) * 0.1  # Small boost per keyword match
        final_score = vector_score + keyword_score

        scored_results.append(
            {
                "content": doc,
                "metadata": meta,
                "score": final_score,
            }
        )

    # Sort by combined score and return top_k
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]


def extract_legal_keywords(query: str) -> List[str]:
    """Extract important legal keywords from a query for boosting."""
    keywords = []

    # Section numbers (e.g., Section 302, Article 21)
    section_matches = re.findall(r"(?:section|article|clause)\s+\d+[a-z]?", query, re.I)
    keywords.extend(section_matches)

    # Act names
    act_matches = re.findall(r"\b(?:IPC|BNS|BNSS|BSA|CrPC|IEA|NIA|CPC)\b", query)
    keywords.extend(act_matches)

    # Important legal terms
    legal_terms = [
        "bail", "FIR", "cognizable", "bailable", "arrest", "warrant",
        "appeal", "petition", "writ", "habeas corpus", "mandamus",
        "divorce", "maintenance", "custody", "property", "contract",
        "murder", "theft", "assault", "cheating", "fraud",
    ]
    for term in legal_terms:
        if term.lower() in query.lower():
            keywords.append(term)

    return keywords


def get_collection_stats(
    collection_name: str = "advocate_legal_docs",
    persist_dir: str = "./data/chroma_db",
) -> Dict:
    """Get statistics about the vector store."""
    collection = get_collection(collection_name, persist_dir)
    return {
        "total_documents": collection.count(),
        "collection_name": collection_name,
    }
