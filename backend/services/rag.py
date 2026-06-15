"""
rag.py — Core RAG pipeline for Advocate AI
Retrieves relevant legal documents and constructs prompts for the LLM.
"""

import logging
from typing import List, Dict, Tuple

from services.vector_store import hybrid_search
from services.llm import generate_response
from utils.guardrails import build_system_prompt, add_disclaimer, check_query
from utils.session import get_history, add_turn, format_history_for_ollama

logger = logging.getLogger(__name__)


def build_rag_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    """
    Construct the final prompt with retrieved legal context injected.
    """
    if not retrieved_docs:
        context = "No specific legal documents were found for this query. Answer based on general legal knowledge."
    else:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            section = doc["metadata"].get("section", "")
            score = doc.get("score", 0)

            header = f"[Document {i} | Source: {source}"
            if section:
                header += f" | Section: {section}"
            header += f" | Relevance: {score:.2f}]"

            context_parts.append(f"{header}\n{doc['content']}")

        context = "\n\n".join(context_parts)

    prompt = f"""Based on the following legal documents, answer the user's question.

=== RETRIEVED LEGAL CONTEXT ===
{context}

=== USER QUESTION ===
{query}

=== INSTRUCTIONS ===
- Answer using the retrieved context above as your primary source
- Cite specific sections/articles when mentioning laws
- If the context doesn't fully answer the question, supplement with your legal knowledge
- Keep the answer clear and easy to understand
- Structure your answer with: 1) Direct answer, 2) Legal basis, 3) Practical implications

Answer:"""

    return prompt


async def process_legal_query(
    query: str,
    session_id: str = None,
    model: str = "llama3",
    ollama_url: str = "http://localhost:11434",
    top_k: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    collection_name: str = "advocate_legal_docs",
    persist_dir: str = "./data/chroma_db",
) -> Dict:
    """
    Full RAG pipeline:
    1. Safety check
    2. Retrieve relevant docs
    3. Build prompt
    4. Generate response
    5. Add disclaimer
    6. Save to session
    """

    # Step 1: Safety check
    is_safe, block_reason = check_query(query)
    if not is_safe:
        return {
            "answer": block_reason,
            "sources": [],
            "model_used": model,
            "blocked": True,
        }

    # Step 2: Retrieve relevant legal documents
    logger.info(f"Searching for: {query}")
    retrieved_docs = hybrid_search(
        query=query,
        top_k=top_k,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )
    logger.info(f"Retrieved {len(retrieved_docs)} documents")

    # Step 3: Build RAG prompt
    rag_prompt = build_rag_prompt(query, retrieved_docs)

    # Step 4: Get conversation history
    history = []
    if session_id:
        raw_history = get_history(session_id)
        history = format_history_for_ollama(raw_history)

    # Step 5: Generate LLM response
    system_prompt = build_system_prompt()
    logger.info(f"Generating response with model: {model}")

    raw_answer = await generate_response(
        prompt=rag_prompt,
        history=history,
        model=model,
        system_prompt=system_prompt,
        ollama_url=ollama_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Step 6: Add disclaimer
    final_answer = add_disclaimer(raw_answer)

    # Step 7: Save to session memory
    if session_id:
        add_turn(session_id, query, final_answer)

    # Build sources for citation display
    sources = [
        {
            "source": doc["metadata"].get("source", "Unknown"),
            "section": doc["metadata"].get("section", ""),
            "score": round(doc.get("score", 0), 3),
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": final_answer,
        "sources": sources,
        "model_used": model,
        "documents_retrieved": len(retrieved_docs),
        "blocked": False,
    }
