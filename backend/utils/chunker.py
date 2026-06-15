"""
chunker.py — Legal-aware document chunker
Splits legal documents by sections (e.g., "Section 302 IPC") 
instead of naive fixed-size chunking.
"""

import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class LegalChunk:
    content: str
    metadata: Dict


# Patterns that indicate a new legal section starts
LEGAL_SECTION_PATTERNS = [
    r"Section\s+\d+[A-Z]?\s*[\.\-\—]",         # Section 420 IPC
    r"Article\s+\d+[A-Z]?\s*[\.\-\—]",          # Article 21 Constitution
    r"SECTION\s+\d+[A-Z]?\s*[\.\-\—]",
    r"Chapter\s+[IVXLCDM]+\s*[\.\-\—]",         # Chapter VII
    r"CHAPTER\s+[IVXLCDM]+",
    r"^\d+\.\s+[A-Z]",                           # 1. Short title
    r"Clause\s+\d+\s*[\.\-]",
    r"Schedule\s+[IVXLCDM]+",
]

COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in LEGAL_SECTION_PATTERNS]


def is_section_boundary(line: str) -> bool:
    """Check if a line represents the start of a new legal section."""
    for pattern in COMPILED_PATTERNS:
        if pattern.search(line):
            return True
    return False


def chunk_legal_document(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[LegalChunk]:
    """
    Chunk a legal document using section-aware splitting.
    Falls back to size-based chunking if no sections found.
    """
    chunks = []
    lines = text.split("\n")

    current_section = []
    current_section_title = "General"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_section_boundary(line):
            # Save previous section
            if current_section:
                section_text = " ".join(current_section)
                sub_chunks = _split_by_size(
                    section_text, chunk_size, chunk_overlap
                )
                for i, chunk in enumerate(sub_chunks):
                    chunks.append(
                        LegalChunk(
                            content=chunk,
                            metadata={
                                "source": source,
                                "section": current_section_title,
                                "chunk_index": i,
                            },
                        )
                    )
            current_section_title = line[:100]  # Use section header as title
            current_section = [line]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_text = " ".join(current_section)
        sub_chunks = _split_by_size(section_text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(sub_chunks):
            chunks.append(
                LegalChunk(
                    content=chunk,
                    metadata={
                        "source": source,
                        "section": current_section_title,
                        "chunk_index": i,
                    },
                )
            )

    # Fallback: if chunking produced nothing, do plain size-based chunking
    if not chunks:
        plain_chunks = _split_by_size(text, chunk_size, chunk_overlap)
        chunks = [
            LegalChunk(
                content=c,
                metadata={"source": source, "section": "Unknown", "chunk_index": i},
            )
            for i, c in enumerate(plain_chunks)
        ]

    return chunks


def _split_by_size(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to end at a sentence boundary
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            end = start + last_period + 1
            chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]
