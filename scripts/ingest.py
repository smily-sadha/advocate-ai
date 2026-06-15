"""
ingest.py — Legal Document Ingestion Script
Run ONCE to process and store legal documents in ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --data-dir ./data/raw --reset

Supported formats: PDF, HTML, TXT
"""

import sys
import argparse
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import settings
from services.vector_store import add_documents, get_collection_stats
from utils.chunker import chunk_legal_document, LegalChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        return ""


def extract_text_from_html(html_path: str) -> str:
    """Extract text from an HTML file."""
    try:
        from bs4 import BeautifulSoup
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.error(f"HTML extraction failed for {html_path}: {e}")
        return ""


def extract_text_from_txt(txt_path: str) -> str:
    """Read plain text file."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Text extraction failed for {txt_path}: {e}")
        return ""


def process_file(file_path: Path) -> list:
    """Process a single file and return chunks."""
    ext = file_path.suffix.lower()
    source = file_path.name

    logger.info(f"Processing: {source}")

    if ext == ".pdf":
        text = extract_text_from_pdf(str(file_path))
    elif ext in [".html", ".htm"]:
        text = extract_text_from_html(str(file_path))
    elif ext in [".txt", ".md"]:
        text = extract_text_from_txt(str(file_path))
    else:
        logger.warning(f"Unsupported file type: {ext}. Skipping {source}")
        return []

    if not text or len(text.strip()) < 100:
        logger.warning(f"No usable text extracted from {source}")
        return []

    # Use legal-aware chunking
    chunks = chunk_legal_document(
        text=text,
        source=source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    logger.info(f"  → {len(chunks)} chunks generated from {source}")
    return [{"content": c.content, "metadata": c.metadata} for c in chunks]


def ingest_directory(data_dir: str, reset: bool = False):
    """Ingest all supported files from a directory."""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Collect all supported files
    supported_extensions = [".pdf", ".html", ".htm", ".txt", ".md"]
    files = []
    for ext in supported_extensions:
        files.extend(data_path.glob(f"*{ext}"))
        files.extend(data_path.glob(f"**/*{ext}"))  # Recursive

    files = list(set(files))  # Deduplicate

    if not files:
        logger.warning(
            f"No supported files found in {data_dir}.\n"
            f"  Add PDF, HTML, or TXT files to {data_dir} and run again."
        )
        # Add sample data for testing
        _create_sample_data(data_dir)
        return

    logger.info(f"Found {len(files)} files to process.")

    if reset:
        logger.info("Reset flag set — clearing existing vector store...")
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        try:
            client.delete_collection(settings.chroma_collection_name)
            logger.info("Existing collection deleted.")
        except Exception:
            pass

    all_chunks = []
    for file_path in files:
        chunks = process_file(file_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks generated. Check your files.")
        return

    logger.info(f"\nTotal chunks to ingest: {len(all_chunks)}")
    logger.info("Generating embeddings and storing in ChromaDB...")

    add_documents(
        chunks=all_chunks,
        collection_name=settings.chroma_collection_name,
        persist_dir=settings.chroma_persist_dir,
    )

    stats = get_collection_stats(
        collection_name=settings.chroma_collection_name,
        persist_dir=settings.chroma_persist_dir,
    )

    logger.info("\n" + "=" * 40)
    logger.info("  INGESTION COMPLETE")
    logger.info(f"  Total documents in DB: {stats['total_documents']}")
    logger.info("=" * 40)


def _create_sample_data(data_dir: str):
    """Create sample legal text for testing when no files are provided."""
    sample_text = """
Section 1: Short title, extent and commencement.
This Act may be called the Bharatiya Nyaya Sanhita, 2023. It extends to the whole of India.

Section 101: Right of private defence of body and of property.
Every person has a right, subject to the restrictions contained in section 102, to defend —
(a) his own body, and the body of any other person, against any offence affecting the human body;
(b) the property, whether movable or immovable, of himself or of any other person, against theft, robbery, mischief or criminal trespass.

Section 103: Punishment for murder.
Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.

Article 21 of the Constitution of India:
Protection of life and personal liberty — No person shall be deprived of his life or personal liberty except according to procedure established by law.

Article 32 of the Constitution of India:
Remedies for enforcement of rights conferred by this Part —
(1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.

Section 138 of Negotiable Instruments Act:
Dishonour of cheque for insufficiency, etc., of funds in the account —
Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge, in whole or in part, of any debt or other liability, is returned by the bank unpaid, either because of the amount of money standing to the credit of that account is insufficient to honour the cheque or that it exceeds the amount arranged to be paid from that account by an agreement made with that bank, such person shall be deemed to have committed an offence.

Bail under BNSS (Bharatiya Nagarik Suraksha Sanhita):
Section 478 — Bail in bailable offences: When any person other than a person accused of a non-bailable offence is arrested or detained without warrant by an officer in charge of a police station, or appears or is brought before a Court, and is prepared at any time while in the custody of such officer or at any stage of the proceeding before such Court to give bail, such person shall be released on bail.
"""

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    sample_path = Path(data_dir) / "sample_indian_laws.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    logger.info(f"Sample legal data created at: {sample_path}")
    logger.info("Re-running ingestion with sample data...")
    ingest_directory(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal documents into ChromaDB")
    parser.add_argument(
        "--data-dir",
        default="./data/raw",
        help="Directory containing legal documents (default: ./data/raw)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing vector store before ingesting",
    )
    args = parser.parse_args()

    logger.info("=" * 40)
    logger.info("  Advocate AI — Document Ingestion")
    logger.info("=" * 40)

    ingest_directory(data_dir=args.data_dir, reset=args.reset)
