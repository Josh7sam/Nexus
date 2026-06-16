"""
Document ingestion pipeline — loads, chunks, and dual-indexes documents
into ChromaDB (dense) and BM25 (sparse) at the same time.

Supports: PDF, TXT, MD files from the configured data directory.
"""

import os
import glob

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from dense import get_embeddings
from sparse import BM25Index
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)


# ═══════════════════════════════════════════════════════════════
#  1. Document Loading
# ═══════════════════════════════════════════════════════════════

def load_documents(source_dir: str | None = None) -> list[Document]:
    """
    Recursively load PDF, TXT, and MD files from the source directory.
    Each file becomes one or more Document objects with source metadata.
    """
    source_dir = source_dir or DATA_DIR
    documents: list[Document] = []

    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        print(f"  Created empty data directory: {source_dir}")
        return documents

    # PDF files
    for path in glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True):
        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"  [WARN] Failed to load {path}: {e}")

    # Text files
    for path in glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True):
        try:
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"  ⚠ Failed to load {path}: {e}")

    # Markdown files
    for path in glob.glob(os.path.join(source_dir, "**/*.md"), recursive=True):
        try:
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"  ⚠ Failed to load {path}: {e}")

    return documents


# ═══════════════════════════════════════════════════════════════
#  2. Chunking + Metadata Tagging
# ═══════════════════════════════════════════════════════════════

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Adds a unique ``chunk_id`` and ``source_file`` to each chunk's metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:04d}"
        # Normalise source path for display
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["source_file"] = os.path.basename(source)

    return chunks


# ═══════════════════════════════════════════════════════════════
#  3. Dual-Index Ingestion
# ═══════════════════════════════════════════════════════════════

def dual_index(chunks: list[Document]) -> None:
    """
    Index every chunk into BOTH:
      • ChromaDB (dense — embedding vectors, HNSW cosine)
      • BM25     (sparse — inverted keyword index, pickled)

    This is the core of the hybrid retrieval strategy.
    """
    print(f"  Indexing {len(chunks)} chunks into dual indexes ...")

    import shutil
    # ── Dense index (ChromaDB) ────────────────────────────────
    print("  * Building ChromaDB dense index ...")
    embeddings = get_embeddings()
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"  * Clearing existing ChromaDB directory to prevent dimension mismatches: {CHROMA_PERSIST_DIR}")
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception as e:
            print(f"    [WARN] Failed to clear {CHROMA_PERSIST_DIR}: {e}")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print(f"    [OK] ChromaDB: {len(chunks)} chunks indexed")

    # ── Sparse index (BM25) ──────────────────────────────────
    print("  * Building BM25 sparse index ...")
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save()
    print(f"    [OK] BM25: {len(chunks)} chunks indexed -> pickled")


# ═══════════════════════════════════════════════════════════════
#  4. Main Entry Point
# ═══════════════════════════════════════════════════════════════

def ingest_documents(source_dir: str | None = None) -> int:
    """
    End-to-end ingestion: load → chunk → dual-index.

    Returns:
        Number of chunks created.
    """
    import shutil
    shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)

    print("=" * 60)
    print("  DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    print("\n1 > Loading documents ...")
    documents = load_documents(source_dir)
    if not documents:
        print("  [WARN] No documents found. Add PDF / TXT / MD files to:")
        print(f"    {source_dir or DATA_DIR}")
        return 0
    print(f"  [OK] Loaded {len(documents)} raw document(s)")

    print("\n2 > Chunking (size={}, overlap={}) ...".format(CHUNK_SIZE, CHUNK_OVERLAP))
    chunks = chunk_documents(documents)
    print(f"  [OK] Created {len(chunks)} chunks")

    print("\n3 > Dual-index ingestion ...")
    dual_index(chunks)

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    return len(chunks)


# Allow running as standalone script: python -m vectorstore.ingest
if __name__ == "__main__":
    ingest_documents()
