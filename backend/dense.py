"""
Dense retriever — ChromaDB vector similarity search via LangChain.

Uses GoogleGenerativeAIEmbeddings (text-embedding-004) for embedding and ChromaDB
with HNSW cosine similarity for nearest-neighbour lookup.
"""

import os
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    GOOGLE_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    DENSE_TOP_K,
)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the shared embedding model instance."""
    api_key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")
        
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=api_key,
    )


# ── Query Embedding LRU Cache ────────────────────────────────
# Caches up to 256 unique query embeddings in memory.
# Repeated identical queries skip the ~800ms Gemini API round-trip.
# Keys are normalized (strip + lowercase) to maximize hit rates.
_embedding_instance = None


def _get_embedding_instance():
    """Lazy singleton for the embedding model used by the cache."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = get_embeddings()
    return _embedding_instance


@lru_cache(maxsize=256)
def cached_embed_query(query: str) -> tuple:
    """
    Generate and cache the embedding vector for a query string.

    The query is normalized (strip + lowercase) before cache lookup
    to maximize hit rates for near-identical queries.

    Returns a tuple (hashable for lru_cache). Convert back to list
    before passing to ChromaDB.
    """
    embed_model = _get_embedding_instance()
    vector = embed_model.embed_query(query)
    print(f"    [CACHE MISS] Embedding generated for: {query[:50]}...")
    return tuple(vector)


# Unified vector store singleton instance
_vectorstore = None


def get_vectorstore() -> Chroma:
    """
    Connect to the persisted ChromaDB collection (lazy singleton).
    Returns a Chroma instance ready for search or ingestion.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def reset_vectorstore() -> None:
    """Clear the cached vectorstore reference (e.g. after ingestion)."""
    global _vectorstore
    _vectorstore = None


def get_dense_retriever():
    """Return a LangChain retriever wrapping the ChromaDB collection."""
    return get_vectorstore().as_retriever(
        search_kwargs={"k": DENSE_TOP_K},
    )


def dense_search(query: str, top_k: int | None = None) -> list[Document]:
    """
    Run dense similarity search against ChromaDB.

    Uses the LRU-cached query embedding to avoid redundant Gemini API
    calls for repeated or identical queries (~800ms saved per cache hit).

    Args:
        query:  Natural-language query string.
        top_k:  Override the default number of results.

    Returns:
        Ranked list of Document objects with metadata.
        Empty list if ChromaDB is empty or API unavailable.
    """
    try:
        store = get_vectorstore()
        # Check if collection has any documents before searching
        collection = store._collection
        count = collection.count()
        if count == 0:
            print("    [INFO] ChromaDB collection is empty — skipping dense search")
            return []

        # Normalize query for cache key (strip whitespace + lowercase)
        normalized_query = query.strip().lower()
        embedding_vector = list(cached_embed_query(normalized_query))
        results = store.similarity_search_by_vector(
            embedding_vector, k=top_k or DENSE_TOP_K
        )

        for i, doc in enumerate(results):
            doc.metadata["retrieval_source"] = "dense"
            doc.metadata["dense_rank"] = i

        return results

    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "api key" in err.lower() or "403" in err or "unauthorized" in err.lower() or "credentials" in err.lower():
            print(f"    [WARN] Dense retrieval failed (Gemini API Authentication error): {e}")
        elif "connection" in err.lower() or "connect" in err.lower() or "timeout" in err.lower() or "http" in err.lower():
            print(f"    [WARN] Dense retrieval failed (Network connection issue): {e}")
        else:
            print(f"    [WARN] Dense retrieval failed: {e}")
        return []
