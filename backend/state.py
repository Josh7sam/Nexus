"""
GraphState — shared state definition for the Hybrid RAG LangGraph agent.

Every node reads from and writes updates to this state dictionary.
Fields are replaced (not accumulated) on each node return.
"""

from typing import TypedDict, Optional
from langchain_core.documents import Document


class GraphState(TypedDict):
    """Typed state flowing through the Corrective RAG graph."""

    # ── User Input ───────────────────────────────────────────
    question: str                    # Current question (may be rewritten)
    rewritten_query: str             # Rewritten query used for web search / fallback retrieval

    # ── Generation ───────────────────────────────────────────
    generation: str                  # LLM-generated answer

    # ── Retrieval Results ────────────────────────────────────
    documents: list[Document]        # Final documents used for generation
    dense_docs: list[Document]       # Raw results from ChromaDB dense search
    sparse_docs: list[Document]      # Raw results from BM25 sparse search
    fused_docs: list[Document]       # RRF-fused results before grading
    web_documents: list[Document]    # Documents retrieved from web crawling fallback

    # ── Query Rewriting ──────────────────────────────────────
    query_rewrites: list[str]        # History of rewritten queries
    rewrite_count: int               # Number of rewrites performed
    retrieval_retries: int           # Count of retrieval retries

    # ── Evaluation ───────────────────────────────────────────
    relevance_score: float           # Fraction of docs graded as relevant
    hallucination_retries: int       # Number of hallucination re-generation attempts
    is_grounded: bool                # Whether the current generation passed grounding check

    # ── Routing ──────────────────────────────────────────────
    route_decision: str              # "retrieve" or "direct"

    # ── RLHF Tracking ────────────────────────────────────────
    interaction_id: str              # Unique ID for this interaction (for feedback)
