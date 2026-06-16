"""
Reciprocal Rank Fusion (RRF) — merges ranked lists from dense and
sparse retrievers into a single unified ranking.

Formula:  Score(d) = α · 1/(k + rank_dense(d)) + β · 1/(k + rank_sparse(d))

Where:
  k = smoothing constant (default 60)
  α = dense retrieval weight  (default 0.6)
  β = sparse retrieval weight (default 0.4)
"""

from langchain_core.documents import Document
from config import RRF_K_CONSTANT, DENSE_WEIGHT, SPARSE_WEIGHT, FUSION_TOP_K


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    *,
    k: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    final_top_k: int | None = None,
) -> list[Document]:
    """
    Merge two ranked document lists using weighted RRF.

    Args:
        dense_docs:   Documents ranked by vector similarity (best first).
        sparse_docs:  Documents ranked by BM25 score (best first).
        k:            RRF smoothing constant (default: 60).
        alpha:        Weight for dense retrieval scores.
        beta:         Weight for sparse retrieval scores.
        final_top_k:  Number of documents to return after fusion.

    Returns:
        Fused list of Documents sorted by RRF score, annotated with
        ``rrf_score`` and ``rrf_sources`` in metadata.
    """
    k = k if k is not None else RRF_K_CONSTANT
    alpha = alpha if alpha is not None else DENSE_WEIGHT
    beta = beta if beta is not None else SPARSE_WEIGHT
    final_top_k = final_top_k if final_top_k is not None else FUSION_TOP_K

    # Use page_content hash as document identity for dedup
    scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}
    source_tracker: dict[int, list[str]] = {}

    # ── Score dense results ───────────────────────────────────
    for rank, doc in enumerate(dense_docs):
        doc_id = hash(doc.page_content)
        rrf_score = alpha * (1.0 / (k + rank + 1))
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
        source_tracker.setdefault(doc_id, []).append("dense")
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # ── Score sparse results ──────────────────────────────────
    for rank, doc in enumerate(sparse_docs):
        doc_id = hash(doc.page_content)
        rrf_score = beta * (1.0 / (k + rank + 1))
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
        source_tracker.setdefault(doc_id, []).append("sparse")
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # ── Sort by fused score ───────────────────────────────────
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    # ── Build result list with metadata ───────────────────────
    results: list[Document] = []
    for doc_id in sorted_ids[:final_top_k]:
        doc = doc_map[doc_id]
        doc.metadata["rrf_score"] = round(scores[doc_id], 6)
        doc.metadata["rrf_sources"] = source_tracker.get(doc_id, [])
        results.append(doc)

    return results
