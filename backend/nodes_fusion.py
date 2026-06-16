"""
RRF Fusion node — merges dense and sparse results using Reciprocal Rank Fusion,
with RLHF-adapted weights and document boosting.
"""

from state import GraphState
from fusion import reciprocal_rank_fusion
from config import DENSE_WEIGHT, SPARSE_WEIGHT


def fusion_rrf(state: GraphState) -> dict:
    """
    Merge dense + sparse ranked lists via weighted RRF.

    Retrieval weights (alpha, beta) are dynamically adjusted by the
    RLHF manager based on accumulated user feedback. Documents that
    historically received positive feedback get a boost score.
    """
    dense_docs = state.get("dense_docs", [])
    sparse_docs = state.get("sparse_docs", [])

    # Attempt to use RLHF-adapted weights; fall back to config defaults
    dense_weight, sparse_weight = DENSE_WEIGHT, SPARSE_WEIGHT
    try:
        from rlhf import get_rlhf_manager
        rlhf = get_rlhf_manager()
        dense_weight, sparse_weight = rlhf.get_current_weights()
    except Exception:
        pass  # First run or no feedback yet

    # RRF fusion parameters
    k_const, top_k = None, None
    try:
        from store import FeedbackStore
        settings = FeedbackStore().get_settings()
        if "rrf_k_constant" in settings:
            k_const = int(settings["rrf_k_constant"])
        if "fusion_top_k" in settings:
            top_k = int(settings["fusion_top_k"])
    except Exception:
        pass

    print(f"  -> [fusion_rrf] Fusing {len(dense_docs)} dense + {len(sparse_docs)} sparse "
          f"(a={dense_weight:.2f}, b={sparse_weight:.2f}, k={k_const or 'default'}, top_k={top_k or 'default'})")

    # RRF fusion
    fused = reciprocal_rank_fusion(
        dense_docs,
        sparse_docs,
        alpha=dense_weight,
        beta=sparse_weight,
        k=k_const,
        final_top_k=top_k,
    )

    # Apply RLHF document boosting
    try:
        fused = rlhf.boost_documents(fused)
    except Exception:
        pass

    print(f"    * Fused results: {len(fused)} documents")

    return {
        "fused_docs": fused,
        "documents": fused,  # Set as primary docs for downstream nodes
    }
