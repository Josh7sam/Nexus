"""
Conditional edge routing functions for the Corrective RAG graph.

Each function inspects the current GraphState and returns the name of
the next node.  Used by StateGraph.add_conditional_edges().
"""

from langgraph.graph import END
from state import GraphState
from config import MAX_REWRITE_ATTEMPTS, MAX_HALLUCINATION_RETRIES


def route_after_router(state: GraphState) -> str:
    """Router → hybrid_retrieve | generate_direct"""
    route = state.get("route_decision", "retrieve")
    return "generate_direct" if route == "direct" else "hybrid_retrieve"


def route_after_grading(state: GraphState) -> str:
    """
    grade_documents → generate | rewrite_query

    If at least one relevant doc exists, proceed to generation.
    Otherwise, rewrite the query — unless we've exhausted retries.
    """
    documents = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    if documents:
        return "generate"

    max_rewrites = MAX_REWRITE_ATTEMPTS
    try:
        from store import FeedbackStore
        settings = FeedbackStore().get_settings()
        if "max_rewrite_attempts" in settings:
            max_rewrites = int(settings["max_rewrite_attempts"])
    except Exception:
        pass

    if rewrite_count >= max_rewrites:
        # Exhausted rewrites — generate with whatever we have
        return "generate"

    return "rewrite_query"


def route_after_hallucination(state: GraphState) -> str:
    """
    hallucination_check → END | generate

    If the answer is grounded, finish.  Otherwise, re-generate up to
    MAX_HALLUCINATION_RETRIES times before giving up.
    """
    is_grounded = state.get("is_grounded", True)
    hallucination_retries = state.get("hallucination_retries", 0)
    documents = state.get("documents", [])

    # Direct answers or grounded answers → done
    if is_grounded or not documents:
        return END

    max_hallucination_retries = MAX_HALLUCINATION_RETRIES
    try:
        from store import FeedbackStore
        settings = FeedbackStore().get_settings()
        if "max_hallucination_retries" in settings:
            max_hallucination_retries = int(settings["max_hallucination_retries"])
    except Exception:
        pass

    # Still have retries left → re-generate
    if hallucination_retries < max_hallucination_retries:
        return "generate"

    # Exhausted retries → end anyway
    return END


def route_after_generate(state: GraphState) -> str:
    """
    generate → hallucination_check | END

    If max_hallucination_retries is 0, skip grounding checks entirely to save latency.
    """
    max_hallucination_retries = MAX_HALLUCINATION_RETRIES
    try:
        from store import FeedbackStore
        settings = FeedbackStore().get_settings()
        if "max_hallucination_retries" in settings:
            max_hallucination_retries = int(settings["max_hallucination_retries"])
    except Exception:
        pass

    if max_hallucination_retries <= 0:
        return END
    return "hallucination_check"

