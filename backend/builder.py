"""
Graph builder — assembles the LangGraph StateGraph, wires all nodes
and conditional edges, and compiles the executable graph.

Architecture follows the CRAG (Corrective RAG) pattern:

    START → router ─┬─[retrieve]──→ hybrid_retrieve → fusion_rrf
                     │                                      ↓
                     │                              grade_documents
                     │                         ┌──────┤          ├──────┐
                     │                    [relevant]             [irrelevant]
                     │                         ↓                       ↓
                     │                     generate              rewrite_query
                     │                         ↓                   ↓ (loop)
                     │                hallucination_check    → hybrid_retrieve
                     │                    ┌────┤
                     │              [grounded] [not grounded]
                     │                    ↓         ↓ (retry)
                     └─[direct]──→ generate_direct → END
"""

from langgraph.graph import StateGraph, END

from state import GraphState
from nodes_agent import (
    router_node,
    grade_documents_node,
    rewrite_query_node,
    web_scrape_node,
    generate_node,
    generate_direct_node,
    hallucination_check_node,
)
from nodes_retrieval import hybrid_retrieve
from nodes_fusion import fusion_rrf
from edges import (
    route_after_router,
    route_after_grading,
    route_after_hallucination,
    route_after_generate,
)



def build_graph():
    """Assemble and compile the Corrective RAG graph."""

    workflow = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────
    workflow.add_node("router", router_node)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve)
    workflow.add_node("fusion_rrf", fusion_rrf)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_scrape", web_scrape_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("generate_direct", generate_direct_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # ── Entry point ───────────────────────────────────────────
    workflow.set_entry_point("router")

    # ── Conditional: router → retrieve or direct ──────────────
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "hybrid_retrieve": "hybrid_retrieve",
            "generate_direct": "generate_direct",
        },
    )

    # ── Deterministic: retrieve → fuse → grade ────────────────
    workflow.add_edge("hybrid_retrieve", "fusion_rrf")
    workflow.add_edge("fusion_rrf", "grade_documents")

    # ── Conditional: grade → generate or rewrite ──────────────
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # ── Rewrite loops to web_scrape fallback ──────────────────
    workflow.add_edge("rewrite_query", "web_scrape")

    # ── Web scrape goes to generate ───────────────────────────
    workflow.add_edge("web_scrape", "generate")

    # ── Generate → hallucination check or END ─────────────────
    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "hallucination_check": "hallucination_check",
            END: END,
        },
    )

    # ── Conditional: hallucination → END or re-generate ───────
    workflow.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {
            END: END,
            "generate": "generate",
        },
    )

    # ── Direct answer path → END ──────────────────────────────
    workflow.add_edge("generate_direct", END)

    # ── Compile ───────────────────────────────────────────────
    compiled = workflow.compile()
    print("[OK] LangGraph compiled successfully")
    return compiled
