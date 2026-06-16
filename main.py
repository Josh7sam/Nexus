"""
Nexus — main entry point.

Usage:
    python main.py               # Interactive REPL mode
    python main.py --serve        # Start FastAPI server
    python main.py --ingest       # Ingest documents from data/documents/
    python main.py --ingest --serve  # Ingest then serve
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse
import os
import sys

# Force UTF-8 encoding for standard streams (prevents crashes with unicode characters on Windows console)
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8')
except Exception:
    pass

# Append backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from config import API_HOST, API_PORT


def run_repl():
    """Interactive terminal chat loop for quick testing."""
    from builder import build_graph
    from store import FeedbackStore

    graph = build_graph()
    store = FeedbackStore()

    print()
    print("+------------------------------------------------------+")
    print("|   Nexus -- Interactive Mode                      |")
    print("|   Type 'quit' to exit                                |")
    print("+------------------------------------------------------+")
    print()

    while True:
        try:
            question = input("You ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print()
        try:
            result = graph.invoke({
                "question": question,
                "generation": "",
                "documents": [],
                "dense_docs": [],
                "sparse_docs": [],
                "fused_docs": [],
                "query_rewrites": [],
                "rewrite_count": 0,
                "relevance_score": 0.0,
                "hallucination_retries": 0,
                "interaction_id": "",
                "route_decision": "",
                "is_grounded": True,
            })
        except Exception as e:
            import uuid
            err = str(e)
            result = {
                "question": question,
                "documents": [],
                "dense_docs": [],
                "sparse_docs": [],
                "fused_docs": [],
                "query_rewrites": [],
                "rewrite_count": 0,
                "relevance_score": 0.0,
                "hallucination_retries": 0,
                "interaction_id": str(uuid.uuid4()),
                "route_decision": "",
                "is_grounded": True,
            }
            if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                result["generation"] = "⚠️ Nexus System Status: Gemini API Daily Free-Tier Quota Limit fully exhausted. The system will resume operational status at Midnight Pacific Time."
            else:
                result["generation"] = f"An unexpected processing error occurred: {str(e)}"

        print(f"\n{'─' * 56}")
        print(f"Assistant ❯ {result['generation']}")

        docs = result.get("documents", [])
        if docs:
            print(f"\n  📄 {len(docs)} source(s) used")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source_file", "?")
                score = doc.metadata.get("rrf_score", 0)
                print(f"     {i}. {src}  (RRF={score:.4f})")

        rw = result.get("rewrite_count", 0)
        if rw:
            print(f"  🔄 Query rewrites: {rw}")

        # Persist interaction
        iid = result.get("interaction_id", "")
        store.save_interaction(
            interaction_id=iid,
            question=question,
            generation=result["generation"],
            documents=docs,
            rewrite_count=rw,
            relevance_score=result.get("relevance_score", 0.0),
        )

        # Ask for feedback
        try:
            fb = input("\n  Rate this answer [👍 l / 👎 d / skip]: ").strip().lower()
            if fb in ("l", "like", "👍"):
                store.save_feedback(iid, "like")
                print("  [OK] Thanks! Positive feedback recorded.")
            elif fb in ("d", "dislike", "👎"):
                store.save_feedback(iid, "dislike")
                print("  [OK] Thanks! Negative feedback recorded.")
        except (EOFError, KeyboardInterrupt):
            pass

        print(f"{'─' * 56}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Nexus — LangGraph + ChromaDB + BM25 + RLHF"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start the FastAPI server with the chat UI",
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Run document ingestion (dual-index into ChromaDB + BM25)",
    )
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)

    args = parser.parse_args()

    # ── Ingest ────────────────────────────────────────────────
    if args.ingest:
        from ingest import ingest_documents
        ingest_documents()
        if not args.serve:
            return

    # ── Serve ─────────────────────────────────────────────────
    if args.serve:
        import uvicorn

        print()
        print("+------------------------------------------------------+")
        print(f"|   Starting server on http://{args.host}:{args.port}          |")
        print("|   Open browser to http://localhost:8000               |")
        print("+------------------------------------------------------+")
        print()

        uvicorn.run(
            "backend.server:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    # ── Default: interactive REPL ─────────────────────────────
    run_repl()


if __name__ == "__main__":
    main()
