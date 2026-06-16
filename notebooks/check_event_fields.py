import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.builder import build_graph

def _default_state(question: str) -> dict:
    return {
        "question": question,
        "rewritten_query": "",
        "generation": "",
        "documents": [],
        "dense_docs": [],
        "sparse_docs": [],
        "fused_docs": [],
        "web_documents": [],
        "query_rewrites": [],
        "rewrite_count": 0,
        "retrieval_retries": 0,
        "relevance_score": 0.0,
        "hallucination_retries": 0,
        "interaction_id": "",
        "route_decision": "",
        "is_grounded": True,
    }

async def trace():
    graph = build_graph()
    question = "JP Morgan Chase & CO, what do they do?"
    
    print("\n--- STARTING EVENT FIELDS CHECK ---")
    async for event in graph.astream_events(_default_state(question), version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        
        if kind == "on_chat_model_stream":
            metadata = event.get("metadata", {})
            tags = event.get("tags", [])
            print(f"EVENT: {kind} | NAME: {name}")
            print(f"  -> tags: {tags}")
            print(f"  -> metadata: {metadata}")
            print(f"  -> langgraph_node: {metadata.get('langgraph_node')}")

if __name__ == "__main__":
    asyncio.run(trace())
