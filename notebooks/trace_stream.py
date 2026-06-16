import sys
import os
import asyncio
import json

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
    
    print("\n--- STARTING ASTREAM_EVENTS ---")
    async for event in graph.astream_events(_default_state(question), version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        
        # Only print details for on_chain_end and on_chat_model_stream
        if kind == "on_chain_end":
            output = event.get("data", {}).get("output", {})
            has_gen = False
            if isinstance(output, dict) and "generation" in output:
                has_gen = True
            print(f"EVENT: {kind} | NAME: {name} | is_dict={isinstance(output, dict)} | has_generation={has_gen}")
            if has_gen:
                # print keys of output
                print(f"  -> keys: {list(output.keys())}")
                if "query_rewrites" in output:
                    print(f"  -> query_rewrites: {output['query_rewrites']} (type: {type(output['query_rewrites'])})")
        elif kind == "on_chat_model_stream":
            print(f"EVENT: {kind} | NAME: {name}")

if __name__ == "__main__":
    asyncio.run(trace())
