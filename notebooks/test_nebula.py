import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from graph.builder import build_graph

def main():
    graph = build_graph()
    question = "Who is Joshua"
    print(f"Invoking graph with question: '{question}'")
    
    result = graph.invoke({
        "question": question,
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
    })
    
    print("\n--- RESULT ---")
    print("Generation:", result.get("generation"))
    print("Documents count:", len(result.get("documents", [])))
    print("Web documents count:", len(result.get("web_documents", [])))
    print("Route decision:", result.get("route_decision"))
    print("Is grounded:", result.get("is_grounded"))
    print("Rewrite count:", result.get("rewrite_count"))
    print("Query rewrites:", result.get("query_rewrites"))

if __name__ == "__main__":
    main()
