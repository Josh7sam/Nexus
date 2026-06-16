import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.builder import build_graph

def run():
    print("Building graph...")
    graph = build_graph()
    
    question = "JP Morgan Chase & CO, what do they do?"
    state = {
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
    
    print(f"Invoking graph with question: '{question}'")
    try:
        res = graph.invoke(state)
        print("Success!")
        print("Generation:", res.get("generation"))
    except Exception as e:
        import traceback
        print("Exception caught:")
        traceback.print_exc()

if __name__ == "__main__":
    run()
