import sys
import os
import time
import asyncio

# Add backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Decorator to measure execution time
def time_node(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"\n>>> Starting Node: {func.__name__} ...")
        res = func(*args, **kwargs)
        end = time.time()
        print(f"<<< Finished Node: {func.__name__} in {end - start:.4f} seconds")
        return res
    return wrapper

# Instrument node functions before importing builder
import graph.nodes_agent as nodes_agent
import graph.nodes_retrieval as nodes_retrieval
import graph.nodes_fusion as nodes_fusion

nodes_agent.router_node = time_node(nodes_agent.router_node)
nodes_agent.grade_documents_node = time_node(nodes_agent.grade_documents_node)
nodes_agent.rewrite_query_node = time_node(nodes_agent.rewrite_query_node)
nodes_agent.web_scrape_node = time_node(nodes_agent.web_scrape_node)
nodes_agent.generate_node = time_node(nodes_agent.generate_node)
nodes_agent.generate_direct_node = time_node(nodes_agent.generate_direct_node)
nodes_agent.hallucination_check_node = time_node(nodes_agent.hallucination_check_node)

nodes_retrieval.hybrid_retrieve = time_node(nodes_retrieval.hybrid_retrieve)
nodes_fusion.fusion_rrf = time_node(nodes_fusion.fusion_rrf)

from graph.builder import build_graph

def run_profile(question: str):
    graph = build_graph()
    state = {
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
    }
    
    start_total = time.time()
    print(f"\n==================================================")
    print(f"PROFILING QUERY: '{question}'")
    print(f"==================================================")
    result = graph.invoke(state)
    end_total = time.time()
    print(f"==================================================")
    print(f"TOTAL TIME: {end_total - start_total:.4f} seconds")
    print(f"Result Generation Length: {len(result['generation'])} chars")
    print(f"Route Decision: {result.get('route_decision')}")
    print(f"Rewrite Count: {result.get('rewrite_count')}")
    print(f"Relevance Score: {result.get('relevance_score')}")
    print(f"Is Grounded: {result.get('is_grounded')}")
    print(f"==================================================\n")

if __name__ == "__main__":
    # Test case 1: Greetings fast-path
    run_profile("hi")
    
    # Test case 2: Relevant question
    run_profile("What is Retrieval-Augmented Generation (RAG)?")
