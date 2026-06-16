import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from graph.builder import build_graph

graph = build_graph()
print("Invoking graph for 'Who is Joshua'...")
result = graph.invoke({
    "question": "Who is Joshua",
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
print("\n--- GRAPH INVOCATION COMPLETE ---")
print("Route Decision:", result.get("route_decision"))
print("Rewrite Count:", result.get("rewrite_count"))
print("Query Rewrites:", result.get("query_rewrites"))
gen = result.get("generation", "")
print("Generation:", gen.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
print("Documents count:", len(result.get("documents", [])))
for i, doc in enumerate(result.get("documents", [])):
    src = doc.metadata.get('source_file') or doc.metadata.get('source')
    src_str = str(src).encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
    print(f"  Doc {i}: page_content_len={len(doc.page_content)}, relevance={doc.metadata.get('relevance')}, source={src_str}")
