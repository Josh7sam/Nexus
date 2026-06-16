"""
Hybrid retrieval node — calls both dense (ChromaDB) and sparse (BM25)
retrievers and stores their raw results in the graph state.
"""

import concurrent.futures
from state import GraphState
from dense import dense_search
from sparse import sparse_search


def hybrid_retrieve(state: GraphState) -> dict:
    """
    Execute dense + sparse retrieval in a single node.

    Internally calls ChromaDB similarity search and BM25 keyword search,
    storing raw ranked lists in state for downstream RRF fusion.
    """
    question = state["question"]
    print(f"  -> [hybrid_retrieve] Searching for: {question[:80]}...")

    # Load dynamic retrieval parameters from settings
    dense_k, sparse_k = None, None
    try:
        from store import FeedbackStore
        settings = FeedbackStore().get_settings()
        if "dense_top_k" in settings:
            dense_k = int(settings["dense_top_k"])
        if "sparse_top_k" in settings:
            sparse_k = int(settings["sparse_top_k"])
    except Exception:
        pass

    # Execute dense and sparse retrieval in parallel threads
    dense_docs = []
    sparse_docs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(dense_search, question, top_k=dense_k)
        sparse_future = executor.submit(sparse_search, question, top_k=sparse_k)

        try:
            dense_docs = dense_future.result()
            print(f"    * Dense (ChromaDB): {len(dense_docs)} results")
        except Exception as e:
            print(f"    [WARN] Dense retrieval failed: {e}")

        try:
            sparse_docs = sparse_future.result()
            print(f"    * Sparse (BM25):    {len(sparse_docs)} results")
        except Exception as e:
            print(f"    [WARN] Sparse retrieval failed: {e}")

    return {
        "dense_docs": dense_docs,
        "sparse_docs": sparse_docs,
    }

