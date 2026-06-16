"""
Sparse retriever — BM25 keyword search via rank-bm25.

Builds and persists a BM25Okapi index alongside the full document corpus
using pickle.  At query time, tokenises the query and returns top-k
documents ranked by BM25 score.
"""

import os
import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import config


class BM25Index:
    """Encapsulates BM25Okapi build / persist / load / search lifecycle."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []

    # ── Build ─────────────────────────────────────────────────

    def build(self, documents: list[Document]) -> None:
        """Tokenise the corpus and build the BM25 index in memory."""
        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        """Pickle the BM25 index and the raw corpus to disk."""
        os.makedirs(os.path.dirname(config.BM25_INDEX_PATH), exist_ok=True)
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(config.BM25_CORPUS_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self) -> None:
        """Restore the BM25 index and corpus from pickle files."""
        with open(config.BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(config.BM25_CORPUS_PATH, "rb") as f:
            self.documents = pickle.load(f)

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        Score the corpus against a query and return the top-k matches.

        Only documents with a positive BM25 score are returned.
        Each result is annotated with its BM25 score in metadata.
        """
        if self.bm25 is None:
            if os.path.exists(config.BM25_INDEX_PATH) and os.path.exists(config.BM25_CORPUS_PATH):
                self.load()
            else:
                print("    [INFO] BM25 sparse index not found — skipping sparse search")
                return []

        k = top_k or config.SPARSE_TOP_K
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Rank indices by descending score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[Document] = []
        for rank, idx in enumerate(ranked[:k]):
            if scores[idx] == 0:
                continue
            doc = Document(
                page_content=self.documents[idx].page_content,
                metadata={
                    **self.documents[idx].metadata,
                    "bm25_score": float(scores[idx]),
                    "retrieval_source": "sparse",
                    "sparse_rank": rank,
                },
            )
            results.append(doc)

        return results

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokeniser."""
        return text.lower().split()


# ── Module-level singleton ────────────────────────────────────
_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Return (and lazily load) the singleton BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        if os.path.exists(config.BM25_INDEX_PATH):
            _bm25_index.load()
    return _bm25_index


def sparse_search(query: str, top_k: int | None = None) -> list[Document]:
    """Convenience wrapper: search the BM25 index for a query."""
    return get_bm25_index().search(query, top_k)
