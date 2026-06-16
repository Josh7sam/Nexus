import os
import sys
import json
import math
import tempfile
import uuid
import pytest
from langchain_core.documents import Document


class TestRRFFusion:
    """Test the Reciprocal Rank Fusion scoring and merging logic."""

    def _make_doc(self, content: str, **meta):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata=meta)

    def test_empty_inputs(self):
        from fusion import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_dense_only(self):
        from fusion import reciprocal_rank_fusion
        docs = [self._make_doc(f"Dense doc {i}") for i in range(3)]
        result = reciprocal_rank_fusion(docs, [], alpha=0.6, beta=0.4, final_top_k=3)
        assert len(result) == 3
        # All should have rrf_score > 0
        for doc in result:
            assert doc.metadata["rrf_score"] > 0

    def test_sparse_only(self):
        from fusion import reciprocal_rank_fusion
        docs = [self._make_doc(f"Sparse doc {i}") for i in range(3)]
        result = reciprocal_rank_fusion([], docs, alpha=0.6, beta=0.4, final_top_k=3)
        assert len(result) == 3

    def test_deduplication(self):
        from fusion import reciprocal_rank_fusion
        doc = self._make_doc("Same content")
        # Same doc appears in both dense and sparse
        result = reciprocal_rank_fusion([doc], [doc], alpha=0.6, beta=0.4, final_top_k=5)
        assert len(result) == 1
        # Score should combine both sources
        assert "dense" in result[0].metadata["rrf_sources"]
        assert "sparse" in result[0].metadata["rrf_sources"]

    def test_ranking_order(self):
        from fusion import reciprocal_rank_fusion
        d1 = self._make_doc("Top doc")
        d2 = self._make_doc("Second doc")
        d3 = self._make_doc("Third doc")
        result = reciprocal_rank_fusion(
            [d1, d2, d3], [d1, d3, d2],
            alpha=0.5, beta=0.5, k=60, final_top_k=3
        )
        # d1 is rank 0 in both — should have highest score
        assert result[0].page_content == "Top doc"

    def test_top_k_limit(self):
        from fusion import reciprocal_rank_fusion
        docs = [self._make_doc(f"Doc {i}") for i in range(20)]
        result = reciprocal_rank_fusion(docs, docs, final_top_k=5)
        assert len(result) == 5

    def test_weights_influence(self):
        from fusion import reciprocal_rank_fusion
        dense_top = self._make_doc("Dense favourite")
        sparse_top = self._make_doc("Sparse favourite")
        # Dense-only doc at rank 0, sparse-only doc at rank 0
        result_dense_heavy = reciprocal_rank_fusion(
            [dense_top], [sparse_top], alpha=0.9, beta=0.1, k=60, final_top_k=2
        )
        # With alpha=0.9, the dense favourite should rank higher
        assert result_dense_heavy[0].page_content == "Dense favourite"


# ═══════════════════════════════════════════════════════════════
#  2. BM25 Sparse Index Tests
# ═══════════════════════════════════════════════════════════════

class TestBM25Index:
    """Test the BM25 sparse retriever build/search/persist lifecycle."""

    def _make_doc(self, content: str, **meta):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata=meta)

    def test_build_and_search(self):
        from sparse import BM25Index
        idx = BM25Index()
        docs = [
            self._make_doc("Python is a programming language"),
            self._make_doc("Java is another programming language"),
            self._make_doc("Cats are wonderful pets"),
        ]
        idx.build(docs)
        results = idx.search("programming language", top_k=2)
        assert len(results) <= 2
        # At least one result should be about programming
        texts = [r.page_content for r in results]
        assert any("programming" in t for t in texts)

    def test_search_returns_empty_for_unrelated(self):
        from sparse import BM25Index
        idx = BM25Index()
        docs = [self._make_doc("Alpha beta gamma delta")]
        idx.build(docs)
        results = idx.search("zzzznotinanytext", top_k=5)
        # BM25 score should be 0 for completely unrelated query
        assert len(results) == 0

    def test_save_and_load(self, tmp_path):
        from sparse import BM25Index
        import config
        # Override paths to temp directory
        original_idx = config.BM25_INDEX_PATH
        original_corpus = config.BM25_CORPUS_PATH
        config.BM25_INDEX_PATH = str(tmp_path / "test_bm25.pkl")
        config.BM25_CORPUS_PATH = str(tmp_path / "test_corpus.pkl")

        try:
            idx = BM25Index()
            docs = [self._make_doc("Quantum computing research")]
            idx.build(docs)
            idx.save()

            # Load into a fresh instance
            idx2 = BM25Index()
            idx2.load()
            results = idx2.search("quantum", top_k=1)
            assert len(results) == 1
            assert "Quantum" in results[0].page_content
        finally:
            config.BM25_INDEX_PATH = original_idx
            config.BM25_CORPUS_PATH = original_corpus


# ═══════════════════════════════════════════════════════════════
#  3. Feedback Store Tests (SQLite)
# ═══════════════════════════════════════════════════════════════

class TestFeedbackStore:
    """Test the SQLite feedback store CRUD operations."""
#  4. RLHF Manager Tests
# ═══════════════════════════════════════════════════════════════

class TestRLHFManager:
    """Test the RLHF reward adaptation and document boosting logic."""
#  5. Graph State & Edge Routing Tests
# ═══════════════════════════════════════════════════════════════

class TestIngestion:
    """Test document loading and chunking without touching real indexes."""

    def test_load_from_empty_dir(self, tmp_path):
        from ingest import load_documents
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_load_txt_file(self, tmp_path):
        from ingest import load_documents
        (tmp_path / "test.txt").write_text("Hello world\nLine two", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content

    def test_load_md_file(self, tmp_path):
        from ingest import load_documents
        (tmp_path / "readme.md").write_text("# Title\nContent here", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_chunk_documents(self):
        from ingest import chunk_documents
        from langchain_core.documents import Document
        # Create a document with enough content to be chunked
        long_text = "This is a test sentence. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "source_file" in chunk.metadata
            assert chunk.metadata["source_file"] == "test.txt"


# ═══════════════════════════════════════════════════════════════
#  7. Scraper Service Tests (Unit — no network)
# ═══════════════════════════════════════════════════════════════

class TestScraperHelpers:
    """Test scraper utility functions that don't require network."""

    def test_extract_markdown_with_string_fallback(self):
        from scraper import _extract_markdown

        class FakeResult:
            markdown = "raw markdown content"

        result = FakeResult()
        assert _extract_markdown(result) == "raw markdown content"

    def test_extract_markdown_with_fit_markdown(self):
        from scraper import _extract_markdown

        class FitContainer:
            fit_markdown = "clean filtered content"

        class FakeResult:
            markdown_v2 = FitContainer()
            markdown = "raw fallback"

        result = FakeResult()
        assert _extract_markdown(result) == "clean filtered content"

    def test_extract_markdown_empty(self):
        from scraper import _extract_markdown

        class FakeResult:
            pass

        result = FakeResult()
        assert _extract_markdown(result) == ""

    def test_skip_domains(self):
        from scraper import _SKIP_DOMAINS
        assert "google.com" in _SKIP_DOMAINS
        assert "youtube.com" in _SKIP_DOMAINS


# ═══════════════════════════════════════════════════════════════
#  8. Config Tests
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    """Verify configuration defaults and consistency."""

    def test_config_loads(self):
        import config
        assert config.DENSE_WEIGHT + config.SPARSE_WEIGHT == pytest.approx(1.0)
        assert config.DENSE_TOP_K > 0
        assert config.SPARSE_TOP_K > 0
        assert config.FUSION_TOP_K > 0
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_embedding_model_matches_env(self):
        """Ensure config default matches what .env expects."""
        import config
        # After the fix, both should agree on gemini-embedding-2
        assert config.GEMINI_EMBEDDING_MODEL == "gemini-embedding-2"