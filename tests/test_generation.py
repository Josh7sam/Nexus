import os
import sys
import json
import math
import tempfile
import uuid
import pytest
from langchain_core.documents import Document


class TestEdgeRouting:
    """Test conditional routing functions with mocked state dicts."""

    def test_route_after_router_direct(self):
        from edges import route_after_router
        assert route_after_router({"route_decision": "direct"}) == "generate_direct"

    def test_route_after_router_retrieve(self):
        from edges import route_after_router
        assert route_after_router({"route_decision": "retrieve"}) == "hybrid_retrieve"

    def test_route_after_router_default(self):
        from edges import route_after_router
        # Missing key defaults to retrieve
        assert route_after_router({}) == "hybrid_retrieve"

    def test_route_after_grading_with_docs(self):
        from edges import route_after_grading
        from langchain_core.documents import Document
        state = {"documents": [Document(page_content="x")], "rewrite_count": 0}
        assert route_after_grading(state) == "generate"

    def test_route_after_grading_no_docs_rewrites_left(self):
        from edges import route_after_grading
        state = {"documents": [], "rewrite_count": 0}
        assert route_after_grading(state) == "rewrite_query"

    def test_route_after_grading_no_docs_exhausted(self):
        from edges import route_after_grading
        state = {"documents": [], "rewrite_count": 10}
        assert route_after_grading(state) == "generate"

    def test_route_after_hallucination_grounded(self):
        from edges import route_after_hallucination
        from langgraph.graph import END
        state = {"is_grounded": True, "hallucination_retries": 0, "documents": ["doc"]}
        assert route_after_hallucination(state) == END

    def test_route_after_hallucination_not_grounded_retries_left(self, monkeypatch):
        from store import FeedbackStore
        monkeypatch.setattr(FeedbackStore, "get_settings", lambda self: {"max_hallucination_retries": 2})
        import edges
        monkeypatch.setattr(edges, "MAX_HALLUCINATION_RETRIES", 2)
        from edges import route_after_hallucination
        state = {"is_grounded": False, "hallucination_retries": 0, "documents": ["doc"]}
        assert route_after_hallucination(state) == "generate"

    def test_route_after_generate_with_retries(self, monkeypatch):
        from store import FeedbackStore
        monkeypatch.setattr(FeedbackStore, "get_settings", lambda self: {"max_hallucination_retries": 2})
        import edges
        monkeypatch.setattr(edges, "MAX_HALLUCINATION_RETRIES", 2)
        from edges import route_after_generate
        # Default MAX_HALLUCINATION_RETRIES is 2, so it should route to hallucination_check
        state = {}
        assert route_after_generate(state) == "hallucination_check"

    def test_route_after_generate_no_retries(self, monkeypatch):
        from store import FeedbackStore
        monkeypatch.setattr(FeedbackStore, "get_settings", lambda self: {"max_hallucination_retries": 0})
        import edges
        monkeypatch.setattr(edges, "MAX_HALLUCINATION_RETRIES", 0)
        from edges import route_after_generate
        from langgraph.graph import END
        state = {}
        assert route_after_generate(state) == END


# ═══════════════════════════════════════════════════════════════
#  5b. Router Node Fast Path Tests
# ═══════════════════════════════════════════════════════════════

class TestRouterNode:
    """Test the rule-based chitchat fast path in router_node."""

    def test_greetings_fast_path(self):
        from nodes_agent import router_node
        # Greetings should go directly to direct
        state = {"question": "Hello there!"}
        result = router_node(state)
        assert result["route_decision"] == "direct"

    def test_gratitude_fast_path(self):
        from nodes_agent import router_node
        # Thank you should go directly to direct
        state = {"question": "thank you very much"}
        result = router_node(state)
        assert result["route_decision"] == "direct"

    def test_farewell_fast_path(self):
        from nodes_agent import router_node
        # Goodbye should go directly to direct
        state = {"question": "see you later"}
        result = router_node(state)
        assert result["route_decision"] == "direct"

    def test_bot_identity_fast_path(self):
        from nodes_agent import router_node
        # Identity questions should go directly to direct
        state = {"question": "who are you?"}
        result = router_node(state)
        assert result["route_decision"] == "direct"


# ═══════════════════════════════════════════════════════════════

#  6. Ingestion Pipeline Tests (Document Loading & Chunking)
# ═══════════════════════════════════════════════════════════════