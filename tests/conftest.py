import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Set mock API keys for testing before any imports occur
os.environ["GOOGLE_API_KEY"] = "mock_key_for_testing"
os.environ["GEMINI_API_KEY"] = "mock_key_for_testing"
os.environ["CHROMA_HOST"] = "localhost"

# Append backend directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

@pytest.fixture(autouse=True)
def mock_gemini_client():
    """Globally mock langchain_google_genai to avoid real API calls."""
    with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm, \
         patch('langchain_google_genai.GoogleGenerativeAIEmbeddings') as mock_embed:
        yield mock_llm, mock_embed

@pytest.fixture(autouse=True)
def mock_chroma_client():
    """Globally mock langchain_chroma and chromadb to avoid network connections."""
    with patch('langchain_chroma.Chroma') as mock_chroma:
        yield mock_chroma

@pytest.fixture
def store(tmp_path):
    from store import FeedbackStore
    db_path = str(tmp_path / "test_feedback.db")
    return FeedbackStore(db_path=db_path)

@pytest.fixture
def rlhf(tmp_path):
    import config
    original = config.FEEDBACK_DB_PATH
    config.FEEDBACK_DB_PATH = str(tmp_path / "rlhf_test.db")
    try:
        from rlhf import RLHFManager
        yield RLHFManager()
    finally:
        config.FEEDBACK_DB_PATH = original
