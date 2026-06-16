"""
Pydantic models for the feedback / RLHF subsystem.
Shared across API, storage, and the reward manager.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FeedbackSignal(str, Enum):
    """Binary human preference signal."""
    LIKE = "like"
    DISLIKE = "dislike"


# ── API Request / Response ────────────────────────────────────

class FeedbackRequest(BaseModel):
    """POST /feedback request body."""
    interaction_id: str = Field(..., description="ID of the chat interaction")
    signal: FeedbackSignal = Field(..., description="like or dislike")
    comment: Optional[str] = Field(None, description="Optional free-text comment")


class FeedbackResponse(BaseModel):
    """POST /feedback response body."""
    success: bool
    feedback_id: str
    weights_updated: bool
    current_weights: dict


# ── Internal Records ──────────────────────────────────────────

class InteractionRecord(BaseModel):
    """Represents a saved chat interaction for audit and RLHF."""
    interaction_id: str
    question: str
    generation: str
    documents: list[dict]
    rewrite_count: int = 0
    relevance_score: float = 0.0
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    timestamp: str = ""


class ChunkReward(BaseModel):
    """Per-chunk accumulated reward from human feedback."""
    chunk_id: str
    total_reward: float = 0.0
    like_count: int = 0
    dislike_count: int = 0
