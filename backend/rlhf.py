"""
RLHF Manager — Reinforcement Learning from Human Feedback.

Uses a lightweight reward-based approach to adapt the RAG pipeline:

  1. **Weight Adaptation**   — shifts dense/sparse retrieval balance
     toward whichever method correlates with positive feedback.
  2. **Document Boosting**   — multiplies RRF scores by a learned
     sigmoid-based boost derived from per-chunk reward history.
  3. **Statistics**          — exposes dashboard-ready metrics.

No external ML framework required; works with plain Python + SQLite.
"""

import math
from langchain_core.documents import Document
from store import FeedbackStore
from config import DENSE_WEIGHT, SPARSE_WEIGHT


class RLHFManager:
    """Stateful manager for processing feedback and adapting retrieval."""

    # ── Hyper-parameters ──────────────────────────────────────
    LEARNING_RATE = 0.05       # How fast weights shift per update
    MIN_WEIGHT    = 0.15       # Floor for any retrieval method
    MAX_WEIGHT    = 0.85       # Ceiling for any retrieval method
    MIN_FEEDBACK  = 5          # Minimum signals before adapting weights
    BOOST_RANGE   = 0.4        # Max boost/penalty magnitude (±)

    def __init__(self) -> None:
        self.store = FeedbackStore()
        settings = {}
        try:
            settings = self.store.get_settings()
        except Exception:
            pass
        self._dense_w = float(settings.get("dense_weight", DENSE_WEIGHT))
        self._sparse_w = float(settings.get("sparse_weight", SPARSE_WEIGHT))

    # ══════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════

    def get_current_weights(self) -> tuple[float, float]:
        """Return (dense_weight, sparse_weight) — possibly RLHF-adapted."""
        stats = self.store.get_feedback_stats()
        # If no weight history exists yet, return settings (or default)
        settings = self.store.get_settings()
        dense = float(settings.get("dense_weight", stats["current_weights"]["dense"]))
        sparse = float(settings.get("sparse_weight", stats["current_weights"]["sparse"]))

        # If weight history exists in database, stats["current_weights"] represents the latest weight history.
        # Check if we have any weight history
        with self.store._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM weight_history").fetchone()
            has_history = row[0] > 0 if row else False

        if has_history:
            return (
                stats["current_weights"]["dense"],
                stats["current_weights"]["sparse"],
            )
        return dense, sparse

    def process_feedback(
        self,
        interaction_id: str,
        signal: str,
        comment: str | None = None,
    ) -> dict:
        """
        Accept a human preference signal, persist it, and trigger
        a weight-adaptation step.

        Returns summary dict for the API response.
        """
        feedback_id = self.store.save_feedback(interaction_id, signal, comment)
        self._adapt_weights()

        return {
            "feedback_id": feedback_id,
            "weights_updated": True,
            "current_weights": {
                "dense": round(self._dense_w, 4),
                "sparse": round(self._sparse_w, 4),
            },
        }

    def boost_documents(self, documents: list[Document]) -> list[Document]:
        """
        Re-score documents using per-chunk RLHF rewards.

        Chunks that historically received positive feedback get a
        multiplicative boost (up to 1 + BOOST_RANGE); disliked chunks
        get penalised (down to 1 - BOOST_RANGE).
        """
        boosts = self._compute_boost_scores()
        if not boosts:
            return documents

        for doc in documents:
            cid = str(doc.metadata.get("chunk_id", ""))
            if cid in boosts:
                factor = boosts[cid]
                original = doc.metadata.get("rrf_score", 1.0)
                doc.metadata["rrf_score"] = original * factor
                doc.metadata["rlhf_boost"] = round(factor, 4)

        # Re-sort by boosted RRF score
        documents.sort(
            key=lambda d: d.metadata.get("rrf_score", 0), reverse=True
        )
        return documents

    def get_stats(self) -> dict:
        """Return comprehensive RLHF metrics for the dashboard."""
        stats = self.store.get_feedback_stats()
        boosts = self._compute_boost_scores()
        return {
            **stats,
            "boosted_chunks": len(boosts),
            "learning_rate": self.LEARNING_RATE,
            "weight_bounds": {"min": self.MIN_WEIGHT, "max": self.MAX_WEIGHT},
        }

    # ══════════════════════════════════════════════════════════
    #  Internal — Weight Adaptation
    # ══════════════════════════════════════════════════════════

    def _adapt_weights(self) -> None:
        """
        Gradient-like weight update based on accumulated satisfaction.

        When satisfaction drops below 50 %, nudge weights away from
        the current balance.  Uses EMA-style clamped updates.
        """
        stats = self.store.get_feedback_stats()
        total = stats["total_likes"] + stats["total_dislikes"]

        if total < self.MIN_FEEDBACK:
            return  # Not enough data

        satisfaction = stats["satisfaction_rate"]

        if satisfaction < 0.5:
            delta = self.LEARNING_RATE * (satisfaction - 0.5)
            self._dense_w = _clamp(
                self._dense_w + delta, self.MIN_WEIGHT, self.MAX_WEIGHT
            )
            self._sparse_w = _clamp(
                1.0 - self._dense_w, self.MIN_WEIGHT, self.MAX_WEIGHT
            )
            # Renormalise to sum = 1
            s = self._dense_w + self._sparse_w
            self._dense_w /= s
            self._sparse_w /= s

        avg_reward = (stats["total_likes"] - stats["total_dislikes"]) / max(total, 1)
        self.store.save_weight_update(self._dense_w, self._sparse_w, avg_reward)

    # ══════════════════════════════════════════════════════════
    #  Internal — Document Boost Scores
    # ══════════════════════════════════════════════════════════

    def _compute_boost_scores(self) -> dict[str, float]:
        """
        Map chunk_id → multiplicative boost factor ∈ [1-BOOST_RANGE, 1+BOOST_RANGE].

        Uses tanh to smoothly squash the normalised reward into a
        bounded range, preventing any single chunk from dominating.
        """
        rewards = self.store.get_chunk_rewards()
        boosts: dict[str, float] = {}

        for cid, data in rewards.items():
            total = data["like_count"] + data["dislike_count"]
            if total == 0:
                continue
            normalised = data["total_reward"] / total  # ∈ [-1, 1]
            boost = 1.0 + self.BOOST_RANGE * math.tanh(normalised)
            boosts[cid] = boost

        return boosts


# ── Helpers ───────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Module-level singleton ────────────────────────────────────
_manager: RLHFManager | None = None


def get_rlhf_manager() -> RLHFManager:
    """Return the global RLHFManager instance."""
    global _manager
    if _manager is None:
        _manager = RLHFManager()
    return _manager
