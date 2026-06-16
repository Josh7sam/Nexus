"""
Feedback store — SQLite-backed persistence for interactions, human
feedback signals, per-chunk rewards, and retrieval weight history.

This is the data layer for the RLHF system.  The RLHFManager reads
from and writes to this store.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from config import FEEDBACK_DB_PATH


class FeedbackStore:
    """Thread-safe SQLite store for RLHF data."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or FEEDBACK_DB_PATH
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    # ══════════════════════════════════════════════════════════
    #  Schema
    # ══════════════════════════════════════════════════════════

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id  TEXT PRIMARY KEY,
                    question        TEXT NOT NULL,
                    generation      TEXT NOT NULL,
                    documents       TEXT NOT NULL DEFAULT '[]',
                    rewrite_count   INTEGER DEFAULT 0,
                    relevance_score REAL    DEFAULT 0.0,
                    dense_weight    REAL    DEFAULT 0.6,
                    sparse_weight   REAL    DEFAULT 0.4,
                    title           TEXT,
                    user_id         TEXT DEFAULT 'default',
                    created_at      TEXT    NOT NULL,
                    UNIQUE(title, user_id)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id              TEXT PRIMARY KEY,
                    interaction_id  TEXT NOT NULL,
                    signal          TEXT NOT NULL,
                    comment         TEXT,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (interaction_id)
                        REFERENCES interactions(interaction_id)
                );

                CREATE TABLE IF NOT EXISTS chunk_rewards (
                    chunk_id        TEXT PRIMARY KEY,
                    chunk_content   TEXT NOT NULL DEFAULT '',
                    total_reward    REAL    DEFAULT 0.0,
                    like_count      INTEGER DEFAULT 0,
                    dislike_count   INTEGER DEFAULT 0,
                    updated_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS weight_history (
                    id              TEXT PRIMARY KEY,
                    dense_weight    REAL NOT NULL,
                    sparse_weight   REAL NOT NULL,
                    avg_reward      REAL NOT NULL,
                    created_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key             TEXT PRIMARY KEY,
                    value           TEXT NOT NULL
                );
            """)

            # Migrate old database if title or user_id columns are missing
            try:
                conn.execute("ALTER TABLE interactions ADD COLUMN title TEXT")
            except sqlite3.OperationalError:
                pass  # Already exists

            try:
                conn.execute("ALTER TABLE interactions ADD COLUMN user_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:
                pass  # Already exists

            try:
                # Clean up existing duplicate titles to allow unique index creation
                conn.execute("""
                    DELETE FROM interactions
                    WHERE title IS NOT NULL
                      AND rowid NOT IN (
                          SELECT MAX(rowid)
                          FROM interactions
                          WHERE title IS NOT NULL
                          GROUP BY title
                      )
                """)
                # Create unique index to enforce unique constraint on title + user_id
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_title_userid ON interactions(title, user_id)")
            except sqlite3.OperationalError:
                pass

            # Seed default settings if empty
            cursor = conn.execute("SELECT COUNT(*) FROM settings")
            if cursor.fetchone()[0] == 0:
                from config import (
                    DENSE_TOP_K, SPARSE_TOP_K, FUSION_TOP_K, RRF_K_CONSTANT,
                    MAX_REWRITE_ATTEMPTS, MAX_HALLUCINATION_RETRIES,
                    DENSE_WEIGHT, SPARSE_WEIGHT
                )
                defaults = {
                    "dense_top_k": str(DENSE_TOP_K),
                    "sparse_top_k": str(SPARSE_TOP_K),
                    "fusion_top_k": str(FUSION_TOP_K),
                    "rrf_k_constant": str(RRF_K_CONSTANT),
                    "max_rewrite_attempts": str(MAX_REWRITE_ATTEMPTS),
                    "max_hallucination_retries": str(MAX_HALLUCINATION_RETRIES),
                    "dense_weight": str(DENSE_WEIGHT),
                    "sparse_weight": str(SPARSE_WEIGHT)
                }
                for k, v in defaults.items():
                    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (k, v))
            else:
                # Migrate old default fusion_top_k from 6 to 3
                conn.execute("UPDATE settings SET value = '3' WHERE key = 'fusion_top_k' AND value = '6'")
                # Migrate old default max_hallucination_retries from 2 to 0
                conn.execute("UPDATE settings SET value = '0' WHERE key = 'max_hallucination_retries' AND value = '2'")

    def _conn(self):
        class ConnectionContext:
            def __init__(self, db_path):
                self.db_path = db_path
                self.conn = None

            def __enter__(self):
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=5000")
                self.conn.__enter__()
                return self.conn

            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    self.conn.__exit__(exc_type, exc_val, exc_tb)
                finally:
                    self.conn.close()

        return ConnectionContext(self.db_path)

    # ══════════════════════════════════════════════════════════
    #  Interactions
    # ══════════════════════════════════════════════════════════

    def save_interaction(
        self,
        interaction_id: str,
        question: str,
        generation: str,
        documents: list,
        rewrite_count: int = 0,
        relevance_score: float = 0.0,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        title: str | None = None,
    ) -> None:
        """Persist a completed chat interaction."""
        docs_json = json.dumps([
            {
                "content": getattr(doc, "page_content", str(doc))[:500],
                "metadata": {
                    k: str(v)
                    for k, v in getattr(doc, "metadata", {}).items()
                },
            }
            for doc in documents
        ]) if documents else "[]"

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO interactions
                   (interaction_id, question, generation, documents,
                    rewrite_count, relevance_score, dense_weight,
                    sparse_weight, title, user_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    interaction_id, question, generation, docs_json,
                    rewrite_count, relevance_score, dense_weight,
                    sparse_weight, title, "default", _now(),
                ),
            )

    # ══════════════════════════════════════════════════════════
    #  Feedback
    # ══════════════════════════════════════════════════════════

    def save_feedback(
        self,
        interaction_id: str,
        signal: str,
        comment: str | None = None,
    ) -> str:
        """
        Record a like / dislike and propagate reward to associated chunks.
        Returns the generated feedback ID.
        """
        feedback_id = str(uuid.uuid4())
        reward = 1.0 if signal == "like" else -1.0
        now = _now()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO feedback
                   (id, interaction_id, signal, comment, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (feedback_id, interaction_id, signal, comment, now),
            )

            # Propagate reward to every chunk used in this interaction
            row = conn.execute(
                "SELECT documents FROM interactions WHERE interaction_id = ?",
                (interaction_id,),
            ).fetchone()

            if row:
                for doc in json.loads(row[0]):
                    chunk_id = str(
                        doc.get("metadata", {}).get("chunk_id", "")
                        or hash(doc.get("content", ""))
                    )
                    content = doc.get("content", "")
                    like_inc = 1 if signal == "like" else 0
                    dislike_inc = 1 if signal == "dislike" else 0

                    conn.execute(
                        """INSERT INTO chunk_rewards
                              (chunk_id, chunk_content, total_reward,
                               like_count, dislike_count, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(chunk_id) DO UPDATE SET
                              total_reward  = total_reward + ?,
                              like_count    = like_count   + ?,
                              dislike_count = dislike_count + ?,
                              updated_at    = ?""",
                        (
                            chunk_id, content, reward,
                            like_inc, dislike_inc, now,
                            reward, like_inc, dislike_inc, now,
                        ),
                    )

        return feedback_id

    # ══════════════════════════════════════════════════════════
    #  Reward Queries
    # ══════════════════════════════════════════════════════════

    def get_chunk_rewards(self) -> dict[str, dict]:
        """Return all per-chunk reward data as {chunk_id: {reward info}}."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, total_reward, like_count, dislike_count "
                "FROM chunk_rewards"
            ).fetchall()

        return {
            r[0]: {"total_reward": r[1], "like_count": r[2], "dislike_count": r[3]}
            for r in rows
        }

    # ══════════════════════════════════════════════════════════
    #  Statistics
    # ══════════════════════════════════════════════════════════

    def get_feedback_stats(self) -> dict:
        """Aggregate statistics for the RLHF dashboard."""
        with self._conn() as conn:
            total_interactions = conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]

            total_likes = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE signal = 'like'"
            ).fetchone()[0]

            total_dislikes = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE signal = 'dislike'"
            ).fetchone()[0]

            row = conn.execute(
                "SELECT dense_weight, sparse_weight "
                "FROM weight_history ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        total_fb = total_likes + total_dislikes
        return {
            "total_interactions": total_interactions,
            "total_likes": total_likes,
            "total_dislikes": total_dislikes,
            "satisfaction_rate": round(
                total_likes / max(total_fb, 1), 3
            ),
            "current_weights": {
                "dense": row[0] if row else 0.6,
                "sparse": row[1] if row else 0.4,
            },
        }

    # ══════════════════════════════════════════════════════════
    #  Weight History
    # ══════════════════════════════════════════════════════════

    def save_weight_update(
        self, dense_weight: float, sparse_weight: float, avg_reward: float
    ) -> None:
        """Append a new weight snapshot to the history table."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO weight_history
                   (id, dense_weight, sparse_weight, avg_reward, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), dense_weight, sparse_weight, avg_reward, _now()),
            )

    def get_recent_interactions(self, limit: int = 20) -> list[dict]:
        """Fetch the most recent interactions with their feedback status."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT i.interaction_id, i.question, i.generation,
                          i.created_at, f.signal, i.documents, i.title
                   FROM interactions i
                   LEFT JOIN feedback f ON i.interaction_id = f.interaction_id
                   ORDER BY i.created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            {
                "interaction_id": r[0],
                "question": r[1],
                "generation": r[2],
                "timestamp": r[3],
                "feedback": r[4],
                "documents": json.loads(r[5]) if r[5] else [],
                "title": r[6] if len(r) > 6 else None,
            }
            for r in rows
        ]

    def get_settings(self) -> dict[str, str]:
        """Fetch all settings from the database."""
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r[0]: r[1] for r in rows}

    def save_settings(self, settings: dict[str, str]) -> None:
        """Update settings in the database."""
        with self._conn() as conn:
            for k, v in settings.items():
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    (k, str(v)),
                )

    def clear_all_data(self) -> None:
        """Clear all interaction, feedback, and chunk reward data to reset test state."""
        with self._conn() as conn:
            conn.execute("DELETE FROM feedback")
            conn.execute("DELETE FROM interactions")
            conn.execute("DELETE FROM chunk_rewards")
            conn.execute("DELETE FROM weight_history")

    def delete_interaction(self, interaction_id: str) -> None:
        """Delete a specific interaction and its associated feedback."""
        with self._conn() as conn:
            conn.execute("DELETE FROM feedback WHERE interaction_id = ?", (interaction_id,))
            conn.execute("DELETE FROM interactions WHERE interaction_id = ?", (interaction_id,))



# ── Helpers ───────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
