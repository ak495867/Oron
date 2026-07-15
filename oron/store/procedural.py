import sqlite3
from typing import Any, Optional, Dict
import json
import os


class ProceduralStore:
    """
    Procedural memory store using SQLite for key-value storage of
    high-frequency patterns and user preferences.
    """

    def __init__(self, db_path: str = "oron_procedural.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memory (
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    use_count INTEGER DEFAULT 1,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, key)
                )
            """)

    def set(self, user_id: str, key: str, value: Any) -> None:
        val_str = json.dumps(value)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO procedural_memory (user_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    use_count = use_count + 1,
                    last_used = CURRENT_TIMESTAMP
            """,
                (user_id, key, val_str),
            )

    def get(self, user_id: str, key: str) -> Optional[Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM procedural_memory WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
            row = cursor.fetchone()
            if row:
                # Update use count on retrieval
                conn.execute(
                    "UPDATE procedural_memory SET use_count = use_count + 1, last_used = CURRENT_TIMESTAMP WHERE user_id = ? AND key = ?",
                    (user_id, key),
                )
                return json.loads(row[0])
        return None

    def get_all(self, user_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT key, value FROM procedural_memory WHERE user_id = ?", (user_id,)
            )
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def delete(self, user_id: str, key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM procedural_memory WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
