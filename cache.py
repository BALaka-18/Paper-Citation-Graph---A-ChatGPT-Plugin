"""
Flat key-value cache keyed by (paper_id, stage), backed by SQLite.

Every expensive stage (metadata fetch, citation fetch, extraction, matching,
classification) writes its result here so an interrupted or re-run pipeline
doesn't burn API calls or rate-limit budget redoing finished work.
"""
import json
import sqlite3
from typing import Any, Optional


class Cache:
    def __init__(self, db_path: str = "cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                paper_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (paper_id, stage)
            )
            """
        )
        self.conn.commit()

    def get(self, paper_id: str, stage: str) -> Optional[Any]:
        row = self.conn.execute(
            "SELECT value FROM cache WHERE paper_id = ? AND stage = ?", (paper_id, stage)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, paper_id: str, stage: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (paper_id, stage, value) VALUES (?, ?, ?)",
            (paper_id, stage, json.dumps(value)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
