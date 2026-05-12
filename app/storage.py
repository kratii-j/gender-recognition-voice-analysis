from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    pitch_hz REAL NOT NULL,
                    rms_energy REAL NOT NULL,
                    zero_crossing_rate REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def insert(self, record: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    id, filename, label, confidence, pitch_hz, rms_energy,
                    zero_crossing_rate, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["filename"],
                    record["label"],
                    record["confidence"],
                    record["pitch_hz"],
                    record["rms_energy"],
                    record["zero_crossing_rate"],
                    record["duration_seconds"],
                    record["created_at"],
                ),
            )

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, label, confidence, pitch_hz, rms_energy,
                       zero_crossing_rate, duration_seconds, created_at
                FROM predictions
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
