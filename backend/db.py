from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Optional, Tuple


DB_PATH = Path(__file__).resolve().parents[1] / "database" / "habits.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS habit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                habit_id INTEGER NOT NULL,
                log_date TEXT NOT NULL,
                status INTEGER NOT NULL CHECK(status IN (0, 1)),
                UNIQUE(habit_id, log_date),
                FOREIGN KEY (habit_id) REFERENCES habits(id) ON DELETE CASCADE
            );
            """
        )


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


def parse_iso_date(s: str) -> date:
    # ISO only for API simplicity.
    return date.fromisoformat(s)


def get_habit_by_name(name: str) -> Optional[Tuple[int, str]]:
    with connect() as conn:
        row = conn.execute("SELECT id, name FROM habits WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return int(row["id"]), str(row["name"])


def get_habit(habit_id: int) -> Optional[Tuple[int, str]]:
    with connect() as conn:
        row = conn.execute("SELECT id, name FROM habits WHERE id = ?", (habit_id,)).fetchone()
        if not row:
            return None
        return int(row["id"]), str(row["name"])

