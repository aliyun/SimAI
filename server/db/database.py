"""SQLite database management with WAL mode for concurrent access."""

import os
import sqlite3
import logging
from contextlib import contextmanager

from server.config import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(WORKSPACE_ROOT, "process_tracker.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    workspace_dir TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_active REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS processes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pid INTEGER NOT NULL,
    session_token TEXT NOT NULL,
    username TEXT NOT NULL,
    workspace_dir TEXT NOT NULL,
    command TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    finished_at REAL,
    return_code INTEGER,
    error_message TEXT,
    FOREIGN KEY (session_token) REFERENCES sessions(token)
);

CREATE INDEX IF NOT EXISTS idx_processes_session ON processes(session_token);
CREATE INDEX IF NOT EXISTS idx_processes_status ON processes(status);
CREATE INDEX IF NOT EXISTS idx_processes_username ON processes(username);
CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username);
"""


@contextmanager
def get_db():
    """Get a database connection with WAL mode for concurrent access."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database schema."""
    with get_db() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", DB_PATH)
