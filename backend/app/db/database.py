import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    source_type TEXT NOT NULL,         -- pdf | url | text
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'ready',  -- ready | processing | failed
    error_message TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,   -- == turbovec uint64 id
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    section     TEXT,
    embedding   BLOB,                  -- float32 vector; lets us rebuild the dense index from the DB
    embed_text  TEXT,                   -- optimized text for embedding (from Reducto embed field)
    hype_questions TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Cuộc trò chuyện mới',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,         -- user | assistant
    content     TEXT NOT NULL,
    citations   TEXT,                  -- JSON
    trace       TEXT,                  -- JSON list of agent events
    status      TEXT NOT NULL DEFAULT 'complete',  -- processing | complete | failed
    error_message TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL,
    summary           TEXT NOT NULL,
    summarized_up_to  INTEGER NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conv_summaries_session ON conversation_summaries(session_id);
"""

_local = threading.local()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conns = getattr(_local, "conns", None)
        if conns is None:
            conns = {}
            _local.conns = conns
        conn = conns.get(self.path)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            conns[self.path] = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(_SCHEMA)
        self._migrate(conn)
        conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "embedding" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
        if "embed_text" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN embed_text TEXT")
        if "hype_questions" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN hype_questions TEXT")

        doc_cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "status" not in doc_cols:
            conn.execute("ALTER TABLE documents ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'")
        if "error_message" not in doc_cols:
            conn.execute("ALTER TABLE documents ADD COLUMN error_message TEXT")

        msg_cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "status" not in msg_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'")
        if "error_message" not in msg_cols:
            conn.execute("ALTER TABLE messages ADD COLUMN error_message TEXT")

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn()
