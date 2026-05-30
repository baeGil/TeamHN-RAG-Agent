import json
import uuid
from typing import Any, Optional

from .database import Database


class Repo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------- documents ----------
    def add_document(self, title: str, source: str, source_type: str) -> int:
        cur = self.db.conn.execute(
            "INSERT INTO documents(title, source, source_type) VALUES (?,?,?)",
            (title, source, source_type),
        )
        self.db.conn.commit()
        return int(cur.lastrowid)

    def set_document_chunk_count(self, doc_id: int, n: int) -> None:
        self.db.conn.execute("UPDATE documents SET n_chunks=? WHERE id=?", (n, doc_id))
        self.db.conn.commit()

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.db.conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int) -> list[int]:
        chunk_ids = [
            int(r["id"])
            for r in self.db.conn.execute(
                "SELECT id FROM chunks WHERE document_id=?", (doc_id,)
            ).fetchall()
        ]
        self.db.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.db.conn.commit()
        return chunk_ids

    # ---------- chunks ----------
    def add_chunk(
        self,
        document_id: int,
        chunk_index: int,
        text: str,
        page: Optional[int],
        section: Optional[str],
    ) -> int:
        cur = self.db.conn.execute(
            "INSERT INTO chunks(document_id, chunk_index, text, page, section) VALUES (?,?,?,?,?)",
            (document_id, chunk_index, text, page, section),
        )
        self.db.conn.commit()
        return int(cur.lastrowid)

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.db.conn.execute(
            f"""SELECT c.*, d.title AS doc_title, d.source AS doc_source,
                       d.source_type AS doc_source_type
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
        return {int(r["id"]): dict(r) for r in rows}

    def all_chunks(self) -> list[dict[str, Any]]:
        rows = self.db.conn.execute(
            """SELECT c.*, d.title AS doc_title, d.source AS doc_source,
                      d.source_type AS doc_source_type
               FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.id"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- sessions ----------
    def create_session(self, title: Optional[str] = None) -> str:
        sid = uuid.uuid4().hex
        self.db.conn.execute(
            "INSERT INTO sessions(id, title) VALUES (?, COALESCE(?, 'Cuộc trò chuyện mới'))",
            (sid, title),
        )
        self.db.conn.commit()
        return sid

    def ensure_session(self, sid: str) -> None:
        exists = self.db.conn.execute(
            "SELECT 1 FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        if not exists:
            self.db.conn.execute("INSERT INTO sessions(id) VALUES (?)", (sid,))
            self.db.conn.commit()

    def touch_session(self, sid: str, title: Optional[str] = None) -> None:
        if title:
            self.db.conn.execute(
                "UPDATE sessions SET updated_at=datetime('now'), title=? WHERE id=?",
                (title, sid),
            )
        else:
            self.db.conn.execute(
                "UPDATE sessions SET updated_at=datetime('now') WHERE id=?", (sid,)
            )
        self.db.conn.commit()

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self.db.conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, sid: str) -> None:
        self.db.conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        self.db.conn.commit()

    # ---------- messages ----------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[list] = None,
        trace: Optional[list] = None,
    ) -> int:
        cur = self.db.conn.execute(
            "INSERT INTO messages(session_id, role, content, citations, trace) VALUES (?,?,?,?,?)",
            (
                session_id,
                role,
                content,
                json.dumps(citations, ensure_ascii=False) if citations is not None else None,
                json.dumps(trace, ensure_ascii=False) if trace is not None else None,
            ),
        )
        self.db.conn.commit()
        return int(cur.lastrowid)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.db.conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["citations"] = json.loads(d["citations"]) if d["citations"] else []
            d["trace"] = json.loads(d["trace"]) if d["trace"] else []
            out.append(d)
        return out

    def recent_history(self, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        rows = self.db.conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
