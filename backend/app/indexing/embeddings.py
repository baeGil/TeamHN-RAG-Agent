"""OpenAI embeddings with L2-normalization (for cosine via inner product) + disk cache."""
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import get_settings


class Embedder:
    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self._client = None
        self._dim: Optional[int] = self.settings.embed_dim
        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._local = threading.local()
        if cache_path is not None:
            self._init_cache()

    # ---- cache (sqlite key-value of float32 blobs) ----
    def _cache_conn(self) -> Optional[sqlite3.Connection]:
        if self._cache_path is None:
            return None
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return conn

    def _init_cache(self) -> None:
        conn = self._cache_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        conn.commit()

    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.settings.embed_model}::{text}".encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, text: str) -> Optional[np.ndarray]:
        conn = self._cache_conn()
        if conn is None:
            return None
        row = conn.execute("SELECT v FROM emb WHERE k=?", (self._key(text),)).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        conn = self._cache_conn()
        if conn is None:
            return
        conn.execute(
            "INSERT OR REPLACE INTO emb(k, v) VALUES (?, ?)",
            (self._key(text), vec.astype(np.float32).tobytes()),
        )
        conn.commit()

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.settings.has_openai:
                raise RuntimeError(
                    "OPENAI_API_KEY chưa được cấu hình. Vui lòng điền vào backend/.env"
                )
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._client

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    # Maximum tokens allowed by the embedding model
    _MAX_EMBED_TOKENS: int = 8000  # slightly under 8192 for safety

    def _truncate_text(self, text: str) -> str:
        """Truncate text to _MAX_EMBED_TOKENS tokens to avoid API limit errors."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.settings.embed_model)
            tokens = enc.encode(text)
            if len(tokens) > self._MAX_EMBED_TOKENS:
                tokens = tokens[: self._MAX_EMBED_TOKENS]
                text = enc.decode(tokens)
        except Exception:
            # Fallback: ~4 chars per token heuristic
            max_chars = self._MAX_EMBED_TOKENS * 4
            if len(text) > max_chars:
                text = text[:max_chars]
        return text

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        # Truncate any oversized chunks before sending to the API
        texts = [self._truncate_text(t) for t in texts]
        dim = self._dim or 1536
        if getattr(self.settings, "_quota_exceeded", False):
            return np.zeros((len(texts), dim), dtype=np.float32)
        try:
            resp = self.client.embeddings.create(model=self.settings.embed_model, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return arr
        except Exception as e:
            err_msg = str(e).lower()
            if "quota" in err_msg or "exceeded" in err_msg or "429" in err_msg:
                self.settings._quota_exceeded = True
                return np.zeros((len(texts), dim), dtype=np.float32)
            raise

    def embed_documents(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        missing_idx: list[int] = []
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)
        for start in range(0, len(missing_idx), batch_size):
            batch = missing_idx[start : start + batch_size]
            arr = self._embed_raw([texts[i] for i in batch])
            arr = self._normalize(arr)
            for j, i in enumerate(batch):
                results[i] = arr[j]
                self._cache_put(texts[i], arr[j])
        out = np.vstack(results).astype(np.float32) if results else np.zeros((0, 1), np.float32)
        if self._dim is None and out.shape[0] > 0:
            self._dim = int(out.shape[1])
        return out

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.embed_documents([text])[0]
        return vec.reshape(1, -1)
