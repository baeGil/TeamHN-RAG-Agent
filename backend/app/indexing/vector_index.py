"""Dense vector index backed by turbovec IdMapIndex (TurboQuant)."""
from pathlib import Path
from typing import Optional

import numpy as np
from turbovec import IdMapIndex


class VectorIndex:
    def __init__(self, bit_width: int = 4, dim: Optional[int] = None) -> None:
        self.bit_width = bit_width
        self._dim = dim
        self._index: Optional[IdMapIndex] = None
        self._count = 0
        if dim is not None:
            self._index = IdMapIndex(dim=dim, bit_width=bit_width)

    @property
    def count(self) -> int:
        return self._count

    def set_count(self, n: int) -> None:
        self._count = int(n)

    @property
    def ready(self) -> bool:
        return self._index is not None and self._count > 0

    def rebuild(self, vectors: np.ndarray, ids: list[int]) -> None:
        """Replace the whole index with the given vectors/ids in one shot."""
        if vectors is None or len(ids) == 0:
            self._index = None
            self._count = 0
            return
        self._dim = int(vectors.shape[1])
        self._index = IdMapIndex(dim=self._dim, bit_width=self.bit_width)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        id_arr = np.asarray(ids, dtype=np.uint64)
        self._index.add_with_ids(vecs, id_arr)
        self._count = len(ids)
        self._index.prepare()

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if vectors.shape[0] == 0:
            return
        if self._index is None:
            self._dim = int(vectors.shape[1])
            self._index = IdMapIndex(dim=self._dim, bit_width=self.bit_width)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        id_arr = np.asarray(ids, dtype=np.uint64)
        self._index.add_with_ids(vecs, id_arr)
        self._count += len(ids)
        self._index.prepare()

    def remove(self, ids: list[int]) -> None:
        if self._index is None:
            return
        for cid in ids:
            for idx in range(10):
                mapped_id = int(cid) * 10 + idx
                try:
                    if self._index.contains(mapped_id):
                        self._index.remove(mapped_id)
                        self._count -= 1
                except Exception:
                    pass
        try:
            self._index.prepare()
        except Exception:
            pass

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.ready:
            return []
        q = np.ascontiguousarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        raw_k = min(k * 5, self._count)
        scores, ids = self._index.search(q, k=raw_k)
        seen = set()
        out: list[tuple[int, float]] = []
        for cid, sc in zip(ids[0].tolist(), scores[0].tolist()):
            cid_int = int(cid) // 10
            if cid_int not in seen:
                seen.add(cid_int)
                out.append((cid_int, float(sc)))
                if len(out) == k:
                    break
        return out

    def save(self, path: Path) -> None:
        if self._index is not None and self._count > 0:
            self._index.write(str(path))

    @classmethod
    def load(cls, path: Path, bit_width: int = 4) -> "VectorIndex":
        idx = cls(bit_width=bit_width)
        loaded = IdMapIndex.load(str(path))
        idx._index = loaded
        try:
            idx._dim = loaded.dim
        except Exception:
            idx._dim = None
        return idx
