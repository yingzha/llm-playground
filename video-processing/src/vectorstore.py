"""Tiny vector store for Arm B retrieval.

Uses FAISS (IndexFlatIP over L2-normalized vectors = cosine) when available,
otherwise falls back to a numpy brute-force search. Metadata is kept parallel to
the vectors and persisted alongside the index.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.metas: list[dict] = []
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
            self._vecs = None
        else:
            self.index = None
            self._vecs = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if _HAS_FAISS:
            self.index.add(vectors)
        else:
            self._vecs = np.vstack([self._vecs, vectors])
        self.metas.extend(metadatas)

    def search(self, qvec: np.ndarray, k: int):
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        k = min(k, len(self.metas)) or 1
        if _HAS_FAISS:
            scores, idxs = self.index.search(q, k)
            idxs, scores = idxs[0], scores[0]
        else:
            sims = (self._vecs @ q[0])
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]
        return [(float(scores[i]), self.metas[idxs[i]])
                for i in range(len(idxs)) if idxs[i] >= 0]

    # ---- persistence ----
    def persist(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"dim": self.dim, "metas": self.metas, "has_faiss": _HAS_FAISS}
        path.with_suffix(".json").write_text(json.dumps(meta))
        if _HAS_FAISS:
            faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        else:
            with open(path.with_suffix(".npy"), "wb") as f:
                pickle.dump(self._vecs, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        store = cls(meta["dim"])
        store.metas = meta["metas"]
        if _HAS_FAISS and path.with_suffix(".faiss").exists():
            store.index = faiss.read_index(str(path.with_suffix(".faiss")))
        elif path.with_suffix(".npy").exists():
            with open(path.with_suffix(".npy"), "rb") as f:
                store._vecs = pickle.load(f)
        return store

    @staticmethod
    def exists(path: str | Path) -> bool:
        path = Path(path)
        return path.with_suffix(".json").exists() and (
            path.with_suffix(".faiss").exists() or path.with_suffix(".npy").exists()
        )
