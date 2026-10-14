"""The collection of rows an exact query scans.

Each row is an id, the chunk text, a metadata dict, and one row of a
NumPy matrix. Exact search scores every row. Metadata equality filters
drop rows before the top-k is chosen, so a filtered query is still the
true nearest neighbor among the rows that remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vectordb.metrics import l2_normalize


@dataclass(frozen=True)
class Record:
    """One stored row, including a copy of its vector."""

    id: str
    text: str
    metadata: dict
    vector: np.ndarray
    row: int


@dataclass(frozen=True)
class SearchHit:
    """A ranked row. `score` is cosine similarity, higher is closer."""

    id: str
    score: float
    text: str
    metadata: dict
    row: int


class VectorStore:
    """In-memory matrix plus the text that belongs to each row."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._id_to_row: dict[str, int] = {}
        self._next = 0

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._id_to_row

    @property
    def next_serial(self) -> int:
        """Monotonic counter used to keep record ids unique after a collision."""
        return self._next

    def add(self, record_id: str, vector: np.ndarray, text: str, metadata: dict | None = None) -> int:
        """Append a normalized row. Returns the row index."""
        if record_id in self._id_to_row:
            raise ValueError(f"duplicate id: {record_id}")
        stored = l2_normalize(np.asarray(vector, dtype=np.float32))
        if stored.shape != (self.dim,):
            raise ValueError(f"expected a vector of length {self.dim}, got {stored.shape}")
        row = len(self.ids)
        self.ids.append(record_id)
        self.texts.append(text)
        self.metadata.append(dict(metadata or {}))
        self.vectors = np.concatenate([self.vectors, stored.reshape(1, -1)], axis=0)
        self._id_to_row[record_id] = row
        self._next += 1
        return row

    def get(self, record_id: str) -> Record | None:
        row = self._id_to_row.get(record_id)
        if row is None:
            return None
        return Record(
            id=record_id,
            text=self.texts[row],
            metadata=dict(self.metadata[row]),
            vector=self.vectors[row].copy(),
            row=row,
        )

    def delete(self, record_id: str) -> None:
        """Remove a row and compact the matrix. Later row indexes shift."""
        row = self._id_to_row.get(record_id)
        if row is None:
            raise KeyError(record_id)
        del self.ids[row]
        del self.texts[row]
        del self.metadata[row]
        self.vectors = np.delete(self.vectors, row, axis=0)
        self._id_to_row = {item_id: index for index, item_id in enumerate(self.ids)}

    def search(self, query: np.ndarray, k: int = 5, where: dict | None = None) -> list[SearchHit]:
        """Score every row and return the top k.

        `where` keeps rows whose metadata matches every key. Comparison
        is equality. Rows that fail are scored as negative infinity so
        they cannot enter the top k.
        """
        if k <= 0 or len(self) == 0:
            return []
        query_vector = l2_normalize(np.asarray(query, dtype=np.float32))
        if query_vector.shape != (self.dim,):
            raise ValueError(f"expected a vector of length {self.dim}, got {query_vector.shape}")

        scores = self.vectors @ query_vector
        if where:
            for index, meta in enumerate(self.metadata):
                if any(meta.get(key) != value for key, value in where.items()):
                    scores[index] = -np.inf

        allowed = int(np.isfinite(scores).sum())
        if allowed == 0:
            return []
        k = min(k, allowed)
        # The largest k scores. argpartition avoids sorting rows we will drop.
        chosen = np.argpartition(scores, -k)[-k:]
        order = chosen[np.argsort(scores[chosen])[::-1]]

        hits: list[SearchHit] = []
        for row in order:
            score = float(scores[row])
            if not np.isfinite(score):
                continue
            hits.append(
                SearchHit(
                    id=self.ids[row],
                    score=score,
                    text=self.texts[row],
                    metadata=dict(self.metadata[row]),
                    row=int(row),
                )
            )
        return hits
