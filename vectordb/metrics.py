"""Similarity math for stored vectors.

Every vector is L2-normalized before it is stored. After that, cosine
similarity is the dot product, so a query can score the whole collection
with one matrix multiplication.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to length 1. A zero vector stays zero."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / np.float32(norm)


def dot(left: np.ndarray, right: np.ndarray) -> float:
    """Dot product. For normalized vectors this is cosine similarity."""
    return float(np.dot(left, right))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Zero vectors compare as 0."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Straight-line distance between two vectors."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return float(np.linalg.norm(left - right))
