"""Turn text into a fixed-length vector without a downloaded model.

The hashing embedder counts tokens into bins. Two notes that reuse the
same words point in a similar direction. Notes that share no words do
not, even if a person would call them related. That limit is useful
here: when a query hits the right note, you can see the shared words.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from vectordb.metrics import l2_normalize

# Tiny function words. They appear in every note, so they would crowd
# out the words that actually distinguish a topic.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "as",
        "by",
        "from",
        "at",
        "into",
        "how",
        "does",
        "do",
        "what",
        "when",
        "where",
        "which",
        "between",
        "then",
        "than",
        "their",
        "its",
        "can",
        "you",
        "your",
        "we",
        "they",
        "about",
        "if",
        "not",
        "but",
        "so",
        "because",
    }
)

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that maps text to one vector of length `dim`.

    A later semantic model can satisfy this with the same method. The
    database only calls `embed`.
    """

    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalized vector for `text`."""


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, with stopwords removed."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class HashingEmbedder:
    """Hash each token into a bin, average the bins, then normalize.

    The same text and seed always produce the same vector. No weights
    are learned. `dim` is the length of every vector this embedder writes.
    """

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            index, sign = self._bucket(token)
            vector[index] += sign
        vector /= np.float32(len(tokens))
        return l2_normalize(vector)

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self.seed}:{token}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "little") % self.dim
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        return bucket, sign
