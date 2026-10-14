"""See the geometry, then query the sample notes.

Run from the project root:

    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vectordb.engine import VectorDB  # noqa: E402
from vectordb.metrics import cosine  # noqa: E402
from vectordb.store import VectorStore  # noqa: E402

QUERIES = [
    "how does approximate search work",
    "cosine similarity between normalized vectors",
    "split long documents into overlapping chunks",
    "how to bake sourdough bread",
]


def demo_points() -> None:
    print("=== Four arrows on a plane ===")
    print("Each point is normalized on insert. The score is the cosine.")
    print()
    labeled = [
        ("east", np.array([1.0, 0.0], dtype=np.float32)),
        ("north", np.array([0.0, 1.0], dtype=np.float32)),
        ("northeast", np.array([1.0, 1.0], dtype=np.float32)),
        ("west", np.array([-1.0, 0.2], dtype=np.float32)),
    ]
    query = np.array([0.9, 0.1], dtype=np.float32)
    store = VectorStore(dim=2)
    for name, vector in labeled:
        store.add(name, vector, name, {"label": name})

    print(f"query: [{query[0]:.1f}, {query[1]:.1f}]  (the store normalizes this before scoring)")
    hits = store.search(query, k=len(store))
    for hit in hits:
        stored = store.vectors[hit.row]
        print(
            f"  {hit.id:<10} cosine={hit.score:.4f}  "
            f"stored={stored[0]:+.3f}, {stored[1]:+.3f}"
        )
    by_hand = cosine(query, labeled[0][1])
    print()
    print(f"Check by hand: cosine(query, east) = {by_hand:.4f}")
    print(f"Nearest label: {hits[0].id}")
    print()


def demo_notes() -> None:
    print("=== Sample notes ===")
    database = VectorDB()
    files = database.add_documents(ROOT / "sample_docs")
    info = database.stats()
    print(f"Loaded {files} files as {info['records']} chunks, dimension {info['dimension']}.")
    print()
    for question in QUERIES:
        print(f"Query: {question}")
        exact = database.search(question, k=3, mode="exact")
        approximate = database.search(question, k=1, mode="hnsw")
        for rank, hit in enumerate(exact, start=1):
            preview = " ".join(hit.text.split())
            if len(preview) > 140:
                preview = preview[:137] + "..."
            print(f"  {rank}. {hit.score:.4f}  {hit.metadata['source']}")
            print(f"     {preview}")
        hnsw_source = approximate[0].metadata["source"] if approximate else "(none)"
        print(f"  hnsw top hit: {hnsw_source}")
        print()


def main() -> None:
    demo_points()
    demo_notes()


if __name__ == "__main__":
    main()
