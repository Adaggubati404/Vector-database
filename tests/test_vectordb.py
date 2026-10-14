"""Checks for the teaching vector database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vectordb.chunker import chunk_text
from vectordb.embedder import HashingEmbedder
from vectordb.engine import VectorDB
from vectordb.hnsw import HNSWIndex
from vectordb.metrics import cosine, dot, l2_distance, l2_normalize
from vectordb.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_docs"


def test_metric_identities():
    east = np.array([1.0, 0.0], dtype=np.float32)
    north = np.array([0.0, 1.0], dtype=np.float32)
    west = np.array([-2.0, 0.0], dtype=np.float32)
    assert cosine(east, east) == pytest.approx(1.0)
    assert cosine(east, north) == pytest.approx(0.0)
    assert cosine(east, west) == pytest.approx(-1.0)
    assert dot(l2_normalize(east), l2_normalize(east)) == pytest.approx(1.0)
    assert l2_distance(np.zeros(2), np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert np.allclose(l2_normalize(np.zeros(3)), np.zeros(3))


def test_chunk_overlap():
    chunks = chunk_text("a" * 600, size=500, overlap=50)
    assert chunks[0] == "a" * 500
    assert chunks[1] == "a" * 150
    assert chunks[0][-50:] == chunks[1][:50]
    assert chunk_text("  hello  ") == ["hello"]
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)


def test_embedder_is_deterministic_and_groups_shared_words():
    embedder = HashingEmbedder(dim=256, seed=0)
    bread = embedder.embed("sourdough bread bake starter loaf")
    again = embedder.embed("sourdough bread bake starter loaf")
    hike = embedder.embed("hiking trail boots elevation ridge")
    assert np.allclose(bread, again)
    assert np.linalg.norm(bread) == pytest.approx(1.0)
    related = embedder.embed("bake sourdough bread loaf")
    assert cosine(bread, related) > cosine(bread, hike)


def test_exact_topk_on_known_points():
    store = VectorStore(dim=2)
    store.add("east", np.array([1.0, 0.0]), "east", {})
    store.add("north", np.array([0.0, 1.0]), "north", {})
    store.add("northeast", np.array([1.0, 1.0]), "northeast", {})
    hits = store.search(np.array([0.9, 0.1]), k=2)
    assert [hit.id for hit in hits] == ["east", "northeast"]
    assert hits[0].score > hits[1].score


def test_metadata_filter_drops_closer_rows():
    store = VectorStore(dim=2)
    store.add("east", np.array([1.0, 0.0]), "east", {"topic": "east"})
    store.add("north", np.array([0.0, 1.0]), "north", {"topic": "north"})
    hits = store.search(np.array([0.2, 0.9]), k=2, where={"topic": "east"})
    assert [hit.id for hit in hits] == ["east"]


def test_get_and_delete():
    database = VectorDB(dim=32, chunk_size=500, chunk_overlap=50)
    database.add_text("sourdough bread bake loaf", {"source": "sourdough.txt", "topic": "sourdough"})
    database.add_text("hiking trail boots elevation", {"source": "hiking.txt", "topic": "hiking"})
    record = database.store.get("sourdough.txt#0")
    assert record is not None
    assert record.metadata["topic"] == "sourdough"
    database.delete("sourdough.txt#0")
    assert database.store.get("sourdough.txt#0") is None
    hits = database.search("sourdough bread bake", k=1, mode="exact")
    assert hits[0].metadata["source"] == "hiking.txt"


def test_save_and_load_round_trip(tmp_path: Path):
    database = VectorDB()
    database.add_text(
        "Approximate search walks a graph. " * 20,
        {"source": "hnsw.txt", "topic": "hnsw"},
    )
    database.add_text(
        "Bake sourdough bread with a starter. " * 20,
        {"source": "sourdough.txt", "topic": "sourdough"},
    )
    before = database.search("how does approximate search work", k=1, mode="exact")
    database.save(tmp_path)
    assert (tmp_path / "meta.json").is_file()
    assert (tmp_path / "vectors.npy").is_file()

    loaded = VectorDB.load(tmp_path)
    after = loaded.search("how does approximate search work", k=1, mode="exact")
    assert after[0].id == before[0].id
    assert after[0].score == pytest.approx(before[0].score)
    assert np.allclose(loaded.store.vectors, database.store.vectors)
    hnsw_after = loaded.search("how does approximate search work", k=1, mode="hnsw")
    assert hnsw_after[0].metadata["source"] == "hnsw.txt"


def test_hnsw_recall_against_exact():
    rng = np.random.default_rng(0)
    count, dim, k = 300, 16, 10
    raw = rng.normal(size=(count, dim)).astype(np.float32)
    vectors = np.stack([l2_normalize(vector) for vector in raw])
    index = HNSWIndex(M=16, ef_construction=64, ef_search=64, seed=0)
    for row, vector in enumerate(vectors):
        index.add(row, vector)

    queries = rng.normal(size=(20, dim)).astype(np.float32)
    recalls = []
    for query in queries:
        query = l2_normalize(query)
        exact = np.argsort(-(vectors @ query))[:k]
        approximate = [row for row, _score in index.search(query, k)]
        recalls.append(len(set(exact.tolist()) & set(approximate)) / k)
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.9, mean_recall


def test_sourdough_query_ranks_the_bread_note_first():
    database = VectorDB()
    database.add_documents(SAMPLE)
    hits = database.search("how to bake sourdough bread", k=3, mode="exact")
    assert hits[0].metadata["source"] == "sourdough.txt"
    approximate = database.search("how to bake sourdough bread", k=3, mode="hnsw")
    assert approximate[0].metadata["source"] == "sourdough.txt"


@pytest.mark.parametrize(
    ("question", "source"),
    [
        ("how does approximate search work", "hnsw.txt"),
        ("cosine similarity between normalized vectors", "cosine.txt"),
        ("split long documents into overlapping chunks", "chunking.txt"),
        ("how to bake sourdough bread", "sourdough.txt"),
    ],
)
def test_sample_queries_hit_the_matching_note(question: str, source: str):
    database = VectorDB()
    database.add_documents(SAMPLE)
    hits = database.search(question, k=3, mode="exact")
    assert hits[0].metadata["source"] == source
