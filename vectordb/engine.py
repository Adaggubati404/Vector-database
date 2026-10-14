"""The facade that chunks, embeds, stores, and searches.

`VectorDB.add_text` cuts a note into windows, embeds each window, appends
it to the matrix, and links that row into the HNSW graph. `search` embeds
the question the same way, then either scores every row or walks the graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vectordb.chunker import chunk_text
from vectordb.embedder import HashingEmbedder
from vectordb.hnsw import HNSWIndex
from vectordb.store import SearchHit, VectorStore


class VectorDB:
    """A small collection you can add to, query, save, and load."""

    def __init__(
        self,
        dim: int = 256,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedder_seed: int = 0,
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        hnsw_seed: int = 1,
    ) -> None:
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = HashingEmbedder(dim=dim, seed=embedder_seed)
        self.store = VectorStore(dim)
        # Seed 1 gives the bundled sample a few rows on shortcut layers.
        self.index = HNSWIndex(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            seed=hnsw_seed,
        )

    def __len__(self) -> int:
        return len(self.store)

    def add_text(self, text: str, metadata: dict | None = None) -> list[str]:
        """Chunk and insert `text`. Returns the new record ids.

        Metadata is copied onto every chunk. A `chunk` key records the
        window index inside this text. The record id is `source#index`
        when metadata includes `source`.
        """
        base = dict(metadata or {})
        source = str(base.get("source", "doc"))
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        record_ids: list[str] = []
        for chunk_index, chunk in enumerate(chunks):
            meta = dict(base)
            meta["chunk"] = chunk_index
            record_id = f"{source}#{chunk_index}"
            if record_id in self.store:
                record_id = f"{source}#{chunk_index}-{self.store.next_serial}"
            vector = self.embedder.embed(chunk)
            row = self.store.add(record_id, vector, chunk, meta)
            self.index.add(row, self.store.vectors[row])
            record_ids.append(record_id)
        return record_ids

    def add_documents(self, directory: str | Path) -> int:
        """Insert every `*.txt` file in `directory`. Returns the file count.

        Each file is stored with `source` set to the filename and `topic`
        set to the filename stem.
        """
        folder = Path(directory)
        files = sorted(folder.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"no .txt files in {folder}")
        for path in files:
            self.add_text(
                path.read_text(encoding="utf-8"),
                {"source": path.name, "topic": path.stem},
            )
        return len(files)

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hnsw",
        where: dict | None = None,
    ) -> list[SearchHit]:
        """Embed `query` and return the closest chunks.

        `mode="exact"` scores every row. `mode="hnsw"` walks the graph.
        A metadata filter uses the exact scan so the filter cannot hide
        a closer row that the graph walk skipped.
        """
        if mode not in ("exact", "hnsw"):
            raise ValueError("mode must be 'exact' or 'hnsw'")
        query_vector = self.embedder.embed(query)
        if mode == "exact" or where:
            return self.store.search(query_vector, k=k, where=where)
        found = self.index.search(query_vector, k)
        return [self._hit(row, score) for row, score in found]

    def delete(self, record_id: str) -> None:
        """Remove one row and rebuild the graph over the rows that remain."""
        self.store.delete(record_id)
        self.index = HNSWIndex(
            M=self.index.M,
            ef_construction=self.index.ef_construction,
            ef_search=self.index.ef_search,
            seed=self.index.seed,
        )
        for row in range(len(self.store)):
            self.index.add(row, self.store.vectors[row])

    def stats(self) -> dict:
        layers = 0 if self.index.max_level < 0 else self.index.max_level + 1
        return {
            "records": len(self.store),
            "dimension": self.dim,
            "layers": layers,
            "max_level": self.index.max_level,
            "M": self.index.M,
            "ef_search": self.index.ef_search,
        }

    def save(self, directory: str | Path) -> None:
        """Write `vectors.npy` and `meta.json` into `directory`."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.store.vectors)
        payload = {
            "dim": self.dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedder": {
                "type": "hashing",
                "dim": self.embedder.dim,
                "seed": self.embedder.seed,
            },
            "ids": self.store.ids,
            "texts": self.store.texts,
            "metadata": self.store.metadata,
            "next_id": self.store._next,
            "hnsw": self.index.to_dict(),
        }
        (folder / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> VectorDB:
        """Rebuild a database from a directory written by `save`."""
        folder = Path(directory)
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"no database at {folder}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy")
        database = cls(
            dim=payload["dim"],
            chunk_size=payload["chunk_size"],
            chunk_overlap=payload["chunk_overlap"],
            embedder_seed=payload["embedder"]["seed"],
            M=payload["hnsw"]["M"],
            ef_construction=payload["hnsw"]["ef_construction"],
            ef_search=payload["hnsw"]["ef_search"],
            hnsw_seed=payload["hnsw"]["seed"],
        )
        database.store.ids = list(payload["ids"])
        database.store.texts = list(payload["texts"])
        database.store.metadata = [dict(item) for item in payload["metadata"]]
        database.store.vectors = np.asarray(vectors, dtype=np.float32)
        database.store._next = int(payload["next_id"])
        database.store._id_to_row = {
            record_id: index for index, record_id in enumerate(database.store.ids)
        }
        database.index = HNSWIndex.from_dict(payload["hnsw"])
        for row, vector in enumerate(database.store.vectors):
            database.index.vectors[row] = np.asarray(vector, dtype=np.float32)
        return database

    def _hit(self, row: int, score: float) -> SearchHit:
        return SearchHit(
            id=self.store.ids[row],
            score=score,
            text=self.store.texts[row],
            metadata=dict(self.store.metadata[row]),
            row=row,
        )
