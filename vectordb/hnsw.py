"""A small Hierarchical Navigable Small World index.

The index is a stack of graphs over row ids. Upper layers are sparse
shortcuts. The bottom layer contains every row. Search starts at the
top, walks greedily toward the query, then drops a layer and looks at
a wider neighborhood.

This is the idea production libraries use. It skips the extra neighbor
selection heuristic, product quantization, and deletion compaction.
Deleting a row is handled by rebuilding the graph from the rows that
remain.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np

from vectordb.metrics import l2_normalize


class HNSWIndex:
    """Approximate nearest-neighbor graph keyed by store row id.

    `M` is the maximum number of links a node keeps on each layer.
    `ef_construction` is how wide the search is while inserting.
    `ef_search` is how wide the search is while querying. A larger beam
    visits more nodes and usually agrees more often with exact search.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        seed: int = 0,
    ) -> None:
        if M < 2:
            raise ValueError("M must be at least 2")
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("ef_construction and ef_search must be positive")
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        # Geometric level distribution: most nodes land on layer 0.
        self._level_scale = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self.entry_id: int | None = None
        self.max_level = -1
        self.layers: list[dict[int, set[int]]] = []
        self.vectors: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.vectors)

    def add(self, node_id: int, vector: np.ndarray) -> None:
        """Insert one normalized vector.

        Steps match the usual HNSW insert:
        1. Sample a level. Most nodes stay on layer 0.
        2. Greedy-search from the entry point down to that level.
        3. On each layer from there down to 0, connect to M neighbors.
        4. If a neighbor now has more than M links, keep the M closest.
        """
        if node_id in self.vectors:
            raise ValueError(f"duplicate node id: {node_id}")
        stored = l2_normalize(np.array(vector, dtype=np.float32, copy=True))
        self.vectors[node_id] = stored
        level = self._sample_level()

        while len(self.layers) <= level:
            self.layers.append({})
        for layer_index in range(level + 1):
            self.layers[layer_index][node_id] = set()

        if self.entry_id is None:
            self.entry_id = node_id
            self.max_level = level
            return

        entry = self.entry_id
        # Upper layers: follow a single closest neighbor down to `level`.
        for layer_index in range(self.max_level, level, -1):
            nearest = self._search_layer(stored, [entry], ef=1, level=layer_index)
            entry = nearest[0][1]

        # From the node's own level downward, collect a wider candidate set.
        for layer_index in range(min(level, self.max_level), -1, -1):
            nearest = self._search_layer(
                stored, [entry], ef=self.ef_construction, level=layer_index
            )
            selected = [other for _, other in nearest[: self.M]]
            for other in selected:
                self.layers[layer_index][node_id].add(other)
                self.layers[layer_index].setdefault(other, set()).add(node_id)
                if len(self.layers[layer_index][other]) > self.M:
                    self._prune(other, layer_index)
            if nearest:
                entry = nearest[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_id = node_id

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k `(row_id, cosine)` pairs, closest first."""
        if self.entry_id is None or k <= 0:
            return []
        query_vector = l2_normalize(np.asarray(query, dtype=np.float32))
        entry = self.entry_id
        for layer_index in range(self.max_level, 0, -1):
            nearest = self._search_layer(query_vector, [entry], ef=1, level=layer_index)
            entry = nearest[0][1]
        beam = max(self.ef_search, k)
        nearest = self._search_layer(query_vector, [entry], ef=beam, level=0)
        return [(node_id, float(1.0 - distance)) for distance, node_id in nearest[:k]]

    def to_dict(self) -> dict:
        """JSON-ready parameters and adjacency lists. Vectors live in the matrix."""
        layers = []
        for layer in self.layers:
            layers.append(
                {str(node_id): sorted(neighbors) for node_id, neighbors in layer.items()}
            )
        return {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "seed": self.seed,
            "entry_id": self.entry_id,
            "max_level": self.max_level,
            "layers": layers,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> HNSWIndex:
        """Restore the graph. The caller fills `vectors` from the matrix."""
        index = cls(
            M=payload["M"],
            ef_construction=payload["ef_construction"],
            ef_search=payload["ef_search"],
            seed=payload["seed"],
        )
        entry = payload["entry_id"]
        index.entry_id = None if entry is None else int(entry)
        index.max_level = int(payload["max_level"])
        index.layers = []
        for layer in payload["layers"]:
            index.layers.append(
                {int(node_id): set(neighbors) for node_id, neighbors in layer.items()}
            )
        return index

    def _sample_level(self) -> int:
        # floor(-ln(U) * mL) with mL = 1 / ln(M). U is kept away from 0.
        draw = max(self._rng.random(), 1e-12)
        return int(math.floor(-math.log(draw) * self._level_scale))

    def _distance(self, left: np.ndarray, right: np.ndarray) -> float:
        """Cosine distance. 0 means the same direction."""
        return float(1.0 - np.dot(left, right))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_ids: list[int],
        ef: int,
        level: int,
    ) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns `(distance, id)` closest first.

        Candidates are a min-heap of distance, so the closest unexplored
        node comes out first. The result set keeps the `ef` closest nodes
        seen so far. The walk stops when every remaining candidate is
        farther than the worst node already kept.
        """
        visited = set(entry_ids)
        candidates: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []
        for entry_id in entry_ids:
            distance = self._distance(query, self.vectors[entry_id])
            heapq.heappush(candidates, (distance, entry_id))
            heapq.heappush(found, (-distance, entry_id))

        while candidates:
            distance, current = heapq.heappop(candidates)
            furthest = -found[0][0]
            if distance > furthest:
                break
            for neighbor in self.layers[level].get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_distance = self._distance(query, self.vectors[neighbor])
                furthest = -found[0][0]
                if neighbor_distance < furthest or len(found) < ef:
                    heapq.heappush(candidates, (neighbor_distance, neighbor))
                    heapq.heappush(found, (-neighbor_distance, neighbor))
                    if len(found) > ef:
                        heapq.heappop(found)

        ranked = sorted((-negative, node_id) for negative, node_id in found)
        return ranked

    def _prune(self, node_id: int, level: int) -> None:
        """Keep the M closest links and drop the rest on both ends."""
        neighbors = list(self.layers[level][node_id])
        neighbors.sort(key=lambda other: self._distance(self.vectors[node_id], self.vectors[other]))
        for other in neighbors[self.M :]:
            self.layers[level][node_id].discard(other)
            if other in self.layers[level]:
                self.layers[level][other].discard(node_id)
