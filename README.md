# Vector Database from Scratch

A small, readable vector database written in Python with **NumPy as its only core dependency**: no FAISS, Chroma, or hnswlib. It splits text into chunks, turns each chunk into a vector, stores the vectors on disk, and finds the closest ones with either an exact scan or an HNSW graph.

## Why I built it

Vector databases sit underneath most RAG and agentic AI systems. I wanted to understand them from first principles instead of treating them as a black box: what a vector is, why cosine similarity turns into a dot product, how an approximate index like HNSW trades accuracy for speed, and what "persisting a collection" really means. Every part of this project is small enough to read in one sitting.

## Features

- **Chunking:** 500-character windows with a 50-character overlap, so a sentence that straddles a cut still appears whole in one window.
- **Embedding:** a deterministic hashing embedder (256 dimensions). No model download, and the same text always gives the same vector.
- **Exact search:** scores every stored vector with one matrix product (`V @ q`) and returns the top-k.
- **HNSW approximate search:** a Hierarchical Navigable Small World graph with configurable `M` (default 16), `ef_construction` and `ef_search` (default 64).
- **Metadata filters:** equality filters such as `{"source": "hnsw.txt"}`, applied before the top-k is chosen.
- **Persistence:** `save` / `load` to `vectors.npy` (float32 matrix) and `meta.json` (text, metadata, embedder settings and the HNSW graph).
- **Delete:** removes a record and rebuilds the graph from the remaining rows.
- **CLI:** `seed`, `add`, `query` (with `--mode exact|hnsw`), and `stats`.
- **Tests:** metric identities, chunk overlap, exact top-k, filters, get/delete, save/load, HNSW recall, and sample queries.

## How it works

```
text ──► chunk_text ──► HashingEmbedder ──► VectorStore (matrix + text + metadata)
                                              │
                                              └──► HNSWIndex (graph over row ids)

query ──► HashingEmbedder ──► exact scan (V @ q)  or  HNSW graph walk ──► top-k hits
```

1. **Chunk.** `chunk_text` cuts a note into overlapping character windows. A long note becomes several records.
2. **Embed.** `HashingEmbedder.embed` lowercases the text, drops common stopwords, hashes each remaining word into one of 256 bins, and scales the result to length 1. This is a hashing trick, not a neural model: notes that share words point in similar directions, and paraphrases with different words do not.
3. **Store.** `VectorStore` appends the vector as a matrix row and keeps the chunk text and metadata (`source`, `topic`, `chunk`) next to it.
4. **Index.** `HNSWIndex` links the new row into a stack of graphs. Each insert samples a level with `floor(-ln(U) / ln(M))`, so most rows live only on layer 0. A node keeps at most `M` neighbors per layer and drops the farthest when it grows past that.
5. **Search.**
   - *Exact:* because every vector is normalized on insert, cosine similarity is a dot product, so one matrix multiplication scores the whole collection. This is the reference answer.
   - *HNSW:* search enters at the top layer, steps greedily toward the query, drops a layer, and on the bottom layer keeps a wider beam (`ef_search`) before returning the best `k`.
   - A metadata filter always uses the exact scan: rejected rows get a score of negative infinity before the top-k is chosen.

### Exact vs. approximate

The exact scan always finds the true nearest neighbors, but its cost grows linearly with the number of rows. HNSW visits only part of the graph, so it can occasionally miss a true neighbor. `tests/test_vectordb.py` checks that, on a few hundred random unit vectors, HNSW recall@10 against the exact scan stays at least 0.9.

## Getting started

**Requirements:** Python 3 and the packages in `requirements.txt` (`numpy>=1.24`, `pytest>=7.0`).

```bash
pip install -r requirements.txt

python -m vectordb seed                                            # load sample_docs/ into ./data
python -m vectordb query "how does approximate search work" --k 3
python -m vectordb query "how to bake sourdough bread" --k 3
python -m vectordb stats
python examples/demo.py                                            # 2-D geometry demo + sample questions
```

Compare the two search modes:

```bash
python -m vectordb query "cosine similarity between normalized vectors" --mode exact --k 3
python -m vectordb query "cosine similarity between normalized vectors" --mode hnsw --k 3
```

Add your own notes:

```bash
python -m vectordb add path/to/note1.txt path/to/note2.txt
```

Each result line shows the rank, the cosine score (higher is closer), the source file, and the chunk text.

| Question | Note that should rank first |
| --- | --- |
| how does approximate search work | `hnsw.txt` |
| cosine similarity between normalized vectors | `cosine.txt` |
| split long documents into overlapping chunks | `chunking.txt` |
| how to bake sourdough bread | `sourdough.txt` |

`sourdough.txt` and `hiking.txt` are off-topic contrast notes, so a question about bread does not land on a note about graphs.

### Sample output

> **TODO (Ashok):** paste the real output of `python -m vectordb query "how does approximate search work" --k 3` here.

```text
<paste output here>
```

### Run the tests

```bash
pytest
```

> **TODO (Ashok):** paste the `pytest` summary line here after running it.

## Project structure

| Path | Role |
| --- | --- |
| `vectordb/metrics.py` | cosine, dot product, L2 distance, normalization |
| `vectordb/chunker.py` | overlapping character windows |
| `vectordb/embedder.py` | hashing embedder and the `Embedder` protocol |
| `vectordb/store.py` | vector matrix, text, metadata, exact top-k, filters |
| `vectordb/hnsw.py` | HNSW approximate nearest-neighbor graph |
| `vectordb/engine.py` | `VectorDB` facade: add, search, delete, save, load |
| `vectordb/__main__.py` | CLI: `seed`, `add`, `query`, `stats` |
| `sample_docs/` | sample notes loaded by `seed` |
| `examples/demo.py` | 2-D vector geometry demo, then the sample questions |
| `tests/test_vectordb.py` | pytest suite |

After `seed`, open `data/meta.json`: `hnsw.layers` lists one object per layer, mapping each row id to the row ids it links to. `data/vectors.npy` is the float32 matrix.

## What I learned

- A vector is just a list of numbers. Similarity search is geometry: after L2 normalization, cosine similarity is a dot product, and a whole collection can be scored with one matrix multiplication.
- Chunk size and overlap decide what a single vector "means". Smaller windows stay specific, and overlap protects sentences at the boundaries.
- The embedder decides what "similar" means. A hashing embedder matches shared words, not meaning, which shows clearly why real systems use learned embedding models.
- HNSW is a speed-for-accuracy trade: layered graphs, greedy descent and a wider search beam on the bottom layer, measured with recall against an exact baseline.
- Metadata filtering and persistence (saving the embedder settings alongside the vectors) are what turn an algorithm into a usable database.

## Next steps

These are not implemented yet:

- Plug in a learned embedding model behind the existing `Embedder` protocol and compare its results with the hashing embedder.
- Add the HNSW neighbor-selection heuristic and in-place deletion instead of rebuilding the graph.
- Benchmark exact vs. HNSW latency and recall as the collection grows.
- Explore what production engines add: a write-ahead log, concurrent writers, product quantization, and sharding.
- Use it as the retrieval layer for a small RAG or agentic workflow.

## License

MIT. See [LICENSE](LICENSE).
