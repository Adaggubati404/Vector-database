# A vector database from scratch

This project is a small vector database written so you can read every part of it. It stores text as numeric vectors, finds the closest ones, and keeps them on disk. NumPy is the only library the core uses. There is no FAISS, Chroma, or hnswlib.

## Run the sample

From this directory:

```bash
pip install -r requirements.txt
python -m vectordb seed
python -m vectordb query "how does approximate search work" --k 3
python -m vectordb query "how to bake sourdough bread" --k 3
python -m vectordb stats
python examples/demo.py
```

`seed` reads the notes in `sample_docs/`, splits them into chunks, turns each chunk into a vector, and writes the collection to `./data`. Running `seed` again replaces that collection. `add` inserts extra `.txt` files into the same directory.

Each result line is a rank, a score, the source file, and the chunk text. The score is cosine similarity. Higher means closer.

Questions to try, and the note that should rank first:

| Question | Top note |
| --- | --- |
| how does approximate search work | `hnsw.txt` |
| cosine similarity between normalized vectors | `cosine.txt` |
| split long documents into overlapping chunks | `chunking.txt` |
| how to bake sourdough bread | `sourdough.txt` |

`sourdough.txt` and `hiking.txt` are there so a question about bread does not land on a note about graphs. The hiking note is the other contrast.

```bash
python -m vectordb query "cosine similarity between normalized vectors" --mode exact --k 3
python -m vectordb query "cosine similarity between normalized vectors" --mode hnsw --k 3
```

`exact` scores every row. `hnsw` walks the graph. On this small collection the two rankings usually agree. `--mode` defaults to `hnsw`.

## What a vector is

A vector is a list of numbers. Here, each chunk of text becomes one vector of 256 floats. The database stores that list next to the original words. A query is converted the same way, and search looks for stored lists that point in a similar direction.

The embedder in `vectordb/embedder.py` is a hashing trick, not a neural model. It lowercases the text, drops tiny function words, and adds each remaining word into a bin chosen by a hash. The bins are averaged and scaled to length 1. The same sentence always produces the same vector, and nothing is downloaded.

Shared words pull two vectors together. Different words do not, even when a person would say the sentences mean the same thing. The sample notes repeat the words their demo questions use, so a hit is something you can check by reading.

## Why cosine becomes a dot product

Cosine similarity is the cosine of the angle between two arrows. It ignores length and compares direction.

On insert, every vector is divided by its length. A normalized vector has length 1. For two normalized vectors, cosine similarity is their dot product:

```text
score = V[row] · query
```

Exact search does that for every row at once. If `V` has shape `(n, dim)` and `q` is the normalized query, the scores are the matrix product `V @ q`. The largest scores are the nearest chunks.

You can see this with four arrows in `examples/demo.py`. The query points almost along the x-axis, so the point labeled `east` ranks first. The printed cosine matches `cosine(query, east)` computed directly.

## From text to a hit

1. `chunk_text` cuts a note into 500-character windows that overlap by 50 characters. A long note becomes several records. A sentence that straddles a cut still appears whole in one window.
2. `HashingEmbedder.embed` turns each window into a unit vector.
3. `VectorStore` appends the vector as a matrix row and keeps the text and metadata (`source`, `topic`, `chunk`).
4. `HNSWIndex` links that row into the graph.
5. A query is embedded, then either the store scores every row or the graph walks toward the query.

A metadata filter such as `{"source": "hnsw.txt"}` uses the exact scan. Rejected rows are scored as negative infinity before the top-k is chosen, so the filter still returns the closest row among the ones it keeps.

## Exact scan and HNSW

The exact scan is the reference answer. It looks at every vector. That is correct, and the work grows in a straight line with the number of rows.

HNSW (Hierarchical Navigable Small World) is a stack of graphs. Upper layers are sparse shortcuts. The bottom layer contains every row. Search enters at the top, steps to the neighbor closest to the query, then drops a layer. On the bottom layer it keeps a wider beam of candidates (`ef_search`, default 64) and returns the best `k`.

Insert samples a level with `floor(-ln(U) / ln(M))`. Most rows live only on layer 0. A new row connects to at most `M` neighbors on each layer it occupies (default `M` is 16). A neighbor that grows past `M` links keeps the `M` closest and drops the rest.

The graph is an approximation. It can miss a true neighbor when the walk takes a poor turn. `tests/test_vectordb.py` checks that, on a few hundred random unit vectors, recall@10 against the exact scan stays at least 0.9.

Open `data/meta.json` after seeding. `hnsw.layers` is one object per layer. Inside a layer, each row id maps to the row ids it links to. The bottom layer lists every row. Layers above it are the shortcuts, and this sample's level draws include a few of those. A smaller draw can land entirely on the bottom layer. `data/vectors.npy` is the float32 matrix.

A window is a raw character span, so a chunk can start or end in the middle of a word. The 50-character overlap keeps the rest of that word in the neighboring chunk.

## Layout

| File | Role |
| --- | --- |
| `vectordb/metrics.py` | cosine, dot product, L2 distance, normalization |
| `vectordb/chunker.py` | overlapping character windows |
| `vectordb/embedder.py` | hashing embedder and the `Embedder` protocol |
| `vectordb/store.py` | matrix, text, metadata, exact top-k |
| `vectordb/hnsw.py` | approximate graph over row ids |
| `vectordb/engine.py` | `VectorDB`: add, search, save, load |
| `vectordb/__main__.py` | `seed`, `add`, `query`, `stats` |
| `sample_docs/` | notes the seed command loads |
| `examples/demo.py` | the 2D arrows, then the sample questions |

`save` writes `meta.json` and `vectors.npy`. `load` rebuilds the store and the graph. The embedder seed and dimension are in the JSON, so a query in a new process uses the same hashing as the rows on disk.

## What a production engine adds

This design stops once the algorithm is visible. A production vector database also keeps a write-ahead log, serves concurrent writers, compresses vectors with product quantization, and spreads shards across machines. Those pieces matter at large scale. They sit around the same core: embed, store, and find nearest neighbors.

## Tests

```bash
pytest
```

The tests cover metric identities, chunk overlap, exact top-k, metadata filters, save/load, HNSW recall, and the sample questions above.
