"""Command line for the sample collection.

    python -m vectordb seed
    python -m vectordb query "how does approximate search work" --k 3
    python -m vectordb stats
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# The debugger runs this file as a script. That puts vectordb/ on sys.path
# instead of the project root, so "import vectordb" cannot see the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vectordb.engine import VectorDB


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vectordb",
        description="A vector database small enough to read.",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="directory for vectors.npy and meta.json (default: ./data)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="load sample_docs into the database")
    seed.add_argument(
        "--sample",
        default=None,
        help="directory of .txt notes (default: ./sample_docs next to this package)",
    )

    add = subparsers.add_parser("add", help="chunk and insert one or more text files")
    add.add_argument("paths", nargs="+", help="text files to insert")

    query = subparsers.add_parser("query", help="search the saved collection")
    query.add_argument("text", help="the question to embed and search")
    query.add_argument("--k", type=int, default=5, help="how many chunks to return")
    query.add_argument(
        "--mode",
        choices=("exact", "hnsw"),
        default="hnsw",
        help="exact scores every row; hnsw walks the graph",
    )

    subparsers.add_parser("stats", help="print how many rows and layers are saved")

    args = parser.parse_args(argv)
    data = Path(args.data)
    if args.command == "seed":
        sample = Path(args.sample) if args.sample else default_sample_dir()
        database = VectorDB()
        files = database.add_documents(sample)
        database.save(data)
        print(f"Seeded {files} files ({len(database)} chunks) into {data.resolve()}")
        return
    if args.command == "add":
        database = VectorDB.load(data) if (data / "meta.json").is_file() else VectorDB()
        for raw_path in args.paths:
            path = Path(raw_path)
            if not path.is_file():
                raise SystemExit(f"File not found: {path}")
            record_ids = database.add_text(
                path.read_text(encoding="utf-8"),
                {"source": path.name, "topic": path.stem},
            )
            print(f"Added {path.name}: {len(record_ids)} chunks")
        database.save(data)
        print(f"Saved {len(database)} chunks to {data.resolve()}")
        return
    if args.command == "query":
        database = open_database(data)
        hits = database.search(args.text, k=args.k, mode=args.mode)
        if not hits:
            print("No matches.")
            return
        print(f"mode={args.mode}  k={args.k}")
        for rank, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "")
            print(f"{rank}. score={hit.score:.4f}  source={source}  id={hit.id}")
            shown = hit.text.strip()
            print(textwrap.fill(shown, width=88, initial_indent="   ", subsequent_indent="   "))
            print()
        return
    if args.command == "stats":
        database = open_database(data)
        info = database.stats()
        print(f"records: {info['records']}")
        print(f"dimension: {info['dimension']}")
        print(f"layers: {info['layers']}")
        print(f"max_level: {info['max_level']}")
        print(f"M: {info['M']}")
        print(f"ef_search: {info['ef_search']}")
        print(f"data: {data.resolve()}")
        return
    raise SystemExit(f"unknown command: {args.command}")


def default_sample_dir() -> Path:
    """`sample_docs/` at the project root, beside the `vectordb` package."""
    return Path(__file__).resolve().parent.parent / "sample_docs"


def open_database(data: Path) -> VectorDB:
    if not (data / "meta.json").is_file():
        raise SystemExit(f"No database at {data}. Run: python -m vectordb seed")
    return VectorDB.load(data)


if __name__ == "__main__":
    main()
