"""Restore a corpus written by export_corpus.py.

    python tools/import_corpus.py data\\corpus-20261105-120000.json
    python tools/import_corpus.py backup.json --replace

Without --replace, sources that already exist are skipped, so running the same
import twice cannot duplicate anything. With --replace, an existing source is
deleted and written again from the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a corpus exported by export_corpus.py.")
    parser.add_argument("input", help="the JSON file written by export_corpus.py")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace sources that already exist (default: skip them)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"could not read {path}: {exc}")
        return 2
    except json.JSONDecodeError as exc:
        print(f"{path} is not valid JSON: {exc}")
        return 2

    if payload.get("format") != "oracle-rag-corpus":
        print(f"{path} is not a corpus export (missing format marker)")
        return 2
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        print(f"{path} contains no documents")
        return 2

    settings = rc.Settings.from_env()
    exported_model = str(payload.get("embedding_model") or "")
    if exported_model and exported_model != settings.embedding_model:
        print(f"warning: export used {exported_model}")
        print(f"         this install uses {settings.embedding_model}")

    engine = rc.RagEngine(settings)
    try:
        print(f"read {len(documents)} chunks from {path}")
        print("loading the embedding model (about 35 s the first time)...")
        engine.warm()
        result = engine.import_documents(documents, replace=args.replace)
        print(
            f"imported {result['chunks']} chunks from {result['sources']} sources"
            + (f", replaced {result['replaced']}" if result["replaced"] else "")
            + (f", skipped {result['skipped']}" if result["skipped"] else "")
        )
        if result["skipped"] and not args.replace:
            print("existing sources were skipped; pass --replace to overwrite them")
        stats = engine.stats()
        print(f"corpus now: {stats['rows']} rows from {stats['sources']} sources")
        return 0
    except rc.RagError as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 2
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
