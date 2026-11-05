"""Write every chunk in the corpus to one JSON file.

    python tools/export_corpus.py
    python tools/export_corpus.py --out D:\\backups\\rag-corpus.json

The file contains the text and source of every chunk, plus the embedding model
that was in use. Embeddings are not stored: they are recomputed on import, so
the backup stays small and works after a model change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc


def main() -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Export the corpus to one JSON file.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / f"corpus-{stamp}.json"),
        help="destination file (default: data/corpus-<timestamp>.json)",
    )
    args = parser.parse_args()

    engine = rc.RagEngine(rc.Settings.from_env())
    try:
        payload = engine.export_corpus()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(out)

        documents = payload["documents"]
        sources = {item["source"] for item in documents}
        print(f"exported {len(documents)} chunks from {len(sources)} sources")
        print(f"file: {out}")
        print("embeddings are not stored; they are recomputed on import")
        return 0
    except rc.RagError as exc:
        print(f"error: {exc}")
        return 2
    except OSError as exc:
        print(f"could not write the file: {exc}")
        return 2
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
