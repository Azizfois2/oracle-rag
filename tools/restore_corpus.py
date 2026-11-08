"""Re-ingest the ten sample documents that live in the notebook.

    python tools/restore_corpus.py

Useful after recreating the Oracle container or the data volume. The notebook
stays the single source of truth: this reads its documents cell and writes each
document whole (one chunk per document, exactly as the notebook does).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc

NOTEBOOK = ROOT / "notebook" / "oracle_agentic_rag_hybrid_search.ipynb"


def documents_from_notebook(path: Path) -> list:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "documents = [" not in source or "Document(" not in source:
            continue
        from langchain_core.documents import Document

        scope: dict = {}
        exec(compile(source, "<notebook documents cell>", "exec"), {"Document": Document}, scope)
        documents = scope.get("documents") or []
        if documents:
            return documents
    raise SystemExit(f"could not find the documents cell in {path}")


def main() -> int:
    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    if not engine.ping()["ok"]:
        print("database unreachable - start the container first")
        return 2

    documents = documents_from_notebook(NOTEBOOK)
    print(f"found {len(documents)} documents in the notebook")
    print("loading the embedding model (about 35 s the first time)...")
    engine.warm()

    added = 0
    for document in documents:
        source = str(document.metadata.get("source") or "unknown")
        if engine.source_exists(source):
            print(f"  skip    {source} (already present)")
            continue
        result = engine.ingest(document.page_content, source, chunk_size=1_000_000,
                               metadata=dict(document.metadata))
        added += int(result["chunks"])
        published = result.get("published") or "undated"
        print(f"  ingest  {source} ({result['chunks']} chunk, {published})")

    print("indexes:", "; ".join(engine.reindex()))
    stats = engine.stats()
    print(f"done: {added} chunks added, {stats['rows']} rows from {stats['sources']} sources")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
