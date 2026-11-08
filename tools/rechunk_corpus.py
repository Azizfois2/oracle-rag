"""Re-ingest the ten sample articles as overlapping chunks.

    python tools/rechunk_corpus.py [--size 900] [--overlap 120]

The notebook keeps the ten articles whole, which is right for teaching. This
tool keeps the same text but splits each article into smaller retrieval units,
so the benchmark has more than ten candidates to choose from. It refuses to run
if chunking would break an expected answer in quality/questions.json.

The corpus changes, so an old evaluation is no longer comparable. The Quality
tab records the corpus fingerprint and rejects a stale run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc
from quality.run_eval import load_questions, needles_for, question_type
from tools.restore_corpus import documents_from_notebook

NOTEBOOK = ROOT / "notebook" / "oracle_agentic_rag_hybrid_search.ipynb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    if args.size < 200:
        print("size must be at least 200 characters")
        return 1

    documents = documents_from_notebook(NOTEBOOK)
    payload: list[dict] = []
    for document in documents:
        source = str(document.metadata.get("source") or "unknown")
        published = rc.published_from_text(document.page_content)
        parts = rc.chunk_text(document.page_content, args.size, args.overlap)
        for index, part in enumerate(parts, start=1):
            payload.append({
                "source": source,
                "text": part,
                "metadata": {"chunk": index, "of": len(parts),
                             "published": published},
            })

    haystack = "\n".join(
        item["source"] + "\n" + item["text"] for item in payload
    ).lower()
    missing = [
        needle
        for question in load_questions()
        if question_type(question) != "unanswerable"
        for needle in needles_for(question)
        if needle.lower() not in haystack
    ]
    if missing:
        print("chunking would lose expected answers:")
        for needle in dict.fromkeys(missing):
            print(f"  - {needle}")
        print("try a larger --size or --overlap")
        return 1

    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    if not engine.ping()["ok"]:
        print("database unreachable")
        return 2
    if not engine.table_exists():
        print(f"table {settings.table} missing")
        return 2

    result = engine.import_documents(payload, replace=True)
    fingerprint = engine.corpus_fingerprint()
    rows = engine.stats().get("rows", 0)
    engine.close()

    print(f"replaced {result['sources']} sources with {result['chunks']} chunks")
    print(f"table now holds {rows} chunks, fingerprint {fingerprint}")
    print("old evaluation runs are stale; re-run Quality -> Run evaluation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
