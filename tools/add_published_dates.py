"""Add published dates to an existing corpus, in place.

    python tools/add_published_dates.py           # dry run, writes nothing
    python tools/add_published_dates.py --apply   # patch metadata and commit

The dates come from the dateline in each article, so the notebook stays the
single source of truth. Nothing is re-embedded: JSON_MERGEPATCH updates the
metadata column only, which keeps row ids, embeddings and the corpus
fingerprint unchanged. Safe to run twice: a row that already has the same date
is counted as unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc
from tools.restore_corpus import documents_from_notebook

NOTEBOOK = ROOT / "notebook" / "oracle_agentic_rag_hybrid_search.ipynb"


def dates_from_notebook(path: Path) -> dict[str, str]:
    """Map source name to ISO date, skipping articles with no dateline."""
    out: dict[str, str] = {}
    for document in documents_from_notebook(path):
        source = str(document.metadata.get("source") or "").strip()
        published = rc.published_from_text(document.page_content)
        if source and published:
            out[source] = published
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the dates; without it this is a dry run")
    args = parser.parse_args()

    wanted = dates_from_notebook(NOTEBOOK)
    if not wanted:
        print("no dated articles found in the notebook")
        return 1

    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    if not engine.ping()["ok"]:
        print("database unreachable - start the container first")
        return 2
    if not engine.table_exists():
        print(f"table {settings.table} is missing - ingest the corpus first")
        return 2

    conn = engine.ensure()
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT JSON_VALUE(metadata, '$.source') AS src, "
            f"COUNT(*) AS chunks, "
            f"COUNT(JSON_VALUE(metadata, '$.published')) AS dated "
            f"FROM {settings.table} "
            f"GROUP BY JSON_VALUE(metadata, '$.source') ORDER BY src"
        )
        rows = cur.fetchall()
        present = {str(src): (int(chunks), int(dated)) for src, chunks, dated in rows}

        missing = [src for src in wanted if src not in present]
        for src in missing:
            print(f"  not in corpus  {src}")
        undated = [src for src in present if src not in wanted]
        for src in undated:
            print(f"  no dateline    {src}")

        changed = 0
        for src in wanted:
            if src not in present:
                continue
            chunks, dated = present[src]
            state = f"{dated}/{chunks} dated"
            print(f"  {src}  ->  {wanted[src]}  ({state})")
            if args.apply:
                cur.execute(
                    f"UPDATE {settings.table} "
                    f"SET metadata = JSON_MERGEPATCH(metadata, :patch) "
                    f"WHERE JSON_VALUE(metadata, '$.source') = :src "
                    f"AND NVL(JSON_VALUE(metadata, '$.published'), 'x') <> :pub",
                    {"patch": f'{{"published":"{wanted[src]}"}}',
                     "src": src, "pub": wanted[src]},
                )
                changed += int(cur.rowcount or 0)

        if args.apply:
            conn.commit()
            cur.execute(
                f"SELECT COUNT(*) FROM {settings.table} "
                f"WHERE JSON_VALUE(metadata, '$.published') IS NOT NULL"
            )
            dated_total = int(cur.fetchone()[0])
            total = engine.stats().get("rows", 0)
            print(f"updated {changed} rows; {dated_total}/{total} chunks now dated")
            lo, hi = engine.published_range()
            print(f"range: {lo or 'none'} to {hi or 'none'}")
        else:
            print("dry run: nothing written. add --apply to save the dates")
    finally:
        cur.close()
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
