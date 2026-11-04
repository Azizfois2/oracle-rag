"""hit@k and MRR per retrieval path.

Used by the Quality tab and runnable on its own:

    python quality/run_eval.py [--k 5] [--weight 0.6]

A question counts as a hit when the expected string appears in the source name or
the text of a returned chunk within the first k results. No LLM involved, so a
full run costs zero quota.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_core as rc

METHODS = ("keyword", "vector", "hybrid")
QUESTIONS = Path(__file__).with_name("questions.json")
LAST_RUN = rc.DATA_DIR / "eval-last.json"


def load_questions(path: Path | None = None) -> list[dict[str, str]]:
    return json.loads(Path(path or QUESTIONS).read_text(encoding="utf-8"))


def _rank(hits: list[rc.Hit], needle: str) -> int | None:
    needle = needle.lower()
    for position, hit in enumerate(hits, start=1):
        haystack = f"{hit.source} {hit.text}".lower()
        if needle in haystack:
            return position
    return None


def evaluate(engine: rc.RagEngine, questions: list[dict[str, str]] | None = None,
             k: int = 5, weight: float = 0.6) -> dict:
    questions = questions or load_questions()
    engine.warm()

    rows = []
    started = time.perf_counter()
    for item in questions:
        bundle = engine.search_all(item["question"], k=k, weight=weight)
        row: dict[str, object] = {
            "question": item["question"],
            "expect": item["expect"],
            "why": item.get("why", ""),
        }
        for method in METHODS:
            hits = getattr(bundle, method)
            rank = _rank(hits, item["expect"])
            row[method] = rank
            row[f"{method}_top"] = round(hits[0].score, 3) if hits else 0.0
        rows.append(row)

    summary = {}
    for method in METHODS:
        ranks = [row[method] for row in rows]
        found = [r for r in ranks if isinstance(r, int)]
        summary[method] = {
            "hit": round(len([r for r in found if r <= k]) / len(rows), 3),
            "mrr": round(sum(1.0 / r for r in found) / len(rows), 3),
            "found": len(found),
        }

    return {
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "weight": round(weight, 2),
        "elapsed": round(time.perf_counter() - started, 1),
        "rows": rows,
        "summary": summary,
    }


def save_run(result: dict) -> None:
    try:
        rc.DATA_DIR.mkdir(parents=True, exist_ok=True)
        LAST_RUN.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_last() -> dict | None:
    try:
        return json.loads(LAST_RUN.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--weight", type=float, default=0.6)
    args = parser.parse_args()

    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    if not engine.ping()["ok"]:
        print("database unreachable")
        return 2
    if not engine.table_exists():
        print(f"table {settings.table} missing")
        return 2

    result = evaluate(engine, k=args.k, weight=args.weight)
    save_run(result)

    print(f"\n{'question':<62} {'kw':>4} {'vec':>4} {'hyb':>4}")
    print("-" * 80)
    for row in result["rows"]:
        def cell(value: object) -> str:
            return "-" if value is None else str(value)
        print(f"{row['question'][:60]:<62} {cell(row['keyword']):>4} "
              f"{cell(row['vector']):>4} {cell(row['hybrid']):>4}")

    print("-" * 80)
    for method in METHODS:
        s = result["summary"][method]
        print(f"{method:<8} hit@{args.k} {s['hit']:.2f}   MRR {s['mrr']:.2f}   found {s['found']}/{len(result['rows'])}")
    print(f"\n{result['elapsed']}s, weight {result['weight']}")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
