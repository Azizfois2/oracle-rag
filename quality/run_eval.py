"""hit@k and MRR per retrieval path.

Used by the Quality tab and runnable on its own:

    python quality/run_eval.py [--k 5] [--weight 0.6]

A single-document question counts as a hit when its expected string appears in
the source name or the text of a returned chunk within the first k results.
A cross-document question needs every string in expect_all, possibly in
different chunks. Unanswerable questions are not scored as hits: retrieval
always returns k chunks, so the run compares their vector top score with
answerable questions. Keyword scores are normalised per query and always peak
at 1.0, so they are not a confidence measure. No LLM is involved, so a full run
costs zero quota.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_core as rc

METHODS = ("keyword", "vector", "hybrid")
TYPES = ("single", "cross", "unanswerable")
QUESTIONS = Path(__file__).with_name("questions.json")
LAST_RUN = rc.DATA_DIR / "eval-last.json"
VERSION = 2


def load_questions(path: Path | None = None) -> list[dict[str, str]]:
    return json.loads(Path(path or QUESTIONS).read_text(encoding="utf-8"))


def questions_digest(questions: list[dict]) -> str:
    raw = json.dumps(questions, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def question_type(item: dict) -> str:
    kind = str(item.get("type") or "").strip().lower()
    if kind in TYPES:
        return kind
    if item.get("expect_all"):
        return "cross"
    if not item.get("expect"):
        return "unanswerable"
    return "single"


def needles_for(item: dict) -> list[str]:
    expected = item.get("expect_all") or []
    needles = [str(value) for value in expected if str(value).strip()]
    if needles:
        return needles
    single = str(item.get("expect") or "").strip()
    return [single] if single else []


def _rank(hits: list[rc.Hit], needles: list[str]) -> int | None:
    """The rank at which every needle has been seen, or None.

    For one needle this is its first position. For a cross-document question
    it is the later of the two positions, so hit@k means both pieces of
    evidence were retrieved inside the top k.
    """
    if not needles:
        return None
    worst = 0
    for needle in needles:
        needle = needle.lower()
        found = None
        for position, hit in enumerate(hits, start=1):
            if needle in f"{hit.source} {hit.text}".lower():
                found = position
                break
        if found is None:
            return None
        worst = max(worst, found)
    return worst or None


def _block(rows: list[dict], k: int) -> dict:
    block = {}
    for method in METHODS:
        ranks = [row[method] for row in rows]
        found = [rank for rank in ranks if isinstance(rank, int)]
        block[method] = {
            "hit": round(len([r for r in found if r <= k]) / len(rows), 3) if rows else 0.0,
            "mrr": round(sum(1.0 / r for r in found) / len(rows), 3) if rows else 0.0,
            "found": len(found),
            "n": len(rows),
        }
    return block


def _tops(rows: list[dict], method: str) -> dict:
    """Top-score range for a set of questions.

    Vector scores are cosine similarity and can be compared across queries.
    Keyword scores are normalised so the best match is always 1.0, and hybrid
    scores are min-max fused, so neither is a confidence measure on its own.
    """
    tops = [float(row[f"{method}_top"]) for row in rows]
    if not tops:
        return {"mean_top": 0.0, "min_top": 0.0, "max_top": 0.0}
    return {
        "mean_top": round(sum(tops) / len(tops), 3),
        "min_top": round(min(tops), 3),
        "max_top": round(max(tops), 3),
    }


def evaluate(engine: rc.RagEngine, questions: list[dict] | None = None,
             k: int = 5, weight: float = 0.6) -> dict:
    questions = questions or load_questions()
    engine.warm()

    rows = []
    started = time.perf_counter()
    for item in questions:
        kind = question_type(item)
        needles = needles_for(item)
        bundle = engine.search_all(item["question"], k=k, weight=weight)
        row: dict[str, object] = {
            "question": item["question"],
            "expect": item.get("expect"),
            "expect_all": item.get("expect_all") or [],
            "type": kind,
            "why": item.get("why", ""),
        }
        for method in METHODS:
            hits = getattr(bundle, method)
            row[method] = _rank(hits, needles)
            row[f"{method}_top"] = round(hits[0].score, 3) if hits else 0.0
        rows.append(row)

    answerable = [row for row in rows if row["type"] != "unanswerable"]
    summary = _block(answerable, k)
    by_type = {
        kind: _block([row for row in rows if row["type"] == kind], k)
        for kind in ("single", "cross")
        if any(row["type"] == kind for row in rows)
    }

    unanswerable = [row for row in rows if row["type"] == "unanswerable"]
    unanswerable_summary = {}
    if unanswerable:
        for method in METHODS:
            unanswerable_summary[method] = _tops(unanswerable, method)

    return {
        "version": VERSION,
        "question_count": len(questions),
        "questions_hash": questions_digest(questions),
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "weight": round(weight, 2),
        "elapsed": round(time.perf_counter() - started, 1),
        "rows": rows,
        "summary": summary,
        "by_type": by_type,
        "answerable_tops": {method: _tops(answerable, method) for method in METHODS},
        "unanswerable": unanswerable_summary,
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

    print(f"\n{'question':<56} {'set':<6} {'kw':>4} {'vec':>4} {'hyb':>4}")
    print("-" * 82)
    for row in result["rows"]:
        def cell(value: object) -> str:
            if row["type"] == "unanswerable":
                return "n/a"
            return "-" if value is None else f"#{value}"
        kind = {"single": "one", "cross": "cross", "unanswerable": "none"}[row["type"]]
        print(f"{row['question'][:54]:<56} {kind:<6} {cell(row['keyword']):>4} "
              f"{cell(row['vector']):>4} {cell(row['hybrid']):>4}")

    print("-" * 82)
    blocks = [("all answerable", result["summary"])]
    blocks += [(kind, block) for kind, block in result["by_type"].items()]
    for label, block in blocks:
        for method in METHODS:
            s = block[method]
            print(f"{label:<14} {method:<8} hit@{args.k} {s['hit']:.2f}   "
                  f"MRR {s['mrr']:.2f}   found {s['found']}/{s['n']}")
        print()

    if result["unanswerable"]:
        base = (result.get("answerable_tops") or {}).get("vector") or {}
        s = result["unanswerable"]["vector"]
        print("unanswerable (no hit metric; vector cosine is the absolute signal)")
        print(f"  vector   answerable mean {base.get('mean_top', 0.0):.2f}, "
              f"unanswerable mean {s['mean_top']:.2f} "
              f"(range {s['min_top']:.2f} to {s['max_top']:.2f})")
        print("  keyword and hybrid scores are normalised per query, so a top "
              "value of 1.0 does not mean the answer was found")

    print(f"\n{result['elapsed']}s, weight {result['weight']}, {result['question_count']} questions")
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
