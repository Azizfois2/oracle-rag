"""Run the answer-quality gate against live Gemini answers.

    python quality/check_answers.py --dry-run     # list cases, zero calls
    python quality/check_answers.py               # 5 questions, about 10-20 calls
    python quality/check_answers.py --all         # every case, watch your quota
    python quality/check_answers.py --resume      # ask only the unanswered cases
    python quality/check_answers.py --all --resume

The free tier allows about 20 requests per day per model, and each question can
use more than one request, so the default is deliberately small. Progress is
saved to data/answers-last.json after every answer, so a quota stop keeps what
was already scored. `--resume` reads that file, skips questions it already
answered, and continues with the rest on the next quota day.

Every run records the corpus fingerprint, so a stale run is easy to spot. No key
is ever written to that file.
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
from quality.answer_eval import (load_cases, pending_cases, score_case,
                                 summarize_rows)

OUT = rc.DATA_DIR / "answers-last.json"


def read_progress(path: Path) -> dict:
    """Read a saved run, or an empty dict when there is none."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def saved_rows(payload: dict) -> list[dict]:
    """Case scores from a saved run, in the older or newer layout."""
    rows = payload.get("cases")
    if not isinstance(rows, list):
        summary = payload.get("summary")
        rows = summary.get("cases") if isinstance(summary, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def payload_for(engine: rc.RagEngine, corpus: list[str], rows: list[dict],
                model: str) -> dict:
    summary = summarize_rows(rows)
    return {
        "version": 1,
        "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus_fingerprint": engine.corpus_fingerprint(),
        "corpus_rows": engine.stats().get("rows", 0),
        "corpus_sources": corpus,
        "model": model,
        "summary": summary,
    }


def save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("\n" + "-" * 60)
    print(f"  {summary['passed']}/{summary['total']} cases passed "
          f"({summary['score'] * 100:.0f}%)")
    for name, bucket in summary["checks"].items():
        print(f"  {name:10s} {bucket['passed']}/{bucket['total']}")
    for kind, bucket in summary["by_type"].items():
        print(f"  {kind:12s} {bucket['passed']}/{bucket['total']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5,
                        help="how many unanswered cases to ask (default 5)")
    parser.add_argument("--all", action="store_true",
                        help="ask every unanswered case, ignoring --limit")
    parser.add_argument("--resume", action="store_true",
                        help="skip cases already scored in the saved run")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the cases and exit without calling a model")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached answers, so every case calls the model")
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"where to save the run (default {OUT.name})")
    args = parser.parse_args()

    cases = load_cases()
    if not cases:
        print("no answer cases found")
        return 1

    if args.dry_run:
        print(f"{len(cases)} answer cases, zero model calls:")
        for index, case in enumerate(cases, start=1):
            kind = case.get("type", "single")
            print(f"  {index:2d}. [{kind:12s}] {case.get('question')}")
        if args.resume:
            rows = saved_rows(read_progress(args.out))
            print(f"\n  {len(rows)} of {len(cases)} already answered in {args.out.name}")
        return 0

    settings = rc.Settings.from_env()
    if not settings.has_llm:
        print("no GEMINI_API_KEY configured, so live answers cannot be checked")
        return 2

    engine, agent = rc.build(settings)
    try:
        if not engine.ping()["ok"]:
            print("database unreachable - start the container first")
            return 2
        corpus = [row["source"] for row in engine.sources()]
        if not corpus:
            print("the corpus is empty")
            return 2

        rows: list[dict] = []
        if args.resume:
            payload = read_progress(args.out)
            if args.out.exists() and not payload:
                print(f"could not read {args.out.name}; starting a fresh run")
            saved = saved_rows(payload)
            if saved:
                saved_fp = str(payload.get("corpus_fingerprint") or "")
                current_fp = engine.corpus_fingerprint()
                if saved_fp != current_fp:
                    print(f"the saved run in {args.out.name} used a different corpus")
                    print(f"  saved:   {saved_fp or 'unknown'}")
                    print(f"  current: {current_fp}")
                    print("delete that file, or run without --resume, to start fresh")
                    return 2
                known = {str(case.get("question") or "") for case in cases}
                rows = [row for row in saved
                        if str(row.get("question") or "") in known]

        remaining = pending_cases(cases, rows)
        if not remaining:
            summary = summarize_rows(rows)
            print(f"all {len(cases)} cases are already answered in {args.out.name}")
            print_summary(summary)
            print(f"  saved to {args.out}")
            return 0 if summary["passed"] == summary["total"] else 1

        selected = remaining if args.all else remaining[: max(1, args.limit)]
        if rows:
            print(f"resuming: {len(rows)} of {len(cases)} cases already answered")
        print(f"asking {len(selected)} of {len(cases)} cases now, "
              f"{len(corpus)} sources in the corpus")
        print("free tier is about 20 requests per day per model; stop with Ctrl+C")

        stopped = ""
        for index, case in enumerate(selected, start=1):
            question = str(case.get("question") or "")
            print(f"\n[{index}/{len(selected)}] {question}")
            try:
                answer = agent.ask(question, use_cache=not args.no_cache)
            except rc.QuotaExhausted as exc:
                stopped = f"quota stopped the run: {exc}"
                print(f"  {stopped}")
                break
            except rc.RagError as exc:
                stopped = f"the run stopped: {exc}"
                print(f"  {stopped}")
                break

            result = score_case(case, answer, corpus)
            rows.append(result.as_dict())
            save_progress(args.out,
                          payload_for(engine, corpus, rows, answer.model))
            mark = "PASS" if result.passed else "FAIL"
            print(f"  {mark}  {len(answer.text)} chars, "
                  f"{len(result.sources)} sources, {answer.calls} calls")
            for check in result.checks:
                if not check.passed:
                    print(f"        {check.name}: {check.detail}")

        if not rows:
            print("\nno answers completed, nothing to save")
            return 1

        summary = summarize_rows(rows)
        print_summary(summary)
        answered = len(rows)
        if answered < len(cases):
            print(f"  {answered}/{len(cases)} cases answered; "
                  f"run again with --resume for the remaining {len(cases) - answered}")
        print(f"  saved to {args.out}")
        if answered < len(cases):
            return 1
        return 0 if summary["passed"] == summary["total"] else 1
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
