"""Run the answer-quality gate against live Gemini answers.

    python quality/check_answers.py --dry-run     # list cases, zero calls
    python quality/check_answers.py               # 5 questions, about 10-20 calls
    python quality/check_answers.py --all         # every case, watch your quota
    python quality/check_answers.py --all --no-cache

The free tier allows about 20 requests per day per model, and each question can
use more than one request, so the default is deliberately small. Every run is
saved to data/answers-last.json, with the corpus fingerprint, so a stale run is
easy to spot. No key is ever written to that file.
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
from quality.answer_eval import load_cases, score_all, score_case

OUT = rc.DATA_DIR / "answers-last.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5,
                        help="how many cases to ask (default 5)")
    parser.add_argument("--all", action="store_true",
                        help="ask every case, ignoring --limit")
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
        return 0

    settings = rc.Settings.from_env()
    if not settings.has_llm:
        print("no GEMINI_API_KEY configured, so live answers cannot be checked")
        return 2

    engine, agent = rc.build(settings)
    selected = cases if args.all else cases[: max(1, args.limit)]
    answers = []
    completed = []
    try:
        if not engine.ping()["ok"]:
            print("database unreachable - start the container first")
            return 2
        corpus = [row["source"] for row in engine.sources()]
        if not corpus:
            print("the corpus is empty")
            return 2

        print(f"checking {len(selected)} of {len(cases)} cases, "
              f"{len(corpus)} sources in the corpus")
        print("free tier is about 20 requests per day per model; stop with Ctrl+C")
        for index, case in enumerate(selected, start=1):
            question = str(case.get("question") or "")
            print(f"\n[{index}/{len(selected)}] {question}")
            try:
                answer = agent.ask(question, use_cache=not args.no_cache)
            except rc.QuotaExhausted as exc:
                print(f"  quota stopped the run: {exc}")
                break
            except rc.RagError as exc:
                print(f"  failed: {exc}")
                break
            answers.append(answer)
            completed.append(case)
            result = score_case(case, answer, corpus)
            mark = "PASS" if result.passed else "FAIL"
            print(f"  {mark}  {len(answer.text)} chars, "
                  f"{len(result.sources)} sources, {answer.calls} calls")
            for check in result.checks:
                if not check.passed:
                    print(f"        {check.name}: {check.detail}")

        if not answers:
            print("\nno answers completed, nothing to save")
            return 1

        summary = score_all(completed, answers, corpus)
        payload = {
            "version": 1,
            "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "corpus_fingerprint": engine.corpus_fingerprint(),
            "corpus_rows": engine.stats().get("rows", 0),
            "corpus_sources": corpus,
            "model": answers[0].model if answers else "",
            "summary": summary,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        print("\n" + "-" * 60)
        print(f"  {summary['passed']}/{summary['total']} cases passed "
              f"({summary['score'] * 100:.0f}%)")
        for name, bucket in summary["checks"].items():
            print(f"  {name:10s} {bucket['passed']}/{bucket['total']}")
        for kind, bucket in summary["by_type"].items():
            print(f"  {kind:12s} {bucket['passed']}/{bucket['total']}")
        print(f"  saved to {args.out}")
        return 0 if summary["passed"] == summary["total"] else 1
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
