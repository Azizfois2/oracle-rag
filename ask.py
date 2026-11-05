"""Ask the corpus from a terminal.

    python ask.py "Which company raised a Series B?"
    python ask.py --local "hybrid search"
    python ask.py --json "Who partnered with NVIDIA?" > answer.json

Local mode never calls a model and spends no quota. It prints the top hybrid
hits with snippets instead of a written answer.

Exit codes
  0  success
  2  bad input or database problem
  3  no usable model (missing key or quota spent) - retry with --local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_core as rc

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass


def print_local(answer: rc.Answer, k: int) -> None:
    print("Local mode: no model was called.")
    print()
    for index, hit in enumerate(answer.hits[:k], start=1):
        print(f"{index}. {hit.source}  ({hit.score:.3f})")
        print(f"   {hit.snippet(200)}")
        print()


def print_remote(answer: rc.Answer, show_steps: bool) -> None:
    print(answer.text.strip())
    print()
    if answer.sources:
        print("Sources")
        for item in answer.sources:
            source = item.get("source", "unknown")
            score = float(item.get("relevance", 0.0) or 0.0)
            print(f"  {source}  {score:.3f}")
        print()
    if show_steps:
        for step in answer.steps:
            detail = f" {step.detail}" if step.detail else ""
            print(f"  {step.kind}: {step.label}{detail}")
        print()
    if answer.cached:
        print(f"cached, 0 API calls (answered by {answer.model})")
    elif answer.local:
        print("0 API calls (local mode)")
    else:
        print(f"{answer.calls} API calls, {answer.elapsed:.1f} s, {answer.model}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Oracle RAG corpus from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="+", help="the question, quoted or as plain words")
    parser.add_argument("--local", action="store_true",
                        help="search only, no model call, zero quota")
    parser.add_argument("--json", action="store_true", help="print the answer as JSON")
    parser.add_argument("--no-cache", action="store_true",
                        help="skip the answer cache and force a fresh model run")
    parser.add_argument("--k", type=int, default=4,
                        help="hits to show in local mode (default: 4)")
    parser.add_argument("--weight", type=float, default=0.6,
                        help="hybrid weight, 0 keyword to 1 vector (default: 0.6)")
    parser.add_argument("--steps", action="store_true", help="print the agent trace")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        print("error: ask something first")
        return 2

    k = max(1, args.k)
    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    agent = rc.Agent(engine, settings)
    try:
        if args.local:
            answer = agent.ask_local(question, k=k, weight=args.weight)
        else:
            answer = agent.ask(question, use_cache=not args.no_cache)
    except (rc.NoModel, rc.QuotaExhausted) as exc:
        print(f"error: {exc}")
        print('try: python ask.py --local "your question"')
        return 3
    except rc.RagError as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 2
    finally:
        engine.close()

    if args.json:
        data = rc.answer_to_dict(answer)
        data["question"] = question
        data["cached"] = answer.cached
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.local:
        print_local(answer, k)
    else:
        print_remote(answer, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
