"""Run every zero-quota test suite and print one summary.

    python tests/run_all.py
    python tests/run_all.py --list
    python tests/run_all.py --only test_answers test_cli
    python tests/run_all.py --fail-fast

This never runs `smoke_test.py --llm` and never runs the live answer check in
`quality/check_answers.py`, so it spends no Gemini quota. Each suite keeps its
full output in `logs/tests/<name>.log`, which is gitignored.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "tests"


@dataclass(frozen=True)
class Suite:
    name: str
    script: str
    note: str


SUITES = (
    Suite("smoke_test", "tests/smoke_test.py",
          "engine, retrieval, dates and the agent contract"),
    Suite("test_app", "tests/test_app.py",
          "headless Streamlit render and live search"),
    Suite("test_answers", "tests/test_answers.py",
          "answer gate scoring with fake answers"),
    Suite("test_cli", "tests/test_cli.py",
          "ask.py one-shot and interactive chat"),
    Suite("test_history", "tests/test_history.py",
          "discussion persistence on a scratch table"),
    Suite("test_mcp", "tests/test_mcp.py",
          "MCP server over real stdio"),
    Suite("test_db_down", "tests/test_db_down.py",
          "offline screen without a traceback"),
    Suite("test_artifacts", "tests/test_artifacts.py",
          "answer artefacts written, read and escaped"),
    Suite("test_tui", "tests/test_tui.py",
          "headless Textual panes, theme and artefacts"),
)


@dataclass
class Result:
    suite: Suite
    ok: bool
    seconds: float
    detail: str = ""
    output: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-quota suites and print one summary.")
    parser.add_argument("--list", action="store_true",
                        help="list the suites and exit")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        choices=[suite.name for suite in SUITES],
                        help="run only these suites")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop after the first failing suite")
    parser.add_argument("--verbose", action="store_true",
                        help="print each suite's full output")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds allowed per suite (default 1800)")
    return parser.parse_args()


def selected(args: argparse.Namespace) -> list[Suite]:
    if not args.only:
        return list(SUITES)
    wanted = set(args.only)
    return [suite for suite in SUITES if suite.name in wanted]


def run_suite(suite: Suite, timeout: int) -> Result:
    script = ROOT / suite.script
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - started
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return Result(suite, False, seconds,
                      f"timed out after {timeout}s", output)
    except OSError as exc:
        return Result(suite, False, time.perf_counter() - started,
                      f"could not start: {exc}", "")

    seconds = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    detail = "" if ok else f"exit code {proc.returncode}"
    return Result(suite, ok, seconds, detail, output)


def tail(output: str, lines: int = 20) -> list[str]:
    return [line for line in output.splitlines() if line.strip()][-lines:]


def main() -> int:
    args = parse_args()
    if args.list:
        print(f"{len(SUITES)} zero-quota suites:")
        for suite in SUITES:
            print(f"  {suite.name:14s} {suite.script:24s} {suite.note}")
        return 0

    suites = selected(args)
    if not suites:
        print("no suites selected")
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Oracle RAG Console - {len(suites)} zero-quota suites")
    print("No Gemini calls: --llm and the live answer check are excluded.\n")

    results: list[Result] = []
    total_started = time.perf_counter()
    for index, suite in enumerate(suites, start=1):
        print(f"[{index}/{len(suites)}] {suite.name} ... ", end="", flush=True)
        result = run_suite(suite, args.timeout)
        results.append(result)
        mark = "pass" if result.ok else "FAIL"
        extra = f"  ({result.detail})" if result.detail else ""
        print(f"{mark} in {result.seconds:.1f}s{extra}")

        log_path = LOG_DIR / f"{suite.name}.log"
        log_path.write_text(result.output, encoding="utf-8")
        if args.verbose:
            print(result.output)
        if not result.ok:
            for line in tail(result.output):
                print(f"    {line}")
            print(f"    full output: {log_path}")
            print(f"    rerun: {sys.executable} {suite.script}")
            if args.fail_fast:
                print("\nstopping after the first failure")
                break

    elapsed = time.perf_counter() - total_started
    passed = sum(1 for result in results if result.ok)
    failed = [result for result in results if not result.ok]
    print("\n" + "-" * 64)
    print(f"  {passed}/{len(results)} suites passed in {elapsed:.1f}s")
    print(f"  logs: {LOG_DIR}")
    if failed:
        print("  failed: " + ", ".join(result.suite.name for result in failed))
        return 1
    if len(results) < len(suites):
        print("  not all selected suites ran")
        return 1
    print("  no Gemini quota was spent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
