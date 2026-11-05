"""The command-line interface must answer without Streamlit and without quota.

    python tests/test_cli.py

Runs ask.py in local mode, so no Gemini key and no API calls are needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(ROOT / "ask.py"), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        timeout=300,
    )


def text(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout.decode("utf-8", errors="replace")


def main() -> int:
    print("== ask.py ==")
    checks: list[tuple[str, bool, str]] = []

    proc = run("--local", "--json", "hybrid search")
    body = text(proc)
    try:
        start, end = body.find("{"), body.rfind("}")
        data = json.loads(body[start:end + 1]) if start >= 0 and end > start else {}
    except json.JSONDecodeError:
        data = {}
    checks.append(("local JSON exits 0", proc.returncode == 0,
                   "" if proc.returncode == 0 else text(proc)[-300:]))
    checks.append(("JSON parses", bool(data), ""))
    checks.append(("question is echoed", data.get("question") == "hybrid search", ""))
    checks.append(("local model recorded", data.get("model") == "local (no LLM)", str(data.get("model"))))
    checks.append(("hits returned", bool(data.get("hits")), f"{len(data.get('hits') or [])} hits"))
    checks.append(("sources returned", bool(data.get("sources")), f"{len(data.get('sources') or [])} sources"))

    proc = run("--local", "hybrid search")
    body = text(proc)
    checks.append(("text mode exits 0", proc.returncode == 0,
                   "" if proc.returncode == 0 else body[-300:]))
    checks.append(("text mode says local", "Local mode" in body, ""))
    checks.append(("text mode shows a source", any(name in body for name in (
        "01_nvidia", "07_startup", "08_langchain", "10_open_source")), ""))

    proc = run()
    checks.append(("missing question is rejected", proc.returncode == 2,
                   text(proc)[-200:]))

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        failed += 0 if ok else 1
    print(f"\n  {len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
