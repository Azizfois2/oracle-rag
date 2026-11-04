"""The console must still start when the database is down, and say how to fix it.

    python tests/test_db_down.py

Re-executes this file in a child process with a deliberately unreachable DSN, so
the cached connection from any other test cannot mask the failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def phase_down() -> int:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    body = "\n".join(str(getattr(element, "value", "")) for element in at.markdown)
    body += "\n" + "\n".join(str(getattr(element, "value", "")) for element in at.code)

    checks = [
        ("no traceback", not at.exception, str(at.exception)),
        ("offline screen shown", "database is not reachable" in body, ""),
        ("cause repeated back", "NOPE" in body, ""),
        ("fix command shown", "docker start oracle-26ai-free" in body, ""),
        ("re-check offered", any("Re-check" in str(b.label) for b in at.button), ""),
    ]
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        failed += 0 if ok else 1
    return 1 if failed else 0


def main() -> int:
    env = os.environ.copy()
    env["ORACLE_DSN"] = "localhost:1/NOPE"
    env["ORACLE_USER"] = "nobody"
    env["ORACLE_PWD"] = "nobody"
    env.pop("RAG_TABLE", None)

    print("== database down ==")
    proc = subprocess.run(
        [sys.executable, __file__, "--down"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0 and proc.stderr.strip():
        sys.stderr.write(proc.stderr[-2000:])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(phase_down() if "--down" in sys.argv else main())
