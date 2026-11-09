"""Read and write answer-gate progress and completed-run history.

Progress lives in `data/answers-last.json` so a quota-stopped run can resume.
Completed runs are appended to `data/answers-history.json` for the trend view.
Both files are gitignored and contain no key.

The paths can be overridden with `ANSWER_PROGRESS_PATH` and
`ANSWER_HISTORY_PATH`, which keeps the tests away from real local state.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from quality.answer_eval import summarize_rows

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HISTORY_VERSION = 1
MAX_RUNS = 100


def progress_path() -> Path:
    raw = os.getenv("ANSWER_PROGRESS_PATH")
    return Path(raw) if raw else DATA_DIR / "answers-last.json"


def history_path() -> Path:
    raw = os.getenv("ANSWER_HISTORY_PATH")
    return Path(raw) if raw else DATA_DIR / "answers-history.json"


def read_progress(path: Path | str | None = None) -> dict[str, Any]:
    """Read a saved run, or an empty dict when there is none."""
    target = Path(path) if path else progress_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def saved_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Case scores from a saved run, in the older or newer layout."""
    rows = payload.get("cases")
    if not isinstance(rows, list):
        summary = payload.get("summary")
        rows = summary.get("cases") if isinstance(summary, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def save_progress(payload: dict[str, Any],
                  path: Path | str | None = None) -> None:
    target = Path(path) if path else progress_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")


def clear_progress(path: Path | str | None = None) -> bool:
    """Delete the saved run so the next check starts from case one."""
    target = Path(path) if path else progress_path()
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def progress_payload(fingerprint: str, corpus_rows: int,
                     corpus_sources: Iterable[str], model: str,
                     rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A progress file that carries the summary and every scored case."""
    return {
        "version": 1,
        "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "corpus_fingerprint": str(fingerprint or ""),
        "corpus_rows": int(corpus_rows or 0),
        "corpus_sources": list(corpus_sources),
        "model": str(model or ""),
        "summary": summarize_rows(rows),
    }


def load_runs(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Completed runs, oldest first."""
    target = Path(path) if path else history_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    runs = data.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, dict)]


def append_run(payload: dict[str, Any],
               path: Path | str | None = None,
               expected_total: int | None = None) -> bool:
    """Archive one completed run. Returns False when it is incomplete or a duplicate."""
    summary = payload.get("summary")
    if not isinstance(summary, dict) or not int(summary.get("total") or 0):
        return False
    total = int(summary.get("total") or 0)
    if expected_total is not None and total != int(expected_total):
        return False
    checks = summary.get("checks") or {}
    by_type = summary.get("by_type") or {}
    entry = {
        "at": str(payload.get("at")
                  or dt.datetime.now().astimezone().isoformat(timespec="seconds")),
        "corpus_fingerprint": str(payload.get("corpus_fingerprint") or ""),
        "corpus_rows": int(payload.get("corpus_rows") or 0),
        "model": str(payload.get("model") or ""),
        "total": int(summary.get("total") or 0),
        "passed": int(summary.get("passed") or 0),
        "score": float(summary.get("score") or 0.0),
        "checks": {str(name): dict(bucket)
                   for name, bucket in checks.items()
                   if isinstance(bucket, dict)},
        "by_type": {str(name): dict(bucket)
                    for name, bucket in by_type.items()
                    if isinstance(bucket, dict)},
    }
    target = Path(path) if path else history_path()
    runs = load_runs(target)
    key = (entry["at"], entry["corpus_fingerprint"],
           entry["total"], entry["passed"])
    if any((str(run.get("at") or ""),
            str(run.get("corpus_fingerprint") or ""),
            int(run.get("total") or 0),
            int(run.get("passed") or 0)) == key for run in runs):
        return False
    runs.append(entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"version": HISTORY_VERSION, "runs": runs[-MAX_RUNS:]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return True
