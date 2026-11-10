"""The command-line interface must answer without Streamlit and without quota.

    python tests/test_cli.py

Runs ask.py in local mode, so no Gemini key and no API calls are needed. The
interactive chat is driven with piped input and a fake agent, so it never
touches a real discussion or the network.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ask  # noqa: E402
import rag_core as rc  # noqa: E402
from rich.console import Console  # noqa: E402


def run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(ROOT / "ask.py"), *args],
        cwd=str(ROOT),
        env=env,
        input=stdin.encode("utf-8") if stdin is not None else None,
        capture_output=True,
        timeout=300,
    )


def text(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout.decode("utf-8", errors="replace")


def feed(lines: list[str]):
    pending = iter(lines)
    return lambda prompt: next(pending, None)


def console_buffer() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=False, width=100), buffer


class FakeAgent:
    """Records the question and the follow-up context it was given."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask_local(self, question, k=4, weight=0.6, context=None):
        self.calls.append({"question": question, "context": list(context or [])})
        return rc.Answer(
            text="",
            model="local (no LLM)",
            local=True,
            hits=[rc.Hit("1", "doc.md", "Text about hybrid search.", 0.9, "hybrid")],
        )

    def ask(self, question, on_step=None, recursion_limit=25, use_cache=True,
            on_token=None, on_reset=None, context=None):
        self.calls.append({"question": question, "context": list(context or [])})
        if on_reset:
            on_reset()
        if on_token:
            for piece in ("Fake ", "answer."):
                on_token(piece)
        return rc.Answer(
            text="Fake answer.",
            model="fake",
            calls=1,
            elapsed=0.1,
            steps=[rc.Step("answer", "answer ready")],
            sources=[{"source": "doc.md", "relevance": 0.9}],
        )


class FakeHistory:
    """An in-memory ChatHistory, so the save and restore path needs no Oracle."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.titles: dict[str, str] = {}

    def append(self, chat_id, role, content, answer=None, title=None):
        if chat_id not in self.titles:
            self.titles[chat_id] = content[:72] if role == "user" else "Untitled discussion"
        self.rows.append({"chat_id": chat_id, "role": role,
                          "content": content, "answer": answer})

    def list(self, limit=20):
        ids: list[str] = []
        for row in reversed(self.rows):
            if row["chat_id"] not in ids:
                ids.append(row["chat_id"])
        out = []
        for chat_id in ids[:limit]:
            count = sum(1 for row in self.rows if row["chat_id"] == chat_id)
            out.append({"id": chat_id, "title": self.titles.get(chat_id, "Untitled"),
                        "messages": count, "updated": "2026-11-07T12:00:00+00:00"})
        return out

    def messages(self, chat_id):
        return [{"role": row["role"], "content": row["content"],
                 "answer": row["answer"]}
                for row in self.rows if row["chat_id"] == chat_id]

    def title(self, chat_id):
        return self.titles.get(chat_id, "Untitled discussion")

    def rename(self, chat_id, title):
        self.titles[chat_id] = title
        return sum(1 for row in self.rows if row["chat_id"] == chat_id)

    def search(self, query, limit=20):
        found = []
        for item in self.list(limit=50):
            hits = [row for row in self.rows
                    if row["chat_id"] == item["id"]
                    and query.lower() in str(row["content"]).lower()]
            if hits:
                found.append({**item, "matches": len(hits),
                              "snippet": str(hits[-1]["content"])[:120]})
        return found[:limit]

    def delete(self, chat_id):
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["chat_id"] != chat_id]
        self.titles.pop(chat_id, None)
        return before - len(self.rows)


def session_checks() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    agent = FakeAgent()
    console, buffer = console_buffer()
    session = ask.ChatSession(
        agent, None, local=True, console=console, tty=False,
        reader=feed(["hybrid search", "/sources", "/quit"]),
    )
    code = session.run()
    body = buffer.getvalue()
    checks.append(("chat session exits 0", code == 0, str(code)))
    checks.append(("chat session prints local hits", "doc.md" in body, ""))
    checks.append(("chat session says local mode", "Local mode" in body, ""))
    checks.append(("chat session says goodbye", "Bye." in body, ""))
    checks.append(("chat session asked the question",
                   bool(agent.calls) and agent.calls[0]["question"] == "hybrid search",
                   str(agent.calls[:1])))
    checks.append(("first question has no context",
                   bool(agent.calls) and agent.calls[0]["context"] == [],
                   str(agent.calls[:1])))

    agent = FakeAgent()
    console, buffer = console_buffer()
    ask.ChatSession(
        agent, None, local=True, console=console, tty=False,
        reader=feed(["first question", "second question", "/quit"]),
    ).run()
    context = agent.calls[1]["context"] if len(agent.calls) > 1 else []
    checks.append(("follow-up sends earlier turns",
                   len(context) == 2
                   and context[0].get("role") == "user"
                   and context[1].get("role") == "assistant",
                   str([item.get("role") for item in context])))

    agent = FakeAgent()
    console, buffer = console_buffer()
    ask.ChatSession(
        agent, None, local=True, console=console, tty=False,
        reader=feed(["first question", "/new", "second question", "/quit"]),
    ).run()
    context = agent.calls[1]["context"] if len(agent.calls) > 1 else ["missing"]
    checks.append(("new discussion clears the context", context == [], str(context)))

    agent = FakeAgent()
    console, buffer = console_buffer()
    ask.ChatSession(
        agent, None, local=False, console=console, tty=False,
        reader=feed(["remote question", "/sources", "/trace", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("remote answer is rendered", "Fake answer." in body, body[-200:]))
    checks.append(("remote sources are listed", "doc.md" in body and "Sources" in body, ""))
    checks.append(("remote trace is listed", "answer ready" in body and "Trace" in body, ""))
    checks.append(("remote footer counts calls", "1 API calls" in body, body[-120:]))

    console, buffer = console_buffer()
    streaming = Console(file=buffer, force_terminal=True, width=100)
    ask.ChatSession(
        FakeAgent(), None, local=False, console=streaming, tty=True,
        reader=feed(["stream please", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("tty streaming ends with the final answer",
                   "Fake answer." in body, body[-160:]))

    console, buffer = console_buffer()
    ask.ChatSession(
        FakeAgent(), None, local=True, console=console, tty=False,
        reader=feed(["/help", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("help lists the commands",
                   all(name in body for name in ("/find", "/sources", "/title", "/delete")),
                   body[-200:]))

    console, buffer = console_buffer()
    ask.ChatSession(
        FakeAgent(), None, local=True, console=console, tty=False,
        reader=feed(["/title", "/delete", "/history", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("history commands explain the session is memory only",
                   "History is off" in body and "No saved discussion" in body,
                   body[-240:]))

    history = FakeHistory()
    console, buffer = console_buffer()
    ask.ChatSession(
        FakeAgent(), history, local=True, console=console, tty=False,
        reader=feed(["saved question", "/title Saved title", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("chat saves both turns", len(history.rows) == 2,
                   str(len(history.rows))))
    checks.append(("chat renames the saved discussion",
                   history.titles.get(history.rows[0]["chat_id"]) == "Saved title",
                   str(history.titles)))
    checks.append(("rename is confirmed on screen", "Renamed to Saved title" in body, ""))

    agent = FakeAgent()
    console, buffer = console_buffer()
    ask.ChatSession(
        agent, history, local=True, console=console, tty=False,
        reader=feed(["follow up", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("chat restores the latest discussion", "Continuing" in body, ""))
    checks.append(("restored title is shown", "Saved title" in body, ""))
    checks.append(("restored discussion feeds the context",
                   bool(agent.calls) and len(agent.calls[0]["context"]) == 2,
                   str(agent.calls[:1])))

    history = FakeHistory()
    console, buffer = console_buffer()
    ask.ChatSession(
        FakeAgent(), history, local=True, console=console, tty=False,
        reader=feed(["question one", "/delete", "/delete", "/quit"]),
    ).run()
    body = buffer.getvalue()
    checks.append(("delete needs two commands", "Type /delete again" in body, ""))
    checks.append(("delete removes the saved rows", history.rows == [],
                   str(len(history.rows))))
    checks.append(("delete reports the row count", "Deleted 2 messages." in body, ""))

    return checks


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

    print("\n== ask.py --chat ==")
    proc = run("--chat", "--local", "--no-history",
               stdin="hybrid search\n/quit\n")
    body = text(proc)
    checks.append(("chat mode exits 0", proc.returncode == 0,
                   "" if proc.returncode == 0 else body[-300:]))
    checks.append(("chat mode says local", "Local mode" in body, ""))
    checks.append(("chat mode shows a source", any(name in body for name in (
        "01_nvidia", "07_startup", "08_langchain", "10_open_source")), ""))
    checks.append(("chat mode says goodbye", "Bye." in body, ""))

    proc = run("--chat", "--local", "--no-history", stdin="/help\n/quit\n")
    body = text(proc)
    checks.append(("chat help lists commands",
                   all(name in body for name in ("/find", "/sources", "/title")), ""))

    proc = run("--chat", "--json", stdin="/quit\n")
    checks.append(("chat rejects JSON output", proc.returncode == 2,
                   text(proc)[-160:]))

    proc = run("--chat", "--tui", stdin="/quit\n")
    checks.append(("chat and TUI are mutually exclusive", proc.returncode == 2,
                   text(proc)[-160:]))

    proc = run("--tui", "--json")
    checks.append(("TUI rejects one-shot JSON", proc.returncode == 2,
                   text(proc)[-160:]))

    print("\n== ChatSession ==")
    checks.extend(session_checks())

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        failed += 0 if ok else 1
    print(f"\n  {len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
