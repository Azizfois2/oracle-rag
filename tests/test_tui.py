"""Headless tests for the Textual interface. No database, no model, no quota.

    python tests/test_tui.py

The app is driven with Textual's Pilot against a fake agent and an in-memory
history. Artefacts are written to a temporary folder.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import artifacts as arts  # noqa: E402
import rag_core as rc  # noqa: E402
import tui  # noqa: E402

failures: list[str] = []
total = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global total
    total += 1
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


class FakeAgent:
    """Streams a deterministic local or remote answer."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask_local(self, question, k=4, weight=0.6, context=None):
        self.questions.append(question)
        return rc.Answer(
            text="",
            model="local (no LLM)",
            local=True,
            hits=[rc.Hit("1", "doc.md", "Text about hybrid search.", 0.9,
                         "hybrid")],
        )

    def ask(self, question, on_step=None, on_token=None, on_reset=None,
            use_cache=True, context=None, **kwargs):
        self.questions.append(question)
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
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.titles: dict[str, str] = {}

    def append(self, chat_id, role, content, answer=None, title=None):
        self.rows.append({"chat_id": chat_id, "role": role,
                          "content": content, "answer": answer})
        self.titles.setdefault(chat_id, content[:40] if role == "user"
                               else "Untitled")

    def list(self, limit=20):
        out = []
        seen: list[str] = []
        for row in reversed(self.rows):
            if row["chat_id"] not in seen:
                seen.append(row["chat_id"])
        for chat_id in seen[:limit]:
            count = sum(1 for row in self.rows if row["chat_id"] == chat_id)
            out.append({"id": chat_id, "title": self.titles.get(chat_id, "Untitled"),
                        "messages": count, "updated": "2026-11-10T12:00:00+00:00"})
        return out

    def messages(self, chat_id):
        return [{"role": row["role"], "content": row["content"],
                 "answer": row["answer"]}
                for row in self.rows if row["chat_id"] == chat_id]

    def title(self, chat_id):
        return self.titles.get(chat_id, "Untitled")

    def rename(self, chat_id, title):
        self.titles[chat_id] = title
        return sum(1 for row in self.rows if row["chat_id"] == chat_id)

    def search(self, query, limit=20):
        return []

    def delete(self, chat_id):
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["chat_id"] != chat_id]
        return before - len(self.rows)


async def exercise(root: Path) -> None:
    agent = FakeAgent()
    history = FakeHistory()
    app = tui.OracleTui(rc.Settings(), agent, history, local=True)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        check("TUI starts dark", app.theme == "textual-dark", app.theme)
        check("transcript is mounted", app.query_one("#transcript", tui.RichLog)
              is not None)
        check("prompt is focused", app.focused is not None)

        await pilot.click("#theme")
        await pilot.pause()
        check("theme button switches to light", app.theme == "textual-light",
              app.theme)
        app.action_toggle_theme()
        await pilot.pause()
        check("theme toggles back to dark", app.theme == "textual-dark",
              app.theme)

        prompt = app.query_one("#prompt", tui.Input)
        prompt.focus()
        prompt.value = "hybrid search"
        await pilot.press("enter")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        check("local answer is stored", app.last_answer is not None)
        check("local answer is local", bool(app.last_answer and app.last_answer.local))
        check("the question reached the agent",
              agent.questions == ["hybrid search"], str(agent.questions))
        check("both turns are saved", len(history.rows) == 2, str(len(history.rows)))
        check("the prompt is cleared", prompt.value == "")
        check("the app is idle again", app._busy is False)

        app.action_save_artefact()
        await pilot.pause()
        saved = arts.list_artefacts(root)
        check("save writes three artefacts", len(saved) == 3, str(len(saved)))
        check("the artefacts pane sees them", len(app._artefacts) == 3,
              str(len(app._artefacts)))

        prompt.value = "/sources"
        await pilot.press("enter")
        await pilot.pause()
        check("sources command runs", not app._busy)

        prompt.value = "/artefacts"
        await pilot.press("enter")
        await pilot.pause()
        check("artefacts command runs", len(app._artefacts) == 3)

        prompt.value = "/preview 1"
        await pilot.press("enter")
        await pilot.pause()
        check("preview command runs", not app._busy)

        prompt.value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        check("help command runs", not app._busy)

        prompt.value = "/new"
        await pilot.press("enter")
        await pilot.pause()
        check("new discussion clears the turns", app.turns == [])
        check("new discussion gets an id", bool(app.chat_id))


def main() -> int:
    tmp = tempfile.TemporaryDirectory(prefix="rag-tui-")
    os.environ["RAG_ARTIFACTS_DIR"] = str(Path(tmp.name) / "artefacts")
    try:
        asyncio.run(exercise(Path(os.environ["RAG_ARTIFACTS_DIR"])))
    finally:
        tmp.cleanup()
    failed = len(failures)
    print(f"\n  {total - failed}/{total} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
