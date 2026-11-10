"""Artefact writer and reader tests. No database, no model, no quota.

    python tests/test_artifacts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import artifacts as arts  # noqa: E402
import rag_core as rc  # noqa: E402

failures: list[str] = []
total = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global total
    total += 1
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


class FakeHistory:
    """The small ChatHistory surface used by save_discussion."""

    def __init__(self) -> None:
        self.rows = [
            {"role": "user", "content": "What is hybrid search?",
             "at": "2026-11-10T10:00:00+00:00"},
            {"role": "assistant", "content": "", "at": "2026-11-10T10:00:02+00:00",
             "answer": rc.Answer(
                 text="It combines keyword and vector scores.",
                 model="fake",
                 sources=[{"source": "doc.md", "relevance": 0.9}],
                 steps=[rc.Step("search", "hybrid", "3 hits")],
             )},
        ]

    def title(self, chat_id: str) -> str:
        return "Hybrid search discussion"

    def messages(self, chat_id: str) -> list[dict]:
        return list(self.rows)

    def export_markdown(self, chat_id: str) -> str:
        return "# Hybrid search discussion\n\n## You\n\nWhat is hybrid search?\n"


def sample_answer() -> rc.Answer:
    return rc.Answer(
        text="Hybrid search combines both signals [1].\n\nA second paragraph.",
        model="fake-model",
        steps=[rc.Step("search", "hybrid search", "4 hits"),
               rc.Step("answer", "answer ready")],
        sources=[{
            "source": "01_nvidia.md",
            "relevance": 0.93,
            "published": "2026-02-14",
            "citation": 1,
            "citations": [1],
            "snippet": "NVIDIA announced infrastructure partnerships.",
        }],
        calls=2,
        elapsed=1.4,
        hits=[rc.Hit("1", "01_nvidia.md", "A passage about NVIDIA.", 0.93,
                     "hybrid")],
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rag-artefacts-") as tmp:
        root = Path(tmp) / "artefacts"
        answer = sample_answer()

        print("\n== writing an answer ==")
        saved = arts.save_answer("What is hybrid search?", answer,
                                 chat_id="abc123", directory_path=root)
        check("three formats written", len(saved) == 3, str(len(saved)))
        check("files exist", all(item.path.is_file() for item in saved))
        check("files have content", all(item.path.stat().st_size > 0 for item in saved))
        check("kinds are markdown, json and html",
              {item.kind for item in saved} == {"markdown", "json", "html"},
              str({item.kind for item in saved}))
        check("title comes from the question",
              all(item.title == "What is hybrid search?" for item in saved),
              str([item.title for item in saved]))
        check("a JSON file names the discussion",
              any('"chat_id": "abc123"' in item.path.read_text(encoding="utf-8")
                  for item in saved if item.kind == "json"))

        md = next(item for item in saved if item.kind == "markdown")
        body = md.path.read_text(encoding="utf-8")
        check("markdown keeps the question", "What is hybrid search?" in body)
        check("markdown keeps the answer", "combines both signals" in body)
        check("markdown keeps the source", "01_nvidia.md" in body)
        check("markdown keeps the published date", "2026-02-14" in body)
        check("markdown keeps the trace", "answer ready" in body)
        check("markdown records the run", "2 API calls" in body)

        js = next(item for item in saved if item.kind == "json")
        data = json.loads(js.path.read_text(encoding="utf-8"))
        check("json declares its format", data.get("format") == arts.FORMAT)
        check("json declares its version", data.get("version") == arts.VERSION)
        check("json keeps the answer payload", data["answer"]["text"].startswith(
            "Hybrid search"))
        check("json keeps call counts", data["answer"]["calls"] == 2)
        check("json records the cache flag", data["answer"]["cached"] is False)

        page = next(item for item in saved if item.kind == "html")
        html_body = page.path.read_text(encoding="utf-8")
        check("html is a complete page", html_body.startswith("<!doctype html>"))
        check("html carries the answer", "combines both signals" in html_body)
        check("html carries the source", "01_nvidia.md" in html_body)
        check("html supports a dark system theme",
              "prefers-color-scheme:dark" in html_body)

        print("\n== directory helpers ==")
        listed = arts.list_artefacts(root)
        check("list finds every file", len(listed) == 3, str(len(listed)))
        check("list is newest first",
              listed == sorted(listed, key=lambda item: item.path.stat().st_mtime,
                               reverse=True))
        stats = arts.stats(root)
        check("stats counts the files", stats["count"] == 3, str(stats["count"]))
        check("stats counts each kind",
              stats["kinds"] == {"markdown": 1, "json": 1, "html": 1, "text": 0},
              str(stats["kinds"]))
        check("stats reports the folder", Path(stats["directory"]) == root.resolve())

        print("\n== reading and deleting ==")
        check("read returns the same bytes",
              arts.read_artefact(md, root) == body)
        check("preview clips a long file",
              md.preview(limit=10).startswith(body[:10].rstrip())
              and "preview clipped" in md.preview(limit=10))
        check("delete removes one file", arts.delete_artefact(md, root))
        check("deleted file is gone", not md.path.exists())
        check("delete is idempotent",
              arts.delete_artefact(md, root) is False)
        raised = False
        try:
            arts.delete_artefact("../outside.md", root)
        except ValueError:
            raised = True
        check("paths outside the folder are rejected", raised)

        print("\n== text and discussions ==")
        text = arts.save_text("A note", "One line.\n", suffix="txt",
                              directory_path=root)
        check("save_text writes a file", text.kind == "text" and text.path.exists())
        check("save_text reads back", arts.read_artefact(text, root) == "One line.\n")

        history = FakeHistory()
        discussion = arts.save_discussion(history, "abc123", directory_path=root)
        check("discussion writes markdown and json", len(discussion) == 2,
              str(len(discussion)))
        check("discussion title is used",
              all(item.title == "Hybrid search discussion" for item in discussion),
              str([item.title for item in discussion]))
        d_json = next(item for item in discussion if item.kind == "json")
        payload = json.loads(d_json.path.read_text(encoding="utf-8"))
        check("discussion payload names its kind", payload.get("kind") == "discussion")
        check("discussion payload keeps both messages",
              len(payload.get("messages") or []) == 2)
        check("discussion payload keeps the saved answer",
              payload["messages"][1]["answer"]["text"].startswith("It combines"))

        print("\n== unsafe input ==")
        hostile = rc.Answer(text="<script>alert(1)</script>", model="fake")
        page = arts.save_answer("", hostile, formats=("html",),
                                directory_path=root)[0]
        page_body = page.path.read_text(encoding="utf-8")
        check("html escapes model output", "<script>alert(1)</script>" not in page_body)
        check("html shows the escaped text",
              "&lt;script&gt;alert(1)&lt;/script&gt;" in page_body)

    failed = len(failures)
    print(f"\n  {total - failed}/{total} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
