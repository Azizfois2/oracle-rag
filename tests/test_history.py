"""Discussion history must survive a restart and live in Oracle.

    python tests/test_history.py

Uses a scratch table, RAG_CHAT_LOG_TEST, and drops it afterwards. The real
discussion history is never touched. No Gemini key and no API quota are used.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc  # noqa: E402

TEST_TABLE = "RAG_CHAT_LOG_TEST"
checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    history = rc.ChatHistory(engine, table=TEST_TABLE)
    chat_id = uuid.uuid4().hex
    other_id = uuid.uuid4().hex

    print("== discussion history ==")
    try:
        try:
            history.drop()
        except rc.RagError:
            pass

        history.ensure()
        info = history.stats()
        check("scratch table created", info["table"] == TEST_TABLE, info["table"])

        try:
            rc.ChatHistory(engine, table="bad name")
            bad_name = False
        except rc.RagError:
            bad_name = True
        check("invalid table name is refused", bad_name)

        try:
            rc.ChatHistory(engine, table=rc.TABLE).drop()
            protected = False
        except rc.RagError:
            protected = True
        check("drop refuses the real corpus table", protected)

        history.append(chat_id, "user",
                       "Which company raised an $85M Series B to build a hybrid search database?")
        listed = history.list(limit=50)
        row = next((item for item in listed if item["id"] == chat_id), None)
        check("discussion appears in the list", row is not None)
        check("title comes from the first question",
              bool(row) and "Series B" in row["title"],
              row["title"] if row else "none")
        check("message count starts at one", bool(row) and row["messages"] == 1,
              str(row["messages"]) if row else "none")
        check("timestamp is ISO", bool(row) and "T" in row["updated"])

        answer = rc.Answer(
            text="VectorForge raised $85M.",
            model="test-model",
            steps=[rc.Step("search", "vector", "3 hits")],
            sources=[{"source": "07_startup_vector_database.md", "relevance": 0.91}],
            calls=2,
            elapsed=1.4,
            hits=[rc.Hit("k1", "07_startup_vector_database.md",
                         "VectorForge raised $85M.", 0.8, "hybrid",
                         {"vector": 0.7, "keyword": 0.9})],
        )
        history.append(chat_id, "assistant", answer.text, answer=answer)
        messages = history.messages(chat_id)
        check("both messages load", len(messages) == 2, str(len(messages)))
        check("user message is intact",
              bool(messages) and messages[0]["role"] == "user"
              and "Series B" in messages[0]["content"])
        got = messages[1].get("answer") if len(messages) > 1 else None
        check("assistant answer is rebuilt",
              isinstance(got, rc.Answer) and got.text == answer.text)
        check("model survives", bool(got) and got.model == "test-model",
              got.model if got else "none")
        check("steps survive",
              bool(got) and got.steps and got.steps[0].label == "vector")
        check("sources survive",
              bool(got) and got.sources
              and got.sources[0]["source"] == "07_startup_vector_database.md")
        check("hit detail survives",
              bool(got) and got.hits and got.hits[0].detail["vector"] == 0.7)
        check("call count survives", bool(got) and got.calls == 2, str(got.calls) if got else "none")

        history.append(chat_id, "error", "Quota exhausted")
        messages = history.messages(chat_id)
        check("error message loads",
              len(messages) == 3 and messages[2]["role"] == "error"
              and "Quota" in messages[2]["content"], str(len(messages)))

        changed = history.rename(chat_id, "  Price   and funding  ")
        row = next((item for item in history.list(limit=50) if item["id"] == chat_id), None)
        check("rename updates every row", changed == 3, str(changed))
        check("rename collapses extra spaces",
              bool(row) and row["title"] == "Price and funding",
              row["title"] if row else "none")
        check("title reads the stored title",
              history.title(chat_id) == "Price and funding", history.title(chat_id))
        check("unknown discussion gets a safe title",
              history.title("missing") == "Untitled discussion")

        exported = history.export_markdown(chat_id)
        check("export includes the question", "Series B" in exported)
        check("export includes the answer", "VectorForge raised $85M." in exported)
        check("export includes sources", "07_startup_vector_database.md" in exported)
        check("export includes the trace", "Trace" in exported and "vector" in exported)
        check("export includes the error", "Quota exhausted" in exported)

        history.append(other_id, "user", "What did AWS release?")
        history.append(other_id, "assistant",
                       "The launch reached 100% of capacity.")
        ids = [item["id"] for item in history.list(limit=50)]
        check("newest discussion comes first", bool(ids) and ids[0] == other_id,
              ids[0] if ids else "none")

        found = history.search("vectorforge", limit=10)
        row = next((item for item in found if item["id"] == chat_id), None)
        check("search finds a saved discussion", row is not None)
        check("search is case-insensitive",
              bool(row) and "VectorForge" in row["snippet"],
              row["snippet"] if row else "none")
        check("search counts matching messages",
              bool(row) and row["matches"] >= 1,
              str(row["matches"]) if row else "none")
        check("search matches a phrase with spaces",
              any(item["id"] == chat_id for item in history.search("series b")))
        check("search reaches error messages",
              any(item["id"] == chat_id for item in history.search("quota")))
        check("search finds the other discussion",
              any(item["id"] == other_id for item in history.search("AWS")))
        percent = history.search("100%", limit=10)
        check("search treats percent as text",
              len(percent) == 1 and percent[0]["id"] == other_id
              and "100%" in percent[0]["snippet"],
              str(len(percent)))
        wildcard = history.search("%", limit=10)
        check("search escapes a bare wildcard",
              len(wildcard) == 1 and wildcard[0]["id"] == other_id,
              str(len(wildcard)))
        check("search with no match returns nothing",
              history.search("no such phrase") == [])
        check("blank search returns nothing", history.search("   ") == [])

        removed = history.delete(chat_id)
        check("delete removes every message", removed == 3, str(removed))
        check("deleted discussion leaves the list",
              all(item["id"] != chat_id for item in history.list(limit=50)))
        check("deleted discussion leaves search",
              all(item["id"] != chat_id for item in history.search("vectorforge")))

        info = history.stats()
        check("stats counts the remaining discussion", info["chats"] == 1,
              str(info["chats"]))
        check("stats counts messages", info["messages"] == 2, str(info["messages"]))
        check("stats counts characters", info["chars"] > 0, str(info["chars"]))
        check("short titles collapse whitespace",
              rc.ChatHistory.short_title("  a   b  ") == "a b")
        check("short titles are bounded",
              len(rc.ChatHistory.short_title("word " * 60)) <= 75)
    finally:
        try:
            history.drop()
        except Exception:
            pass
        engine.close()

    failed = [label for label, ok, _ in checks if not ok]
    print()
    print("-" * 60)
    if failed:
        print(f"  {len(failed)} failed: {', '.join(failed)}")
        return 1
    print(f"  {len(checks)}/{len(checks)} checks passed")
    print("  discussions persist in Oracle and rebuild their answers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
