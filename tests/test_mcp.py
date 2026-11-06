"""The MCP server must answer over real stdio, not just import cleanly.

    python tests/test_mcp.py

Starts mcp_server.py as a child process, lists its tools, then calls each one.
The search calls load the local embedding and rerank models, so the first run
can take about a minute. No Gemini key and no API quota are used.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
except ImportError:
    print("mcp is not installed in this environment.")
    print(f'Run: "{sys.executable}" -m pip install -r requirements.txt')
    raise SystemExit(1)

checks: list[tuple[str, bool, str]] = []
QUESTION = "Which company raised an $85M Series B to build a hybrid search database?"
EXPECTED_SOURCE = "07_startup_vector_database.md"


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))


async def run_checks() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server.py")],
        cwd=str(ROOT),
    )
    async with Client(params) as client:
        listed = await client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        check("three tools are listed",
              set(tools) == {"search_corpus", "list_sources", "corpus_info"},
              ", ".join(sorted(tools)))
        check("tools are marked read-only",
              all(tool.annotations and tool.annotations.read_only_hint
                  for tool in tools.values()))

        info = await client.call_tool("corpus_info", {})
        payload = info.structured_content or {}
        check("corpus_info succeeds", not info.is_error)
        check("corpus_info reports rows", int(payload.get("rows") or 0) > 0,
              str(payload.get("rows")))
        check("corpus_info reports sources", int(payload.get("sources") or 0) > 0,
              str(payload.get("sources")))
        check("corpus_info reports a fingerprint",
              bool(payload.get("corpus_fingerprint")),
              str(payload.get("corpus_fingerprint")))
        check("corpus_info reports both indexes",
              {item.get("name") for item in payload.get("indexes", [])}
              >= {"TECH_NEWS_HNSW_IDX", "TECH_NEWS_TEXT_IDX"})

        sources = await client.call_tool("list_sources", {})
        source_payload = sources.structured_content or {}
        check("list_sources succeeds", not sources.is_error)
        check("list_sources count matches corpus_info",
              int(source_payload.get("count") or 0) == int(payload.get("sources") or 0),
              str(source_payload.get("count")))
        check("list_sources returns rows",
              bool(source_payload.get("sources")))

        search = await client.call_tool("search_corpus", {
            "query": QUESTION, "mode": "hybrid", "k": 3,
        })
        result = search.structured_content or {}
        hits = result.get("hits") or []
        check("hybrid search succeeds", not search.is_error)
        check("hybrid search returns k hits", len(hits) == 3, str(len(hits)))
        check("hybrid top hit is the expected source",
              bool(hits) and hits[0].get("source") == EXPECTED_SOURCE,
              hits[0].get("source") if hits else "none")
        check("hybrid top hit contains the answer",
              bool(hits) and "VectorForge" in hits[0].get("text", ""))
        check("hybrid hit carries its score and detail",
              bool(hits) and "score" in hits[0] and "detail" in hits[0])

        reranked = await client.call_tool("search_corpus", {
            "query": QUESTION, "mode": "rerank", "k": 3,
        })
        rerank_result = reranked.structured_content or {}
        rerank_hits = rerank_result.get("hits") or []
        check("rerank search succeeds", not reranked.is_error)
        check("rerank top hit is the expected source",
              bool(rerank_hits) and rerank_hits[0].get("source") == EXPECTED_SOURCE,
              rerank_hits[0].get("source") if rerank_hits else "none")
        check("rerank hits come from the cross-encoder",
              bool(rerank_hits) and all(hit.get("method") == "rerank"
                                        for hit in rerank_hits))
        check("rerank keeps the raw cross-encoder score",
              bool(rerank_hits) and "cross_encoder" in rerank_hits[0].get("detail", {}))

        bad_mode = await client.call_tool("search_corpus", {
            "query": QUESTION, "mode": "nonsense",
        })
        check("invalid mode returns a tool error", bad_mode.is_error)

        empty = await client.call_tool("search_corpus", {"query": "   "})
        check("empty query returns a tool error", empty.is_error)


def finish() -> int:
    failed = [label for label, ok, _ in checks if not ok]
    print()
    print("-" * 60)
    if failed:
        print(f"  {len(failed)} failed: {', '.join(failed)}")
        return 1
    print(f"  {len(checks)}/{len(checks)} checks passed")
    print("  MCP server speaks stdio and returns corpus results")
    return 0


def main() -> int:
    print("== mcp server over stdio ==")
    try:
        asyncio.run(run_checks())
    except Exception as exc:
        print(f"\n  [FAIL] server crashed: {type(exc).__name__}: {exc}")
        return 1
    return finish()


if __name__ == "__main__":
    sys.exit(main())
