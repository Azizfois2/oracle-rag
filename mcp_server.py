"""mcp_server.py - read-only MCP server over the Oracle RAG corpus.

    python mcp_server.py            # speak MCP on stdin/stdout
    python mcp_server.py --check    # print corpus info and exit

The server exposes three tools: search_corpus, list_sources, corpus_info. It
never writes to the database and never calls Gemini, so a query costs no quota.

MCP uses stdout for the protocol. Diagnostics go to stderr; do not print to
stdout from this process.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import sys
import threading
import time
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import rag_core as rc
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

LOG = logging.getLogger("oracle-rag-mcp")
MODES = ("keyword", "vector", "hybrid", "rerank")
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

SERVER = MCPServer(
    name="oracle-rag",
    title="Oracle RAG Corpus",
    version="1.0.0",
    instructions=(
        "Read-only access to a local Oracle 26ai corpus of ten AI news articles. "
        "Call search_corpus to retrieve passages, and cite the returned source. "
        "The default hybrid mode matches the app. Rerank mode uses a local "
        "cross-encoder for the best ordering. No Gemini call is made, so a query "
        "costs no API quota."
    ),
)

# RagEngine is cheap to construct. The connection and models load on first use.
_ENGINE = rc.RagEngine()
_LOCK = threading.RLock()


def _close() -> None:
    with _LOCK:
        _ENGINE.close()


atexit.register(_close)


def _as_tool_error(exc: Exception) -> ToolError:
    detail = " ".join(str(exc).split())
    if len(detail) > 320:
        detail = detail[:317] + "..."
    return ToolError(f"Oracle RAG: {detail or type(exc).__name__}")


def _hit_payload(hit: rc.Hit, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "source": hit.source,
        "title": hit.title,
        "score": round(float(hit.score), 4),
        "method": hit.method,
        "text": hit.text,
        "detail": {str(k): round(float(v), 4) for k, v in hit.detail.items()},
    }


@SERVER.tool(title="Search corpus", annotations=READ_ONLY)
def search_corpus(query: str, mode: str = "hybrid", k: int = 5,
                  weight: float = 0.6) -> dict[str, Any]:
    """Retrieve passages from the tech-news corpus.

    Args:
        query: Natural-language question or keywords.
        mode: keyword, vector, hybrid (default), or rerank.
        k: Number of passages to return, from 1 to 10.
        weight: Hybrid balance. 0 = keyword only, 1 = vector only.

    Scores rank results within this query. They are not confidence scores.
    """
    text = str(query or "").strip()
    if not text:
        raise ToolError("query must not be empty")

    method = str(mode or "hybrid").strip().lower()
    if method not in MODES:
        raise ToolError("mode must be one of: " + ", ".join(MODES))

    try:
        limit = max(1, min(10, int(k)))
        balance = max(0.0, min(1.0, float(weight)))
    except (TypeError, ValueError) as exc:
        raise ToolError("k must be a whole number and weight must be between 0 and 1") from exc

    with _LOCK:
        started = time.perf_counter()
        try:
            if method == "keyword":
                hits = _ENGINE.keyword_search(text, limit)
            elif method == "vector":
                hits = _ENGINE.vector_search(text, limit)
            elif method == "hybrid":
                hits = _ENGINE.hybrid_search(text, limit, balance)
            else:
                hits = _ENGINE.reranked_search(text, limit, balance)
            fingerprint = _ENGINE.corpus_fingerprint()
        except Exception as exc:
            LOG.exception("search failed")
            raise _as_tool_error(exc) from exc
        elapsed = (time.perf_counter() - started) * 1000

    return {
        "query": text,
        "mode": method,
        "k": limit,
        "weight": round(balance, 2),
        "elapsed_ms": round(elapsed),
        "corpus_fingerprint": fingerprint,
        "hits": [_hit_payload(hit, rank) for rank, hit in enumerate(hits, start=1)],
    }


@SERVER.tool(title="List sources", annotations=READ_ONLY)
def list_sources() -> dict[str, Any]:
    """List every indexed source with its chunk count and average size."""
    with _LOCK:
        try:
            sources = _ENGINE.sources()
            fingerprint = _ENGINE.corpus_fingerprint()
        except Exception as exc:
            LOG.exception("listing sources failed")
            raise _as_tool_error(exc) from exc
    return {
        "count": len(sources),
        "corpus_fingerprint": fingerprint,
        "sources": sources,
    }


@SERVER.tool(title="Corpus info", annotations=READ_ONLY)
def corpus_info() -> dict[str, Any]:
    """Show corpus size, database version, models, and index status."""
    with _LOCK:
        try:
            stats = _ENGINE.stats()
            fingerprint = _ENGINE.corpus_fingerprint()
        except Exception as exc:
            LOG.exception("reading corpus info failed")
            raise _as_tool_error(exc) from exc

    return {
        "table": stats["table"],
        "rows": stats["rows"],
        "sources": stats["sources"],
        "avg_chars": stats["avg_chars"],
        "max_chars": stats["max_chars"],
        "size_mb": stats["size_mb"],
        "database_version": stats["version"],
        "embedding_model": _ENGINE.s.embedding_model,
        "rerank_model": _ENGINE.s.rerank_model,
        "corpus_fingerprint": fingerprint,
        "indexes": [
            {"name": item["name"], "type": item["type"], "status": item["status"]}
            for item in stats.get("indexes", [])
            if item["name"].upper() in (rc.HNSW_INDEX.upper(), rc.TEXT_INDEX.upper())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only MCP server for the Oracle RAG corpus.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the database, print corpus info, and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub",
                  "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.check:
        try:
            info = corpus_info()
        except ToolError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    LOG.info("starting Oracle RAG MCP server on stdio")
    SERVER.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
