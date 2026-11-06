"""rag_core.py - the single contract for the Oracle RAG console.

Everything that talks to Oracle or Gemini lives here. ``app.py``, the eval
harness and the tests all call these functions; nothing else re-implements
retrieval, prompts or SQL.

Public surface
--------------
Settings.from_env()          configuration (env discovery, model candidates)
RagEngine                    connect, search, corpus management, stats
Agent                        ReAct agent over the corpus + local (no-LLM) mode
Quota                        per-model free-tier accounting, persisted
container_status()           Docker plumbing for the Health tab

Types: Hit, SearchBundle, Step, Answer.
"""

from __future__ import annotations

import array
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import oracledb
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TABLE = "TECH_NEWS_COLLECTION"
HNSW_INDEX = "TECH_NEWS_HNSW_IDX"
TEXT_INDEX = "TECH_NEWS_TEXT_IDX"
CONTAINER = "oracle-26ai-free"

# Free-tier quota is per model, so an exhausted model is not a dead end.
DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

PROMPT = (
    "You are a helpful AI research analyst. Use the search_tech_news tool "
    "to find information and answer questions about technology companies, "
    "AI products, partnerships, and industry trends. Always cite your "
    "sources. Be concise and factual."
)

ENV_CANDIDATES = (
    ROOT / ".env",
    ROOT.parent / ".env",                    # Desktop\.env (the notebook's file)
    ROOT.parent / "rag-oneclick" / ".env",
    Path.home() / ".env",
)

oracledb.defaults.program = "oracle-rag-console"


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class RagError(RuntimeError):
    """Base class for everything the UI should show as a friendly message."""


class DatabaseUnavailable(RagError):
    pass


class TableMissing(RagError):
    pass


class NoModel(RagError):
    """No LLM configured at all (no API key)."""


class QuotaExhausted(RagError):
    """Every candidate model refused this request for quota reasons."""


class AgentError(RagError):
    pass


def _first_line(exc: BaseException | str) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0][:300]


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def find_env() -> Path | None:
    for path in ENV_CANDIDATES:
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class Settings:
    user: str = "vector"
    password: str = "vector"
    dsn: str = "localhost:1521/FREEPDB1"
    table: str = TABLE
    embedding_model: str = EMBEDDING_MODEL
    rerank_model: str = RERANK_MODEL
    gemini_key: str | None = None
    gemini_model: str | None = None
    openai_key: str | None = None
    env_path: Path | None = None
    daily_limit: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        env = find_env()
        if env is not None:
            load_dotenv(env, override=False)

        raw_limit = os.getenv("GEMINI_DAILY_LIMIT", "20")
        try:
            limit = max(1, int(raw_limit))
        except ValueError:
            limit = 20

        key = (os.getenv("GEMINI_API_KEY") or "").strip()
        pinned = (os.getenv("GEMINI_MODEL") or "").strip()
        return cls(
            user=os.getenv("ORACLE_USER", "vector"),
            password=os.getenv("ORACLE_PWD", "vector"),
            dsn=os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1"),
            table=os.getenv("RAG_TABLE", TABLE),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL),
            rerank_model=os.getenv("RAG_RERANK_MODEL", RERANK_MODEL),
            gemini_key=key or None,
            gemini_model=pinned or None,
            openai_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
            env_path=env,
            daily_limit=limit,
        )

    # -- helpers ------------------------------------------------------------
    def models(self) -> list[str]:
        if self.gemini_model:
            return [self.gemini_model]
        return list(DEFAULT_MODELS)

    @property
    def has_llm(self) -> bool:
        return bool(self.gemini_key or self.openai_key)

    def snapshot(self) -> dict[str, Any]:
        """Safe-to-display view. The API key itself is never included."""
        return {
            "dsn": self.dsn,
            "user": self.user,
            "table": self.table,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "env_file": str(self.env_path) if self.env_path else None,
            "gemini_key": bool(self.gemini_key),
            "gemini_key_hint": (
                f"{self.gemini_key[:3]}…{len(self.gemini_key)} chars"
                if self.gemini_key
                else None
            ),
            "gemini_model_pinned": self.gemini_model,
            "openai_key": bool(self.openai_key),
            "daily_limit": self.daily_limit,
        }


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    key: str
    source: str
    text: str
    score: float
    method: str
    detail: dict[str, float] = field(default_factory=dict)

    def snippet(self, limit: int = 240) -> str:
        flat = " ".join(self.text.split())
        if len(flat) <= limit:
            return flat
        cut = flat[:limit].rsplit(" ", 1)[0]
        return cut + "…"

    @property
    def title(self) -> str:
        return self.source.rsplit(".", 1)[0].replace("_", " ")


@dataclass
class SearchBundle:
    query: str
    keyword: list[Hit]
    vector: list[Hit]
    hybrid: list[Hit]
    timings: dict[str, float]
    rerank: list[Hit] = field(default_factory=list)

    def all_keys(self) -> dict[str, set[str]]:
        return {
            "keyword": {h.key for h in self.keyword},
            "vector": {h.key for h in self.vector},
            "hybrid": {h.key for h in self.hybrid},
            "rerank": {h.key for h in self.rerank},
        }

    def consensus(self) -> list[str]:
        found = self.all_keys()
        keys = set().union(*found.values())
        return [k for k in keys if sum(k in v for v in found.values()) >= 2]

    @property
    def total(self) -> int:
        return len(set().union(*self.all_keys().values()))


@dataclass
class Step:
    kind: str            # search | sources | answer | note
    label: str
    detail: str = ""


@dataclass
class Answer:
    text: str
    model: str
    steps: list[Step] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    elapsed: float = 0.0
    local: bool = False
    hits: list[Hit] = field(default_factory=list)
    cached: bool = False


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def pacific_today() -> str:
    """Free-tier quota resets at midnight Pacific."""
    try:
        from zoneinfo import ZoneInfo

        return dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        now = dt.datetime.now(dt.timezone.utc)
        return (now - dt.timedelta(hours=8)).strftime("%Y-%m-%d")


def pacific_reset_hint() -> str:
    return "quota resets at midnight Pacific"


def _as_text(value: Any) -> str:
    """CLOBs arrive as LOB objects; strings and None arrive as themselves."""
    if value is None:
        return ""
    if hasattr(value, "read"):
        try:
            return value.read()
        except Exception:
            return str(value)
    return str(value)


def _as_key(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)


def _meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _jsonable(value: Any) -> Any:
    """Turn Oracle JSON values into plain JSON values.

    Oracle returns JSON numbers as ``Decimal``, which ``json.dumps`` refuses.
    Exporting the corpus must stay plain JSON so a backup can be written to
    disk without a custom encoder.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _source(value: Any) -> str:
    return str(_meta(value).get("source") or "unknown")


def _terms(query: str, limit: int = 6) -> list[str]:
    """Words usable in an Oracle Text expression. Keeps it predictable."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_+#.-]{1,}", query)
    stop = {
        "the", "and", "for", "with", "what", "which", "who", "how", "are", "was",
        "were", "does", "did", "about", "from", "that", "this", "into", "their",
        "have", "has", "its", "they", "you", "tell", "give", "explain", "why",
    }
    picked: list[str] = []
    for word in words:
        if word.lower() in stop:
            continue
        if word.lower() not in (w.lower() for w in picked):
            picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def message_text(message: Any) -> str:
    """Gemini returns AIMessage.content as a list of blocks, not a string."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


_SOURCE_LINE = re.compile(r"\[Source:\s*(?P<src>[^|\]]+?)\s*\|\s*Relevance:\s*(?P<score>[0-9.]+)\s*\]")


def sources_from_tool(text: str) -> list[dict[str, Any]]:
    out = []
    for match in _SOURCE_LINE.finditer(text or ""):
        try:
            score = float(match.group("score"))
        except ValueError:
            score = 0.0
        out.append({"source": match.group("src").strip(), "relevance": score})
    return out


def extract_text(filename: str, data: bytes) -> str:
    """Turn an uploaded file into plain text. PDF support is optional."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:                     # pragma: no cover
            raise RagError("PDF support needs pypdf: pip install pypdf") from exc
        import io

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8-sig", errors="replace")


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(overlap, max(0, size // 3)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def container_status(name: str = CONTAINER) -> dict[str, Any]:
    """Read Docker state without ever raising. The UI degrades, it does not die."""
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f",
             "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.Config.Image}}",
             name],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        return {"name": name, "state": "no-docker", "health": "", "image": "", "note": "docker not on PATH"}
    except Exception as exc:                            # pragma: no cover
        return {"name": name, "state": "unknown", "health": "", "image": "", "note": _first_line(exc)}

    if proc.returncode != 0:
        return {"name": name, "state": "missing", "health": "", "image": "",
                "note": "container not found"}

    state, health, image = (proc.stdout.strip().split("|") + ["", "", ""])[:3]
    return {"name": name, "state": state or "unknown", "health": health or "none", "image": image}


# --------------------------------------------------------------------------- #
# quota + model memory
# --------------------------------------------------------------------------- #
class Quota:
    """Per-model free-tier accounting, persisted to data/quota.json."""

    def __init__(self, path: Path | None = None, limit: int = 20) -> None:
        self.path = path or DATA_DIR / "quota.json"
        self.limit = limit

    # -- storage ------------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        if data.get("date") != pacific_today():
            data = {"date": pacific_today(), "models": {}}
        data.setdefault("models", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass  # accounting must never break a request

    # -- api ----------------------------------------------------------------
    def record(self, model: str, calls: int = 1) -> None:
        data = self._read()
        entry = data["models"].setdefault(model, {"calls": 0, "exhausted": False})
        entry["calls"] = int(entry.get("calls", 0)) + calls
        self._write(data)

    def mark_exhausted(self, model: str, reason: str = "") -> None:
        data = self._read()
        entry = data["models"].setdefault(model, {"calls": 0, "exhausted": False})
        entry["exhausted"] = True
        entry["reason"] = reason[:200]
        self._write(data)

    def is_exhausted(self, model: str) -> bool:
        return bool(self._read()["models"].get(model, {}).get("exhausted"))

    def calls(self, model: str) -> int:
        return int(self._read()["models"].get(model, {}).get("calls", 0))

    def status(self) -> dict[str, Any]:
        data = self._read()
        models = {}
        for name in list(data["models"].keys()):
            entry = data["models"][name]
            models[name] = {
                "calls": int(entry.get("calls", 0)),
                "exhausted": bool(entry.get("exhausted")),
                "reason": entry.get("reason", ""),
            }
        spent = sum(m["calls"] for m in models.values())
        return {
            "date": data["date"],
            "limit": self.limit,
            "models": models,
            "spent": spent,
            "budget": self.limit * max(1, len(DEFAULT_MODELS)),
        }

    def clear(self) -> None:
        self._write({"date": pacific_today(), "models": {}})


class ModelMemory:
    """Remembers the last model that worked, so the app stops probing from the top."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_DIR / "model.json"

    def get(self) -> str | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("model") or None
        except Exception:
            return None

    def set(self, model: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"model": model, "at": dt.datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# answer cache
# --------------------------------------------------------------------------- #
def hit_to_dict(hit: Hit) -> dict[str, Any]:
    return {
        "key": hit.key,
        "source": hit.source,
        "text": hit.text,
        "score": hit.score,
        "method": hit.method,
        "detail": dict(hit.detail),
    }


def hit_from_dict(data: dict[str, Any]) -> Hit:
    return Hit(
        key=str(data.get("key", "")),
        source=str(data.get("source", "unknown")),
        text=str(data.get("text", "")),
        score=float(data.get("score", 0.0) or 0.0),
        method=str(data.get("method", "hybrid")),
        detail={str(k): float(v) for k, v in (data.get("detail") or {}).items()},
    )


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    return {
        "text": answer.text,
        "model": answer.model,
        "steps": [{"kind": s.kind, "label": s.label, "detail": s.detail} for s in answer.steps],
        "sources": list(answer.sources),
        "calls": answer.calls,
        "elapsed": answer.elapsed,
        "local": answer.local,
        "hits": [hit_to_dict(hit) for hit in answer.hits],
    }


def answer_from_dict(data: dict[str, Any]) -> Answer:
    steps = [
        Step(
            kind=str(step.get("kind", "note")),
            label=str(step.get("label", "")),
            detail=str(step.get("detail", "")),
        )
        for step in data.get("steps") or []
        if isinstance(step, dict)
    ]
    sources = [dict(item) for item in data.get("sources") or [] if isinstance(item, dict)]
    hits = [hit_from_dict(item) for item in data.get("hits") or [] if isinstance(item, dict)]
    return Answer(
        text=str(data.get("text", "")),
        model=str(data.get("model", "unknown")),
        steps=steps,
        sources=sources,
        calls=int(data.get("calls", 0) or 0),
        elapsed=float(data.get("elapsed", 0.0) or 0.0),
        local=bool(data.get("local")),
        hits=hits,
    )


class AnswerCache:
    """Answers remembered in data/answers.json, keyed by question and corpus.

    The key contains a fingerprint of the corpus (row count plus a hash of the
    ids), so adding or removing documents cannot serve an answer that predates
    them. The file is plain JSON: inspect it, copy it, delete it.
    """

    VERSION = 1
    MAX_ENTRIES = 200

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_DIR / "answers.json"

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict) or data.get("version") != self.VERSION:
            return {"version": self.VERSION, "entries": {}}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            entries = {}
        return {"version": self.VERSION, "entries": entries}

    def _write(self, data: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass  # a cache that breaks a request is worse than a cold cache

    @staticmethod
    def key(question: str, corpus: str, mode: str = "agent",
            extra: dict[str, Any] | None = None) -> str:
        payload = {
            "q": " ".join((question or "").lower().split()),
            "corpus": corpus,
            "mode": mode,
            "extra": extra or {},
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, key: str) -> Answer | None:
        entry = self._read()["entries"].get(key)
        if not isinstance(entry, dict):
            return None
        try:
            answer = answer_from_dict(entry.get("answer") or {})
        except Exception:
            return None
        answer.cached = True
        answer.elapsed = 0.0
        return answer

    def put(self, key: str, answer: Answer, question: str, corpus: str) -> None:
        data = self._read()
        entries = data["entries"]
        entries[key] = {
            "question": question,
            "at": dt.datetime.now().isoformat(timespec="seconds"),
            "corpus": corpus,
            "answer": answer_to_dict(answer),
        }
        while len(entries) > self.MAX_ENTRIES:
            oldest = min(entries, key=lambda item: str(entries[item].get("at", "")))
            entries.pop(oldest, None)
        self._write(data)

    def stats(self) -> dict[str, Any]:
        entries = self._read()["entries"]
        try:
            size_kb = round(self.path.stat().st_size / 1024, 1)
        except Exception:
            size_kb = 0.0
        return {"entries": len(entries), "size_kb": size_kb, "path": self.path}

    def clear(self) -> int:
        count = len(self._read()["entries"])
        self._write({"version": self.VERSION, "entries": {}})
        return count


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class RagEngine:
    """One connection, one embedding model, three paths plus an optional reranker."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings.from_env()
        self._conn: oracledb.Connection | None = None
        self._vs: Any = None
        self._emb: Any = None
        self._ce: Any = None
        self._vec_cache: dict[str, array.array] = {}
        self._emb_lock = threading.Lock()
        self._ce_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> oracledb.Connection:
        self.close()
        try:
            conn = oracledb.connect(user=self.s.user, password=self.s.password, dsn=self.s.dsn)
        except oracledb.Error as exc:
            raise DatabaseUnavailable(_first_line(exc)) from exc
        except Exception as exc:                        # pragma: no cover
            raise DatabaseUnavailable(_first_line(exc)) from exc
        conn.autocommit = False
        self._conn = conn
        return conn

    def ensure(self) -> oracledb.Connection:
        if self._conn is None:
            return self.connect()
        try:
            self._conn.ping()
        except Exception:
            return self.connect()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._vs = None

    @property
    def embeddings(self) -> Any:
        if self._emb is None:
            with self._emb_lock:
                if self._emb is None:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._emb = HuggingFaceEmbeddings(model_name=self.s.embedding_model)
        return self._emb

    @property
    def reranker(self) -> Any:
        """The local cross-encoder. It is loaded on first use, never at import."""
        if self._ce is None:
            with self._ce_lock:
                if self._ce is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._ce = CrossEncoder(self.s.rerank_model)
                    except Exception as exc:
                        raise RagError(f"reranker unavailable: {_first_line(exc)}") from exc
        return self._ce

    def warm_reranker(self) -> None:
        _ = self.reranker

    def rerank_hits(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
        """Reorder hybrid candidates with a local cross-encoder.

        The score is min-max normalised over this candidate pool so the score
        bar stays readable; the raw logit is kept in ``detail``. This is a
        local model, so it costs no API quota.
        """
        candidates = list(hits)
        if not candidates:
            return []
        raw = [float(score) for score in self.reranker.predict(
            [(query, hit.text) for hit in candidates], show_progress_bar=False)]
        low, high = min(raw), max(raw)
        span = high - low
        ranked: list[Hit] = []
        for hit, score in zip(candidates, raw):
            shown = 1.0 if span < 1e-9 else (score - low) / span
            ranked.append(
                Hit(
                    key=hit.key,
                    source=hit.source,
                    text=hit.text,
                    score=round(shown, 4),
                    method="rerank",
                    detail={"hybrid": hit.score, "cross_encoder": round(score, 4)},
                )
            )
        ranked.sort(key=lambda h: (-h.score, h.key))
        return ranked[:k]

    def reranked_search(self, query: str, k: int = 5, weight: float = 0.6) -> list[Hit]:
        """Hybrid retrieval over a wider pool, then cross-encoder reranking."""
        self._require_table()
        pool = self.hybrid_search(query, max(k * 6, 30), weight)
        return self.rerank_hits(query, pool, k)

    @property
    def vector_store(self) -> Any:
        conn = self.ensure()
        if self._vs is None:
            from langchain_oracledb import OracleVS
            from langchain_oracledb.vectorstores.oraclevs import DistanceStrategy

            self._vs = OracleVS(
                client=conn,
                embedding_function=self.embeddings,
                table_name=self.s.table,
                distance_strategy=DistanceStrategy.COSINE,
            )
        return self._vs

    # -- introspection ------------------------------------------------------
    def table_exists(self) -> bool:
        cur = self.ensure().cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM user_tables WHERE table_name = :1", [self.s.table.upper()])
            return int(cur.fetchone()[0]) > 0
        finally:
            cur.close()

    def _require_table(self) -> None:
        if not self.table_exists():
            raise TableMissing(
                f"table {self.s.table} does not exist yet - ingest a document in the "
                "Corpus tab, or run the notebook once"
            )

    def indexes(self) -> list[dict[str, str]]:
        cur = self.ensure().cursor()
        try:
            cur.execute(
                "SELECT index_name, index_type, status FROM user_indexes WHERE table_name = :1",
                [self.s.table.upper()],
            )
            return [
                {"name": str(name), "type": str(kind), "status": str(status)}
                for name, kind, status in cur.fetchall()
            ]
        finally:
            cur.close()

    def stats(self) -> dict[str, Any]:
        conn = self.ensure()
        out: dict[str, Any] = {
            "table": self.s.table,
            "rows": 0,
            "sources": 0,
            "avg_chars": 0,
            "max_chars": 0,
            "size_mb": 0.0,
            "version": "?",
            "indexes": [],
            "columns": {},
            "embedding_model": self.s.embedding_model,
        }
        cur = conn.cursor()
        try:
            cur.execute("SELECT banner FROM v$version WHERE ROWNUM = 1")
            row = cur.fetchone()
            out["version"] = str(row[0]) if row else "?"

            cur.execute(
                "SELECT column_name, data_type FROM user_tab_columns "
                "WHERE table_name = :1 ORDER BY column_id",
                [self.s.table.upper()],
            )
            out["columns"] = {str(name): str(kind) for name, kind in cur.fetchall()}

            if out["columns"]:
                cur.execute(
                    f"SELECT COUNT(*), COUNT(DISTINCT JSON_VALUE(metadata, '$.source')), "
                    f"ROUND(AVG(LENGTH(text))), MAX(LENGTH(text)) FROM {self.s.table}"
                )
                rows, sources, avg, biggest = cur.fetchone()
                out["rows"] = int(rows or 0)
                out["sources"] = int(sources or 0)
                out["avg_chars"] = int(avg or 0)
                out["max_chars"] = int(biggest or 0)

                cur.execute(
                    "SELECT ROUND(SUM(bytes)/1024/1024, 2) FROM user_segments WHERE segment_name = :1",
                    [self.s.table.upper()],
                )
                size = cur.fetchone()
                out["size_mb"] = float(size[0] or 0) if size else 0.0
                out["indexes"] = self.indexes()
        finally:
            cur.close()
        return out

    def sources(self) -> list[dict[str, Any]]:
        if not self.table_exists():
            return []
        cur = self.ensure().cursor()
        try:
            cur.execute(
                f"SELECT JSON_VALUE(metadata, '$.source') AS src, COUNT(*) AS chunks, "
                f"ROUND(AVG(DBMS_LOB.GETLENGTH(text))) AS avg_bytes "
                f"FROM {self.s.table} GROUP BY JSON_VALUE(metadata, '$.source') "
                f"ORDER BY src"
            )
            return [
                {"source": str(src or "unknown"), "chunks": int(chunks), "avg_chars": int(avg or 0)}
                for src, chunks, avg in cur.fetchall()
            ]
        finally:
            cur.close()

    def export_corpus(self) -> dict[str, Any]:
        """Every chunk as plain JSON, so a backup survives the volume.

        Embeddings are not exported. They are recomputed from the text on
        import, which keeps the file small and portable across embedding models.
        """
        self._require_table()
        cur = self.ensure().cursor()
        try:
            cur.execute(
                f"SELECT JSON_VALUE(metadata, '$.source') AS src, text, metadata "
                f"FROM {self.s.table} ORDER BY src, id"
            )
            rows = cur.fetchall()
        finally:
            cur.close()

        documents = []
        for src, text, meta in rows:
            metadata = _meta(meta)
            source = str(src or metadata.get("source") or "unknown")
            extra = {
                k: v for k, v in metadata.items()
                if k not in ("source", "__orcl_internal_doc_id")
            }
            documents.append({"source": source, "text": _as_text(text),
                              "metadata": _jsonable(extra)})
        return {
            "format": "oracle-rag-corpus",
            "version": 1,
            "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
            "table": self.s.table,
            "embedding_model": self.s.embedding_model,
            "documents": documents,
        }

    def import_documents(self, documents: Iterable[dict[str, Any]],
                         replace: bool = False) -> dict[str, Any]:
        """Write exported chunks back, re-embedding once per source.

        Existing sources are skipped unless replace=True, so importing the same
        backup twice cannot silently duplicate a source. Each source is written
        with one add_documents call, which keeps its chunk boundaries intact.
        """
        from langchain_core.documents import Document

        grouped: dict[str, list[Document]] = {}
        skipped = 0
        for item in documents:
            if not isinstance(item, dict):
                skipped += 1
                continue
            source = str(item.get("source") or "unknown").strip()
            text = str(item.get("text") or "").strip()
            if not source or not text:
                skipped += 1
                continue
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata.pop("__orcl_internal_doc_id", None)
            metadata["source"] = source
            grouped.setdefault(source, []).append(Document(page_content=text, metadata=metadata))

        if not grouped:
            raise RagError("nothing to import: no documents with text and a source")

        added = 0
        replaced = 0
        for source, docs in grouped.items():
            if self.source_exists(source):
                if not replace:
                    skipped += len(docs)
                    continue
                self.delete_source(source)
                replaced += 1
            self.vector_store.add_documents(docs)
            added += len(docs)
        self.ensure().commit()
        self._vec_cache.clear()
        return {
            "sources": len(grouped),
            "chunks": added,
            "replaced": replaced,
            "skipped": skipped,
        }

    def corpus_fingerprint(self) -> str | None:
        """Cheap identity of the current corpus, used to key cached answers."""
        try:
            if not self.table_exists():
                return None
            cur = self.ensure().cursor()
            try:
                cur.execute(
                    f"SELECT COUNT(*), SUM(ORA_HASH(RAWTOHEX(id), 1000000)) FROM {self.s.table}"
                )
                rows, digest = cur.fetchone()
            finally:
                cur.close()
            return f"{int(rows or 0)}:{int(digest or 0)}"
        except Exception:
            return None

    # -- retrieval ----------------------------------------------------------
    def _query_vector(self, query: str) -> array.array:
        cached = self._vec_cache.get(query)
        if cached is not None:
            return cached
        vec = array.array("f", self.embeddings.embed_query(query))
        if len(self._vec_cache) > 16:
            self._vec_cache.clear()
        self._vec_cache[query] = vec
        return vec

    def _vector_pool(self, query: str, pool: int) -> list[Hit]:
        limit = max(1, int(pool))
        cur = self.ensure().cursor()
        try:
            cur.execute(
                f"""
                SELECT id, text, metadata,
                       1 - VECTOR_DISTANCE(embedding, :q, COSINE) AS sim
                FROM {self.s.table}
                ORDER BY VECTOR_DISTANCE(embedding, :q, COSINE)
                FETCH APPROX FIRST {limit} ROWS ONLY
                """,
                {"q": self._query_vector(query)},
            )
            rows = cur.fetchall()
        finally:
            cur.close()

        hits = []
        for rid, text, meta, sim in rows:
            score = max(0.0, min(1.0, float(sim or 0.0)))
            hits.append(Hit(_as_key(rid), _source(meta), _as_text(text), score, "vector"))
        hits.sort(key=lambda h: (-h.score, h.key))
        return hits

    def vector_search(self, query: str, k: int = 5) -> list[Hit]:
        self._require_table()
        return self._vector_pool(query, max(k * 4, 20))[:k]

    def _keyword_pool(self, query: str, pool: int) -> list[Hit]:
        terms = _terms(query)
        if not terms:
            return []
        expr = " OR ".join(terms)
        cur = self.ensure().cursor()
        rows: list[tuple[Any, Any, Any, float]] = []
        try:
            try:
                cur.execute(
                    f"""
                    SELECT id, text, metadata, SCORE(1) AS kw
                    FROM {self.s.table}
                    WHERE CONTAINS(text, :expr, 1) > 0
                    ORDER BY kw DESC, id
                    FETCH FIRST {int(pool)} ROWS ONLY
                    """,
                    {"expr": expr},
                )
                fetched = cur.fetchall()
                rows = [(r[0], r[1], r[2], float(r[3] or 0.0)) for r in fetched]
            except oracledb.Error:
                # Oracle Text refused the expression (or the index is missing).
                # Fall back to a substring scan so the keyword column never lies.
                rows = self._keyword_fallback(cur, terms, pool)
        finally:
            cur.close()

        if not rows:
            return []
        best = max(r[3] for r in rows) or 1.0
        hits = [
            Hit(_as_key(rid), _source(meta), _as_text(text), round(min(1.0, score / best), 4), "keyword")
            for rid, text, meta, score in rows
        ]
        hits.sort(key=lambda h: (-h.score, h.key))
        return hits

    def _keyword_fallback(self, cur: Any, terms: Sequence[str], pool: int) -> list[tuple[Any, Any, Any, float]]:
        found: dict[str, tuple[Any, Any, Any, float]] = {}
        for term in terms[:4]:
            try:
                cur.execute(
                    f"SELECT id, text, metadata FROM {self.s.table} "
                    "WHERE DBMS_LOB.INSTR(UPPER(text), :w) > 0 "
                    f"ORDER BY id FETCH FIRST {int(pool)} ROWS ONLY",
                    {"w": term.upper()},
                )
            except oracledb.Error:
                continue
            for rid, text, meta in cur.fetchall():
                key = _as_key(rid)
                body = _as_text(text).upper()
                score = sum(1 for t in terms if t.upper() in body)
                previous = found.get(key)
                if previous is None or score > previous[3]:
                    found[key] = (rid, text, meta, float(score))
        return list(found.values())

    def keyword_search(self, query: str, k: int = 5) -> list[Hit]:
        self._require_table()
        return self._keyword_pool(query, max(k * 4, 20))[:k]

    def hybrid_search(self, query: str, k: int = 5, weight: float = 0.6) -> list[Hit]:
        self._require_table()
        weight = max(0.0, min(1.0, float(weight)))
        pool = max(k * 4, 20)
        vec = {h.key: h for h in self._vector_pool(query, pool)}
        kw = {h.key: h for h in self._keyword_pool(query, pool)}

        # Cosine similarity lives in a narrow band (~0.2-0.6 here) while keyword
        # scores are already 0-1, so a raw weighted sum would silently favour
        # keyword. Min-max each component over the candidate pool: now the slider
        # means what it says, and w=0 / w=1 reproduce the pure rankings exactly.
        def normalised(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}
            low, high = min(scores.values()), max(scores.values())
            if high - low < 1e-9:
                return {key: 1.0 for key in scores}
            return {key: (value - low) / (high - low) for key, value in scores.items()}

        vec_norm = normalised({key: hit.score for key, hit in vec.items()})
        kw_norm = normalised({key: hit.score for key, hit in kw.items()})

        # At the extremes the slider must be exactly the pure path, not "roughly".
        if weight <= 0.0:
            order = list(kw.keys())
        elif weight >= 1.0:
            order = list(vec.keys())
        else:
            order = list(vec.keys()) + [key for key in kw if key not in vec]

        fused: list[Hit] = []
        for key in order:
            base = vec.get(key) or kw[key]
            v_raw = vec[key].score if key in vec else 0.0
            k_raw = kw[key].score if key in kw else 0.0
            score = weight * vec_norm.get(key, 0.0) + (1 - weight) * kw_norm.get(key, 0.0)
            fused.append(
                Hit(
                    key=key,
                    source=base.source,
                    text=base.text,
                    score=round(score, 4),
                    method="hybrid",
                    detail={"vector": round(v_raw, 4), "keyword": round(k_raw, 4)},
                )
            )
        fused.sort(key=lambda h: (-h.score, h.key))
        return fused[:k]

    def warm(self) -> None:
        """Load the embedding model now, so the first real query is not 60 s slow."""
        _ = self.embeddings

    def warm_async(self) -> None:
        """Load in the background: the UI renders immediately, search is ready shortly."""
        if self._emb is None:
            threading.Thread(target=self.warm, name="embed-warm", daemon=True).start()

    def search_all(self, query: str, k: int = 5, weight: float = 0.6,
                   rerank: bool = False) -> SearchBundle:
        self._require_table()
        _ = self.embeddings          # never charge model loading to a search's latency
        self._query_vector(query)    # embed once; the three paths share the vector
        timings: dict[str, float] = {}

        start = time.perf_counter()
        keyword = self.keyword_search(query, k)
        timings["keyword"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        vector = self.vector_search(query, k)
        timings["vector"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        hybrid = self.hybrid_search(query, k, weight)
        timings["hybrid"] = (time.perf_counter() - start) * 1000

        reranked: list[Hit] = []
        if rerank:
            _ = self.reranker        # load once, outside the timed rerank stage
            start = time.perf_counter()
            candidates = self.hybrid_search(query, max(k * 6, 30), weight)
            reranked = self.rerank_hits(query, candidates, k)
            timings["rerank"] = (time.perf_counter() - start) * 1000

        return SearchBundle(query=query, keyword=keyword, vector=vector,
                            hybrid=hybrid, timings=timings, rerank=reranked)

    # -- corpus -------------------------------------------------------------
    def source_exists(self, source: str) -> bool:
        if not self.table_exists():
            return False
        cur = self.ensure().cursor()
        try:
            cur.execute(
                f"SELECT COUNT(*) FROM {self.s.table} WHERE JSON_VALUE(metadata, '$.source') = :s",
                {"s": source},
            )
            return int(cur.fetchone()[0]) > 0
        finally:
            cur.close()

    def ingest(self, text: str, source: str, chunk_size: int = 900,
               overlap: int = 120, replace: bool = False) -> dict[str, Any]:
        body = (text or "").strip()
        if not body:
            raise RagError("nothing to ingest: the text is empty")
        if self.source_exists(source):
            if not replace:
                raise RagError(f"source already exists: {source}")
            self.delete_source(source)

        chunks = chunk_text(body, chunk_size, overlap)
        if not chunks:
            raise RagError("nothing to ingest: chunking produced no text")

        from langchain_core.documents import Document

        docs = [Document(page_content=c, metadata={"source": source}) for c in chunks]
        self.vector_store.add_documents(docs)
        self.ensure().commit()
        self._vec_cache.clear()
        return {"source": source, "chunks": len(chunks), "chars": len(body)}

    def delete_source(self, source: str) -> int:
        if not self.table_exists():
            return 0
        conn = self.ensure()
        cur = conn.cursor()
        try:
            cur.execute(
                f"DELETE FROM {self.s.table} WHERE JSON_VALUE(metadata, '$.source') = :s",
                {"s": source},
            )
            removed = int(cur.rowcount or 0)
            conn.commit()
            self._vec_cache.clear()
            return removed
        finally:
            cur.close()

    def rebuild_text_index(self) -> None:
        """Rebuild the Oracle Text index so keyword scores have a clean start.

        Oracle Text scoring uses index-level statistics. Inserts and deletes
        change those statistics, so two runs over the same rows can rank keyword
        hits differently. The benchmark rebuilds first; no data is changed.
        """
        self._require_table()
        if not any(i["name"].upper() == TEXT_INDEX.upper() for i in self.indexes()):
            self.reindex()
            return
        conn = self.ensure()
        cur = conn.cursor()
        try:
            cur.execute(f"ALTER INDEX {TEXT_INDEX} REBUILD")
            conn.commit()
        finally:
            cur.close()

    def reindex(self) -> list[str]:
        """Rebuild whichever index went missing. Never drops data."""
        self._require_table()
        report: list[str] = []
        present = {i["name"].upper(): i for i in self.indexes()}
        conn = self.ensure()

        if HNSW_INDEX.upper() not in present:
            from langchain_oracledb.vectorstores.oraclevs import create_index

            create_index(
                conn,
                self.vector_store,
                params={"idx_name": HNSW_INDEX, "idx_type": "HNSW", "accuracy": 95, "parallel": 4},
            )
            conn.commit()
            report.append(f"created {HNSW_INDEX} (HNSW)")
        else:
            report.append(f"{HNSW_INDEX} present ({present[HNSW_INDEX.upper()]['status']})")

        text_index = present.get(TEXT_INDEX.upper())
        if text_index is None:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"CREATE INDEX {TEXT_INDEX} ON {self.s.table}(text) "
                    "INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('SYNC (ON COMMIT)')"
                )
                conn.commit()
                report.append(f"created {TEXT_INDEX} (Oracle Text)")
            except oracledb.Error as exc:
                report.append(f"{TEXT_INDEX} could not be created: {_first_line(exc)}")
            finally:
                cur.close()
        else:
            cur = conn.cursor()
            try:
                cur.execute(f"BEGIN CTX_DDL.SYNC_INDEX('{TEXT_INDEX}'); END;")
                conn.commit()
                report.append(f"{TEXT_INDEX} synced")
            except oracledb.Error:
                report.append(f"{TEXT_INDEX} present ({text_index['status']})")
            finally:
                cur.close()

        return report

    def ping(self) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            conn = self.ensure()
            cur = conn.cursor()
            try:
                cur.execute("SELECT banner FROM v$version WHERE ROWNUM = 1")
                row = cur.fetchone()
            finally:
                cur.close()
            return {
                "ok": True,
                "version": str(row[0]) if row else "?",
                "ms": round((time.perf_counter() - start) * 1000, 1),
            }
        except Exception as exc:
            return {"ok": False, "error": _first_line(exc), "ms": round((time.perf_counter() - start) * 1000, 1)}


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #
class _CountCalls(BaseCallbackHandler):
    """LangChain callback: count every model invocation against the quota.

    Inherits BaseCallbackHandler for the flags the callback manager probes
    (raise_error, ignore_chain, ignore_llm, run_inline) and the no-op hooks.
    """

    def __init__(self, quota: Quota, model: str) -> None:
        super().__init__()
        self.quota = quota
        self.model = model
        self.calls = 0

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.calls += 1
        self.quota.record(self.model)


def build_search_tool(engine: RagEngine) -> Any:
    from langchain_core.tools import tool

    @tool
    def search_tech_news(query: str) -> str:
        """Search the tech news knowledge base for information about AI companies,
        products, partnerships, and technology trends. Use this tool whenever
        you need to find specific facts or details about technology topics."""
        hits = engine.vector_search(query, k=4)
        if not hits:
            return "No relevant documents found."
        return "\n\n---\n\n".join(
            f"[Source: {h.source} | Relevance: {h.score:.2f}]\n{h.text}" for h in hits
        )

    return search_tech_news


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()


def _is_unavailable_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "404" in text
        or "NOT_FOUND" in text
        or "no longer available" in text.lower()
        or "is not found" in text.lower()
    )


_TRANSIENT = re.compile(
    r"^\s*(500|502|503|504)\b|\b(500|502|503|504)\s+(UNAVAILABLE|INTERNAL|BAD_GATEWAY|DEADLINE)",
    re.IGNORECASE,
)


def _is_transient_error(exc: BaseException) -> bool:
    """Server-side hiccups that are worth another model, but are not quota."""
    text = str(exc)
    if _TRANSIENT.search(text):
        return True
    return "UNAVAILABLE" in text or "currently experiencing high demand" in text.lower()


class Agent:
    """ReAct agent over the corpus, with quota-aware model fallback."""

    def __init__(self, engine: RagEngine, settings: Settings | None = None,
                 quota: Quota | None = None, cache: AnswerCache | None = None) -> None:
        self.engine = engine
        self.s = settings or engine.s
        self.quota = quota or Quota(limit=self.s.daily_limit)
        self.cache = cache or AnswerCache()
        self.memory = ModelMemory()
        self._graphs: dict[str, Any] = {}

    # -- models -------------------------------------------------------------
    def model_order(self) -> list[str]:
        candidates = self.s.models()
        last = self.memory.get()
        if last and last in candidates:
            candidates = [last] + [m for m in candidates if m != last]
        return candidates

    def usable_models(self) -> list[str]:
        return [m for m in self.model_order() if not self.quota.is_exhausted(m)]

    def current_model(self) -> str | None:
        usable = self.usable_models()
        return usable[0] if usable else None

    def _llm(self, model: str) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=self.s.gemini_key,
            max_retries=0,
            timeout=60,
        )

    def _graph(self, model: str) -> Any:
        if model not in self._graphs:
            from langgraph.prebuilt import create_react_agent

            self._graphs[model] = create_react_agent(
                self._llm(model),
                tools=[build_search_tool(self.engine)],
                prompt=PROMPT,
            )
        return self._graphs[model]

    # -- ask ----------------------------------------------------------------
    def ask(self, question: str, on_step: Callable[[str, str], None] | None = None,
            recursion_limit: int = 25, use_cache: bool = True) -> Answer:
        question = (question or "").strip()
        if not question:
            raise RagError("ask something first")

        def emit(kind: str, text: str) -> None:
            if on_step is not None:
                try:
                    on_step(kind, text)
                except Exception:
                    pass

        # A cached answer is served before the key check, so it still works on a
        # machine whose quota is spent or whose key has been removed.
        cache_key: str | None = None
        corpus: str | None = None
        if use_cache:
            corpus = self.engine.corpus_fingerprint()
            if corpus:
                cache_key = self.cache.key(question, corpus)
                cached = self.cache.get(cache_key)
                if cached is not None:
                    emit("note", "cached answer, no model call")
                    return cached

        if not self.s.gemini_key:
            raise NoModel("No GEMINI_API_KEY configured - use local mode, or add one to .env")

        usable = self.usable_models()
        if not usable:
            raise QuotaExhausted(
                f"every Gemini model is out of free-tier quota for today - {pacific_reset_hint()}"
            )

        errors: list[str] = []
        for model in usable:
            counter = _CountCalls(self.quota, model)
            steps: list[Step] = []
            sources: list[dict[str, Any]] = []
            started = time.perf_counter()
            try:
                final: str | None = None
                for update in self._graph(model).stream(
                    {"messages": [("user", question)]},
                    config={"callbacks": [counter], "recursion_limit": recursion_limit},
                    stream_mode="updates",
                ):
                    if not isinstance(update, dict):
                        continue
                    for _node, payload in update.items():
                        messages = payload.get("messages", []) if isinstance(payload, dict) else []
                        for msg in messages:
                            kind = getattr(msg, "type", "")
                            tool_calls = getattr(msg, "tool_calls", None)
                            if kind == "ai" and tool_calls:
                                for call in tool_calls:
                                    q = str((call.get("args") or {}).get("query", ""))
                                    steps.append(Step("search", call.get("name", "tool"), q))
                                    emit("search", q)
                            elif kind == "tool":
                                found = sources_from_tool(message_text(msg))
                                sources.extend(found)
                                steps.append(
                                    Step("sources", f"{len(found)} documents",
                                         ", ".join(s["source"] for s in found[:4]))
                                )
                                emit("sources", f"{len(found)} documents")
                            elif kind == "ai":
                                text = message_text(msg)
                                if text.strip():
                                    final = text
                if not final:
                    raise AgentError("the agent finished without producing an answer")

                steps.append(Step("answer", "answer ready", ""))
                emit("answer", "ready")
                self.memory.set(model)
                answer = Answer(
                    text=final,
                    model=model,
                    steps=steps,
                    sources=_dedupe_sources(sources),
                    calls=counter.calls,
                    elapsed=round(time.perf_counter() - started, 2),
                )
                if cache_key is not None and corpus:
                    self.cache.put(cache_key, answer, question, corpus)
                return answer
            except Exception as exc:                    # noqa: BLE001 - classified below
                if _is_quota_error(exc):
                    self.quota.mark_exhausted(model, _first_line(exc))
                    errors.append(f"{model}: out of quota")
                    emit("note", f"{model} exhausted - trying the next model")
                    continue
                if _is_unavailable_error(exc):
                    self.quota.mark_exhausted(model, _first_line(exc))
                    errors.append(f"{model}: unavailable")
                    emit("note", f"{model} unavailable - trying the next model")
                    continue
                if _is_transient_error(exc):
                    errors.append(f"{model}: busy")
                    emit("note", f"{model} is busy - trying the next model")
                    continue
                raise AgentError(_first_line(exc)) from exc

        detail = "; ".join(errors) if errors else "no response"
        if errors and all(error.endswith("out of quota") for error in errors):
            raise QuotaExhausted(
                f"every Gemini model is out of free-tier quota for today ({detail}) - {pacific_reset_hint()}"
            )
        raise AgentError(
            f"no Gemini model answered the request ({detail}) - Gemini may be busy, "
            "try again in a minute, or switch on Local mode"
        )

    # -- local mode ---------------------------------------------------------
    def ask_local(self, question: str, k: int = 4, weight: float = 0.6) -> Answer:
        question = (question or "").strip()
        if not question:
            raise RagError("ask something first")
        started = time.perf_counter()
        bundle = self.engine.search_all(question, k=k, weight=weight)
        steps = [
            Step("search", "keyword", f"{len(bundle.keyword)} hits · {bundle.timings['keyword']:.0f} ms"),
            Step("search", "vector", f"{len(bundle.vector)} hits · {bundle.timings['vector']:.0f} ms"),
            Step("search", "hybrid", f"{len(bundle.hybrid)} hits · {bundle.timings['hybrid']:.0f} ms"),
        ]
        sources = [{"source": h.source, "relevance": h.score} for h in bundle.hybrid]
        return Answer(
            text="",
            model="local (no LLM)",
            steps=steps,
            sources=_dedupe_sources(sources),
            calls=0,
            elapsed=round(time.perf_counter() - started, 2),
            local=True,
            hits=bundle.hybrid,
        )


def _dedupe_sources(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        src = str(item.get("source", "unknown"))
        score = float(item.get("relevance", 0.0) or 0.0)
        if src not in seen or score > seen[src]["relevance"]:
            seen[src] = {"source": src, "relevance": round(score, 3)}
    return sorted(seen.values(), key=lambda s: s["relevance"], reverse=True)


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def build(settings: Settings | None = None) -> tuple[RagEngine, Agent]:
    s = settings or Settings.from_env()
    engine = RagEngine(s)
    return engine, Agent(engine, s)


if __name__ == "__main__":                              # tiny manual smoke check
    s = Settings.from_env()
    print("env:", s.env_path)
    print("key present:", bool(s.gemini_key), "| models:", s.models()[:3])
    eng = RagEngine(s)
    print("ping:", eng.ping())
    if eng.table_exists():
        print("stats:", {k: v for k, v in eng.stats().items() if k not in ("columns", "indexes")})
        b = eng.search_all("hybrid search companies", k=3)
        print("keyword:", [(h.source, h.score) for h in b.keyword])
        print("vector :", [(h.source, h.score) for h in b.vector])
        print("hybrid :", [(h.source, h.score) for h in b.hybrid])
        print("timings:", {k: round(v, 1) for k, v in b.timings.items()})
