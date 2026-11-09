"""Smoke tests for rag_core. Spends zero Gemini quota unless --llm is passed.

    python tests/smoke_test.py            # engine only
    python tests/smoke_test.py --llm      # also asks the agent one question (2-6 calls)

Writes only to the scratch source "__smoke_test__.md" and removes it afterwards.
Exit code 0 = all passed, 1 = at least one failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc
from quality.run_eval import (_block, _rank, load_questions, needles_for,
                              question_type, questions_digest)
from tools.restore_corpus import documents_from_notebook

SCRATCH = "__smoke_test__.md"
results: list[tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((bool(condition), name, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def main() -> int:
    want_llm = "--llm" in sys.argv

    print("\n== settings ==")
    settings = rc.Settings.from_env()
    check("env file found", settings.env_path is not None, str(settings.env_path))
    check("oracle user configured", bool(settings.user), settings.user)
    check("models list non-empty", len(settings.models()) >= 1, ", ".join(settings.models()[:2]))

    print("\n== text helpers ==")
    terms = rc._terms("What are the major cloud providers doing with AI infrastructure?")
    check("_terms drops stopwords", "the" not in [t.lower() for t in terms] and len(terms) >= 3, str(terms))
    chunks = rc.chunk_text("para one. " * 200, size=400, overlap=50)
    check("chunk_text splits and respects size", len(chunks) > 1 and all(len(c) <= 420 for c in chunks),
          f"{len(chunks)} chunks, max {max(len(c) for c in chunks)} chars")
    check("chunk_text handles tiny input", rc.chunk_text("hello world") == ["hello world"])
    check("published_from_text reads a dateline",
          rc.published_from_text("**Santa Clara, CA - March 10, 2026**") == "2026-03-10",
          rc.published_from_text("**Santa Clara, CA - March 10, 2026**"))
    check("published_from_text ignores a bare year",
          rc.published_from_text("In 2026 the market grew quickly.") == "")
    tool_text = (
        "[Source: a.md | Relevance: 0.85 | Published: 2026-02-14]\nbody text"
        "\n\n---\n\n[Source: b.md | Relevance: 0.50]\nother text"
    )
    parsed = rc.sources_from_tool(tool_text)
    check("tool sources carry a snippet and a date",
          len(parsed) == 2 and parsed[0].get("snippet") == "body text"
          and parsed[0].get("published") == "2026-02-14", str(parsed))
    check("tool sources without a date still parse",
          parsed[1].get("source") == "b.md" and "published" not in parsed[1])
    numbered = (
        "[1] [Source: a.md | Relevance: 0.85 | Published: 2026-02-14]\nbody text"
        "\n\n---\n\n[2] [Source: a.md | Relevance: 0.70 | Published: 2026-02-14]\nmore"
        "\n\n---\n\n[3] [Source: b.md | Relevance: 0.50]\nother text"
    )
    numbered_sources = rc.sources_from_tool(numbered)
    check("tool sources keep their citation numbers",
          [item.get("citation") for item in numbered_sources] == [1, 2, 3],
          str([item.get("citation") for item in numbered_sources]))
    mapping = rc.citation_numbers(numbered_sources)
    check("citation numbers map to their sources",
          sorted(mapping) == [1, 2, 3] and mapping[1]["source"] == "a.md",
          str(sorted(mapping)))
    deduped = rc._dedupe_sources(numbered_sources)
    first = next(item for item in deduped if item["source"] == "a.md")
    check("one document keeps every number that cited it",
          first.get("citations") == [1, 2], str(first.get("citations")))
    check("each citation keeps its own passage",
          first.get("passages") == {"1": "body text", "2": "more"},
          str(first.get("passages")))
    check("prompt asks for numbered citations", "[1]" in rc.PROMPT)

    print("\n== quality questions ==")
    questions = load_questions()
    kinds = [question_type(item) for item in questions]
    counts = {kind: kinds.count(kind) for kind in ("single", "cross", "unanswerable")}
    check("question set has 40 entries", len(questions) == 40, str(len(questions)))
    check("question mix is 24 single / 8 cross / 8 unanswerable",
          counts == {"single": 24, "cross": 8, "unanswerable": 8}, str(counts))
    check("single and cross questions have needles",
          all(needles_for(q) for q, kind in zip(questions, kinds) if kind != "unanswerable"))
    check("cross questions need at least two needles",
          all(len(needles_for(q)) >= 2 for q, kind in zip(questions, kinds) if kind == "cross"))
    check("unanswerable questions have no needles",
          all(not needles_for(q) for q, kind in zip(questions, kinds) if kind == "unanswerable"))
    check("question digest is stable",
          questions_digest(questions) == questions_digest(load_questions()))

    with redirect_stdout(StringIO()):
        sample = documents_from_notebook(
            ROOT / "notebook" / "oracle_agentic_rag_hybrid_search.ipynb")
    haystack = "\n".join(
        str(doc.metadata.get("source", "")) + "\n" + doc.page_content for doc in sample
    ).lower()
    missing = [needle for item, kind in zip(questions, kinds)
               if kind != "unanswerable"
               for needle in needles_for(item) if needle.lower() not in haystack]
    check("every expected string exists in the sample corpus", not missing,
          str(missing[:3]))

    chunked = [part for doc in sample
               for part in rc.chunk_text(doc.page_content, 900, 120)]
    chunk_haystack = "\n".join(chunked).lower()
    missing_chunks = [needle for item, kind in zip(questions, kinds)
                      if kind != "unanswerable"
                      for needle in needles_for(item)
                      if needle.lower() not in chunk_haystack]
    check("default chunking preserves every expected string", not missing_chunks,
          str(missing_chunks[:3]))

    hits = [
        rc.Hit("a", "doc_a.md", "Alpha evidence", 0.9, "vector"),
        rc.Hit("b", "doc_b.md", "Beta evidence", 0.8, "vector"),
        rc.Hit("c", "doc_c.md", "Gamma evidence", 0.7, "vector"),
    ]
    check("_rank finds a single needle", _rank(hits, ["Beta"]) == 2)
    check("_rank takes the later of two needles", _rank(hits, ["Alpha", "Gamma"]) == 3)
    check("_rank is case-insensitive", _rank(hits, ["beta"]) == 2)
    check("_rank also matches the source name", _rank(hits, ["doc_b"]) == 2)
    check("_rank returns None when a needle is absent", _rank(hits, ["Alpha", "Delta"]) is None)
    check("_rank with no needles is None", _rank(hits, []) is None)

    ranked = [
        {"keyword": 1, "vector": 2, "hybrid": None, "rerank": 1},
        {"keyword": 10, "vector": None, "hybrid": 4, "rerank": 3},
    ]
    block = _block(ranked, k=5)
    check("_block counts keyword hits inside k",
          block["keyword"]["hit"] == 0.5 and block["keyword"]["found"] == 2,
          str(block["keyword"]))
    check("_block treats None as a miss",
          block["hybrid"]["hit"] == 0.5 and block["hybrid"]["found"] == 1,
          str(block["hybrid"]))
    check("_block excludes misses from MRR",
          block["vector"]["mrr"] == 0.25, str(block["vector"]))
    check("_block scores the rerank path",
          block["rerank"]["hit"] == 1.0 and block["rerank"]["found"] == 2,
          str(block["rerank"]))
    check("question_type infers cross from expect_all",
          question_type({"expect_all": ["A", "B"]}) == "cross")
    check("question_type infers unanswerable from an empty expect",
          question_type({"expect": None}) == "unanswerable")

    print("\n== container ==")
    docker = rc.container_status()
    check("container visible to docker", docker["state"] in ("running", "exited", "paused"), 
          f"{docker['state']} / {docker['health']}")

    print("\n== connection ==")
    engine = rc.RagEngine(settings)
    ping = engine.ping()
    check("database reachable", ping["ok"], f"{ping.get('version', ping.get('error'))} in {ping['ms']} ms")
    if not engine.ping()["ok"]:
        engine.close()
        return finish()

    check("table exists", engine.table_exists(), settings.table)
    stats = engine.stats()
    check("table has rows", stats["rows"] > 0, f"{stats['rows']} rows, {stats['sources']} sources")
    check("table has the OracleVS columns",
          {"ID", "TEXT", "METADATA", "EMBEDDING"} <= set(stats["columns"]),
          ", ".join(stats["columns"]))
    names = {i["name"].upper() for i in stats["indexes"]}
    check("HNSW vector index present", rc.HNSW_INDEX.upper() in names)
    check("Oracle Text index present", rc.TEXT_INDEX.upper() in names,
          " (falls back to substring scan if absent)")

    print("\n== retrieval ==")
    t0 = time.perf_counter()
    engine.warm()
    print(f"       embedding model warm in {time.perf_counter() - t0:.1f}s")

    kw = engine.keyword_search("LangChain enterprise databases", k=5)
    check("keyword returns hits", len(kw) > 0, f"{len(kw)} hits")
    check("keyword scores in (0,1]", all(0 < h.score <= 1.0 for h in kw),
          ", ".join(f"{h.score:.2f}" for h in kw[:3]))

    vec = engine.vector_search("companies building AI into their data stack", k=5)
    check("vector returns hits", len(vec) > 0, f"{len(vec)} hits")
    check("vector scores in [0,1]", all(0 <= h.score <= 1.0 for h in vec),
          ", ".join(f"{h.score:.2f}" for h in vec[:3]))
    check("vector hit has a source", all(h.source != "unknown" for h in vec))
    check("vector hit has text", all(len(h.text) > 50 for h in vec))

    hyb = engine.hybrid_search("LangChain integrations with enterprise databases", k=5, weight=0.6)
    check("hybrid returns hits", len(hyb) > 0, f"{len(hyb)} hits")
    check("hybrid exposes both components",
          all({"vector", "keyword"} <= set(h.detail) for h in hyb) if hyb else False,
          json.dumps(hyb[0].detail) if hyb else "")

    bundle = engine.search_all("Which companies are putting money into hybrid search?", k=5,
                               weight=0.6, rerank=True)
    check("bundle has four paths",
          bool(bundle.keyword) and bool(bundle.vector)
          and bool(bundle.hybrid) and bool(bundle.rerank))
    check("reranked hits use the reranker",
          all(h.method == "rerank" for h in bundle.rerank))
    check("rerank keeps the raw cross-encoder score",
          all("cross_encoder" in h.detail for h in bundle.rerank),
          json.dumps(bundle.rerank[0].detail) if bundle.rerank else "")
    check("timings recorded and sane",
          all(0 <= v < 20000 for v in bundle.timings.values()),
          ", ".join(f"{k} {v:.0f} ms" for k, v in bundle.timings.items()))
    check("consensus computed", isinstance(bundle.consensus(), list), f"{len(bundle.consensus())} docs in >=2 paths")

    plain = engine.search_all("Which companies are putting money into hybrid search?", k=5)
    check("rerank is opt-in and off by default",
          plain.rerank == [] and "rerank" not in plain.timings)

    target = "07_startup_vector_database.md"
    if engine.source_exists(target):
        filtered = engine.search_all("hybrid search", k=5, source=target)
        hits = filtered.keyword + filtered.vector + filtered.hybrid
        check("source filter is recorded", filtered.source == target, filtered.source)
        check("source filter keeps every hit in that source",
              bool(hits) and all(h.source == target for h in hits),
              f"{len(hits)} hits")
        reranked = engine.search_all("hybrid search", k=5, source=target, rerank=True)
        check("source filter keeps reranked hits in that source",
              bool(reranked.rerank)
              and all(h.source == target for h in reranked.rerank),
              f"{len(reranked.rerank)} hits")
    else:
        check("source filter target exists", False, target)

    lo, hi = engine.published_range()
    check("published range known", bool(lo and hi), f"{lo} to {hi}")
    day = "2026-02-14"
    dated = engine.search_all("hybrid search", k=5, date_from=day, date_to=day)
    dated_hits = dated.keyword + dated.vector + dated.hybrid
    check("date filter is recorded",
          dated.date_from == day and dated.date_to == day,
          f"{dated.date_from} to {dated.date_to}")
    check("date filter keeps every hit on that day",
          bool(dated_hits) and all(h.published == day for h in dated_hits),
          f"{len(dated_hits)} hits, dates "
          f"{sorted({h.published for h in dated_hits})}")
    dated_rerank = engine.search_all("hybrid search", k=5, rerank=True,
                                     date_from=day, date_to=day)
    check("date filter keeps reranked hits on that day",
          bool(dated_rerank.rerank)
          and all(h.published == day for h in dated_rerank.rerank),
          f"{len(dated_rerank.rerank)} hits")
    check("every hit carries its published date",
          all(h.published for h in engine.vector_search("AI", k=5)))
    check("source list carries published dates",
          all(row.get("published") for row in engine.sources()))
    check("source passages are available",
          bool(engine.source_passages(target, limit=2)))
    both = engine.search_all("hybrid", k=3, source=target,
                             date_from=day, date_to=day)
    combined = both.keyword + both.vector + both.hybrid
    check("source and date filters combine",
          bool(combined) and all(h.source == target and h.published == day
                                 for h in combined),
          f"{len(combined)} hits")

    # the weight slider must actually change the ranking
    zero = engine.hybrid_search("hybrid search", k=5, weight=0.0)
    one = engine.hybrid_search("hybrid search", k=5, weight=1.0)
    kw_ref = engine.keyword_search("hybrid search", k=5)
    vec_ref = engine.vector_search("hybrid search", k=5)
    check("weight=0 reproduces the keyword ranking",
          [h.key for h in zero] == [h.key for h in kw_ref],
          f"{len(zero)} vs {len(kw_ref)}")
    check("weight=1 reproduces the vector ranking",
          [h.key for h in one] == [h.key for h in vec_ref],
          f"{len(one)} vs {len(vec_ref)}")

    print("\n== corpus round-trip ==")
    before = engine.stats()["rows"]
    if engine.source_exists(SCRATCH):
        engine.delete_source(SCRATCH)
    doc = ("VectorForge and Oracle are both building hybrid retrieval. " * 40)
    added = engine.ingest(doc, SCRATCH, chunk_size=600, overlap=80)
    check("ingest wrote chunks", added["chunks"] >= 2, f"{added['chunks']} chunks")
    check("row count grew", engine.stats()["rows"] > before)
    hits = engine.vector_search("hybrid retrieval at VectorForge", k=5)
    check("new document is retrievable", any(h.source == SCRATCH for h in hits))
    try:
        engine.ingest(doc, SCRATCH)
        duplicate_blocked = False
    except rc.RagError:
        duplicate_blocked = True
    check("duplicate source refused", duplicate_blocked)

    exported = engine.export_corpus()
    check("export covers every row",
          len(exported["documents"]) == engine.stats()["rows"],
          f"{len(exported['documents'])} documents")
    scratch_chunks = [d for d in exported["documents"] if d["source"] == SCRATCH]
    check("export keeps the scratch chunks", len(scratch_chunks) == added["chunks"],
          f"{len(scratch_chunks)} chunks")
    check("export stores text, not embeddings",
          all("embedding" not in d for d in exported["documents"]))
    check("export is plain JSON",
          bool(json.dumps(exported, ensure_ascii=False)))
    check("export records the embedding model", bool(exported.get("embedding_model")),
          str(exported.get("embedding_model")))

    removed = engine.delete_source(SCRATCH)
    check("delete removed the chunks", removed == added["chunks"], f"{removed} rows")
    check("source is gone before import", not engine.source_exists(SCRATCH))

    restored = engine.import_documents(scratch_chunks)
    check("import restores the chunks",
          restored["chunks"] == added["chunks"] and engine.source_exists(SCRATCH),
          f"{restored['chunks']} chunks")
    again = engine.import_documents(scratch_chunks)
    check("import skips an existing source by default",
          again["chunks"] == 0 and again["skipped"] == added["chunks"],
          f"{again['skipped']} skipped")
    replaced = engine.import_documents(scratch_chunks, replace=True)
    check("import replaces on request",
          replaced["replaced"] == 1 and replaced["chunks"] == added["chunks"],
          f"replaced {replaced['replaced']}")

    removed = engine.delete_source(SCRATCH)
    check("delete removed the chunks again", removed == added["chunks"], f"{removed} rows")
    check("row count restored", engine.stats()["rows"] == before)

    print("\n== quota bookkeeping ==")
    with tempfile.TemporaryDirectory() as tmp:
        quota = rc.Quota(Path(tmp) / "quota.json", limit=20)
        quota.record("model-a", 3)
        quota.record("model-b", 1)
        quota.mark_exhausted("model-b", "429")
        status = quota.status()
        check("calls counted per model", status["models"]["model-a"]["calls"] == 3,
              json.dumps(status["models"]))
        check("exhaustion recorded", quota.is_exhausted("model-b"))
        check("other models unaffected", not quota.is_exhausted("model-a"))
        check("spend total", status["spent"] == 4, str(status["spent"]))
        quota.clear()
        check("clear resets", quota.status()["models"] == {})

    print("\n== answer cache ==")
    with tempfile.TemporaryDirectory() as tmp:
        cache = rc.AnswerCache(Path(tmp) / "answers.json")
        fingerprint = engine.corpus_fingerprint()
        check("corpus fingerprint computed", bool(fingerprint), str(fingerprint))

        answer = rc.Answer(
            text="Cached facts.",
            model="test-model",
            steps=[rc.Step("search", "vector", "1 hit")],
            sources=[{"source": SCRATCH, "relevance": 0.9}],
            calls=3,
            elapsed=2.5,
        )
        key = cache.key("What is cached?", fingerprint)
        cache.put(key, answer, "What is cached?", fingerprint)
        got = cache.get(key)
        check("cache returns the answer", bool(got) and got.text == "Cached facts.")
        check("cache marks it as cached", bool(got) and got.cached)
        check("cache keeps steps and sources",
              bool(got) and got.steps[0].label == "vector"
              and got.sources[0]["source"] == SCRATCH)
        check("key ignores case and spacing",
              cache.key("  what is   CACHED? ", fingerprint) == key)
        check("key changes when the corpus changes",
              cache.key("What is cached?", fingerprint + "x") != key)
        check("stats reports one entry", cache.stats()["entries"] == 1,
              str(cache.stats()["entries"]))
        check("clear empties the cache",
              cache.clear() == 1 and cache.stats()["entries"] == 0)

        # a hit must be served without touching a model
        cached_agent = rc.Agent(engine, settings, cache=cache)
        hit_key = cache.key("Cached question", engine.corpus_fingerprint() or "")
        cache.put(hit_key, rc.Answer(text="from cache", model="cached-model", calls=2, elapsed=1.0),
                  "Cached question", engine.corpus_fingerprint() or "")

        def fail(model: str) -> None:
            raise AssertionError("the model was called despite a cache hit")

        cached_agent._graph = fail                     # type: ignore[assignment]
        served = cached_agent.ask("Cached question")
        check("agent serves a hit without a model call",
              served.text == "from cache" and served.cached, served.text)
        check("cached answer reports zero elapsed", served.elapsed == 0.0)

    print("\n== streaming contract ==")
    class FakeChunk:
        def __init__(self, text: str, kind: str = "ai", tool: bool = False) -> None:
            self.content = text
            self.type = kind
            self.tool_calls = ([{"name": "search_tech_news",
                                 "args": {"query": "x"}}] if tool else [])
            self.tool_call_chunks = ([{"name": "search_tech_news", "args": "{}"}]
                                     if tool else [])

    class FakeGraph:
        def __init__(self) -> None:
            self.inputs: list[dict] = []

        def stream(self, *args: object, **kwargs: object):
            if args and isinstance(args[0], dict):
                self.inputs.append(args[0])
            yield ("updates", {"agent": {"messages": []}})
            yield ("messages", (FakeChunk("Hello "), {}))
            yield ("messages", (FakeChunk("world"), {}))
            yield ("messages", (FakeChunk("ignore preamble", tool=True), {}))
            yield ("messages", (FakeChunk("SOURCE TEXT", kind="tool"), {}))
            yield ("updates", {"agent": {"messages": []}})

    class FakeMemory:
        def get(self) -> None:
            return None

        def set(self, model: str) -> None:
            return None

    with tempfile.TemporaryDirectory() as tmp:
        stream_agent = rc.Agent(
            engine,
            rc.Settings(gemini_key="test-key"),
            cache=rc.AnswerCache(Path(tmp) / "answers.json"),
        )
        graph = FakeGraph()
        stream_agent._graph = lambda model: graph  # type: ignore[assignment]
        stream_agent.memory = FakeMemory()  # type: ignore[assignment]
        tokens: list[str] = []
        resets: list[bool] = []
        streamed = stream_agent.ask(
            "stream please", use_cache=False,
            on_token=tokens.append, on_reset=lambda: resets.append(True),
        )
        check("streamed tokens arrive in order", "".join(tokens) == "Hello world",
              repr("".join(tokens)))
        check("streamed answer is complete", streamed.text == "Hello world",
              streamed.text)
        check("streaming resets once before the model", len(resets) == 1,
              str(len(resets)))
        check("tool output is not part of the answer",
              "SOURCE" not in streamed.text and "ignore" not in streamed.text)

        follow = stream_agent.ask(
            "and who funded them?", use_cache=False,
            on_token=lambda text: None,
            context=[
                {"role": "user", "content": "Which company raised money?"},
                {"role": "assistant",
                 "answer": rc.Answer(text="VectorForge did.", model="fake")},
            ],
        )
        sent = graph.inputs[-1]["messages"]
        check("follow-up context precedes the question",
              len(sent) == 3 and sent[0] == ("user", "Which company raised money?")
              and sent[1] == ("assistant", "VectorForge did.")
              and sent[2] == ("user", "and who funded them?"),
              repr(sent))
        check("follow-up context is noted in the trace",
              any(step.kind == "note" for step in follow.steps),
              str([(s.kind, s.label) for s in follow.steps]))
        check("context changes the answer cache key",
              rc.AnswerCache.key("q", "1:2")
              != rc.AnswerCache.key("q", "1:2", extra={"context": [("user", "x")]}))
        check("errors are not sent as context",
              rc._context_messages([{"role": "error", "content": "quota"},
                                    {"role": "user", "content": "q"}])
              == [("user", "q")])

    if want_llm:
        print("\n== agent (spends quota) ==")
        agent = rc.Agent(engine, settings)
        steps: list[tuple[str, str]] = []
        try:
            answer = agent.ask("What is LangChain's role in enterprise RAG?",
                               on_step=lambda kind, text: steps.append((kind, text)))
            check("agent returned an answer", len(answer.text) > 40, f"{len(answer.text)} chars")
            check("agent cited sources", len(answer.sources) > 0, str(len(answer.sources)))
            check("agent recorded steps", len(answer.steps) > 0, str(len(answer.steps)))
            check("call count positive", answer.calls > 0, f"{answer.calls} calls")
            check("steps streamed to callback", len(steps) > 0, str(steps[:2]))
        except rc.QuotaExhausted as exc:
            print(f"  [SKIP] quota exhausted: {exc}")
        except rc.RagError as exc:
            check("agent answered without error", False, str(exc))
        engine.quota_used = True

    engine.close()
    return finish()


def finish() -> int:
    failed = [r for r in results if not r[0]]
    print("\n" + "-" * 60)
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("  failed:")
        for _, name, detail in failed:
            print(f"    - {name} {detail}")
        return 1
    print("  everything passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
