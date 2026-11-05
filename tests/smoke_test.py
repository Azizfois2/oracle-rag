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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_core as rc

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

    bundle = engine.search_all("Which companies are putting money into hybrid search?", k=5, weight=0.6)
    check("bundle has three paths", bool(bundle.keyword) and bool(bundle.vector) and bool(bundle.hybrid))
    check("timings recorded and sane",
          all(0 <= v < 20000 for v in bundle.timings.values()),
          ", ".join(f"{k} {v:.0f} ms" for k, v in bundle.timings.items()))
    check("consensus computed", isinstance(bundle.consensus(), list), f"{len(bundle.consensus())} docs in >=2 paths")

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
