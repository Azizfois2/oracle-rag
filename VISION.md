# VISION - Oracle RAG Console

> A local query console for the Oracle 26ai hybrid-search corpus.
> **Ask. Compare. Inspect. Trust.**

---

## 1. Why this exists

The notebook proved the idea: Oracle 26ai can hold vectors, text and JSON in one
table, and a single SQL statement can combine semantic and keyword retrieval.
That is a genuinely good demo. It is not a usable tool.

Three specific gaps:

| Gap | Consequence |
|---|---|
| You must be inside Jupyter, in the right cell, in the right order | A colleague who has never seen a notebook cannot use it |
| The interesting result - *why* hybrid beats keyword - is buried in `print()` output | The insight is asserted, not shown |
| Every experiment is destructive (the notebook drops and re-ingests the table) | You cannot explore safely once the corpus matters |

This app closes those gaps without throwing away the notebook.

---

## 2. Who it is for

1. **You, debugging retrieval.** Ask a question, see which tool fired, which
   chunks came back, what each path scored. No cell re-runs, no lost state.
2. **The colleague.** Double-click, type a question, read an answer with sources.
   Never sees Python.
3. **Anyone evaluating the work.** Vector vs keyword vs hybrid, side by side,
   with latency and hit-rate. The claim becomes measurable.

---

## 3. The one thing it must do well

> **Make the difference between retrieval methods visible and measurable.**

Every score on screen must be reproducible by a SQL query the user can copy.
No magic numbers, no "trust me" relevance scores. If the app says hybrid won,
the user can see *which* documents each path found and *why* the fusion ranked
them that way.

This is the product's reason to exist. Everything else is supporting cast.

---

## 4. Principles

1. **The notebook is the lab; the app is the product.**
   Nothing reaches the app that has not been proven in a notebook or a test first.
   The app never drops tables, never re-ingests silently, never surprises.

2. **`rag_core.py` is the only place that talks to Oracle or Gemini.**
   `app.py` contains zero SQL and zero prompt strings. One contract, many callers
   (app, eval harness, tests, future CLI).

3. **Every number is explained.**
   Similarity, keyword score, fusion weight, latency, API calls - surfaced, not
   hidden. A score with no explanation is a lie with extra steps.

4. **Degrade, never die.**
   - No API key → the app still searches (local mode).
   - Quota exhausted → say so, offer local mode, reset time.
   - Database down → a status screen with the exact fix, not a traceback.
   - Model retired → fall through the candidate list automatically.

5. **The free tier is a design constraint, not an excuse.**
   Embeddings run locally (zero API cost). One question = one agent run.
   The quota meter is always visible. Answer caching is a feature, not a hack.

6. **No hidden state.**
   Anything remembered lives in `data/*.json` - quota counters, last model,
   last eval run. Inspectable, deletable, obvious.

---

## 5. The five surfaces

### Ask - *the product*
Chat with the ReAct agent over the corpus.
- Streams step-level progress (searching → reading → writing), not a dead spinner.
- Every answer carries a trace: tool calls, the exact queries sent, sources with
  relevance, model used, API calls spent, wall-clock time.
- **Local mode** toggle: hybrid retrieval with no LLM at all. Zero quota, still
  useful, and the honest fallback when the free tier is empty.
- Empty state teaches: four example questions, one click each.

*Success:* a non-technical person gets a sourced answer in under 30 seconds and
can tell where it came from.
*Failure:* it answers without searching - the trace makes that obvious immediately.

### Search Lab - *the proof*
One query, three retrieval paths, one screen.
- Three columns: keyword (Oracle Text), vector (HNSW), hybrid (weighted fusion).
- Each hit: rank, source, score bar, snippet, and badges showing which *other*
  paths found it too.
- A weight slider that moves fusion from keyword-only to vector-only, live.
- Per-path latency in milliseconds.
- A consensus line: how many documents all three methods agree on.

*Success:* you can answer "is hybrid actually better here?" by looking.
*Failure:* the three columns return identical results - which is itself a finding.

### Corpus - *the control*
- Document and chunk counts, average chunk size, table size, index status.
- Every source listed with its chunk count.
- Ingest: paste text or upload `.txt` / `.md` / `.pdf`, with chunk size control.
- Delete by source, behind an explicit confirmation.
- Re-index: rebuild the HNSW or Oracle Text index if either went missing.

*Success:* you can add a document and query it thirty seconds later.
*Failure:* double-ingest - so an existing source is detected and reported, never
silently duplicated.

### Quality - *the honesty*
A small labelled question set, run against all three paths.
- hit@k and MRR per method, side by side.
- Per-question breakdown: which method found the expected document, at what rank.
- Results saved to `data/eval-last.json` so runs are comparable over time.

*Success:* changing a parameter has a number attached to it.
*Failure:* a question nobody can retrieve - visible, not swept away.

### Health - *the plumbing*
Database version, container state, table/index status, embedding model,
API key presence (never the value), and the per-model quota table with reset time.

*Success:* "why is it broken?" is answered on one screen.
*Failure:* the page lies about state - so every value is read live, never cached
beyond a few seconds.

---

## 6. Design direction

**A query console, not a chatbot skin.**

- Graphite ground, one brand accent (Oracle red `#C74634`), everything else quiet.
- **Colour carries meaning:** amber = keyword, blue = vector, green = hybrid.
  The same three colours appear in the lab, the trace and the eval table.
- Numbers live in monospace. Prose stays in a clean sans. Scores align right.
- Density over decoration: this is a tool for reading results, not a landing page.
- Cards, not boxes-in-boxes. One border, one accent stripe, generous line-height.
- Empty states teach (example queries, "ingest your first document").
- Errors name the cause and the fix, in that order.

---

## 7. Architecture

```
        notebook/                      quality/, tests/
   (explore, break things)          (prove it, measure it)
              \                          /
               \                        /
                ┌──────────────────┐
                │    rag_core.py   │   ← the only contract
                └──────────────────┘
                 connect · search · ingest · agent · quota
                          |
                    app.py (Streamlit)
                          |
              ask · lab · corpus · quality · health
```

`rag_core.py` public surface (stable - the app and tests both use it):

| Area | Functions |
|---|---|
| Config | `Settings.from_env()`, `Settings.models()` |
| Connection | `RagEngine.connect()`, `.ensure()`, `.stats()` |
| Retrieval | `.keyword_search(q,k)`, `.vector_search(q,k)`, `.hybrid_search(q,k,w)`, `.search_all(q,k,w)` |
| Corpus | `.sources()`, `.ingest(text, source)`, `.delete_source(s)`, `.reindex()`, `.export_corpus()`, `.import_documents(docs, replace=)` |
| Agent | `Agent.ask(q, on_step=...)`, `ask_local(q)` |
| Budget | `Quota.status()`, `.record()`, `.mark_exhausted()` |
| Cache | `AnswerCache.key()`, `.get()`, `.put()`, `.stats()`, `.clear()` |
| CLI | `ask.py "question"`, `--local`, `--json`, `--no-cache` |

Types: `Hit` (key, source, text, score, method, detail) and `Answer`
(text, model, steps, sources, calls, elapsed, cached). No dicts-of-dicts leaking
into the UI.

---

## 8. Data and privacy

- Documents, embeddings and indexes never leave the machine.
- Only the question and the retrieved passages go to Gemini.
- Prompts may be used by Google to improve products on the free tier - so the
  corpus is synthetic sample news, and the README says so plainly.
- The API key is read from `.env`, never printed, never committed.

---

## 9. Non-goals

- Multi-user, auth, deployment, sharing links. This is one person, one machine.
- Replacing JupyterLab. The notebook stays the place to try things.
- Being a vector-database benchmark. It is a teaching and demo console.
- A custom agent framework. LangGraph is already there and already works.
- Streaming token-by-token at the cost of reliability. Step-level progress is enough.

---

## 10. Constraints, and what they force

| Constraint | Design consequence |
|---|---|
| Free Gemini: ~20 requests/day **per model**, 2–6 per question | Quota meter, local mode, no probe calls, model fallback, answer cache |
| Database must survive a rebuilt container | Named volume `oracle-26ai-data` plus `--restart unless-stopped`; the README says what deleting the volume destroys |
| Windows, one machine, Python 3.11 venv | No Docker/Node build step; `.bat` launcher next to the existing one |
| 8–16 GB RAM, CPU only | 384-dim MiniLM embeddings; no GPU assumptions; no torch re-download |
| Beginner user | Every error message ends with the literal next action |

---

## 11. Roadmap

**v1 - now**
Five surfaces, local mode, quota meter, answer cache, eval harness, smoke tests.

**v1.1**
A named data volume with a restart policy, so a container rebuild or a reboot
does not cost the corpus.

**v1.2**
Corpus export/import, a download button in Health, and a terminal CLI, so a
volume loss is recoverable and the corpus is scriptable outside the browser.

**Next**
Local cross-encoder reranking, metadata filters (source, date),
multi-collection support, a larger labelled question set.

**Later**
Streaming answers, conversation history persisted in Oracle itself,
MCP server so other agents can query this corpus.

---

## 12. Definition of done (v1)

- [x] App starts and renders with the database **down**, explaining what to do.
- [x] All five surfaces render with a healthy database.
- [x] Keyword, vector and hybrid each return scored hits for the same query.
- [x] Every displayed score is reproducible with the SQL shown in `rag_core.py`.
- [x] Ingest → search → delete round-trip works without dropping the table.
- [x] Quota is counted, persisted, and visible; exhausted models are skipped.
- [x] Agent answers carry a trace with sources and call count.
- [x] `python tests/smoke_test.py` passes without spending API quota.
- [x] No secret in git history; `.env` and `data/` ignored.
- [x] A colleague can go from clone to browser in under two minutes if the venv exists.

---

## 13. Decision log

| Decision | Why |
|---|---|
| **Streamlit**, not React/FastAPI | One language, one process, no build step. The UI is data display, which Streamlit does natively. |
| **No PyInstaller** | torch + sentence-transformers → a ~2 GB exe that unpacks slowly and breaks on DLLs. Ship the folder and the existing venv. |
| **Extract `rag_core.py` before the UI** | Otherwise the notebook and the app become two implementations that drift, and every bug gets fixed twice. |
| **Raw SQL for retrieval, not only `OracleVS`** | The lab needs per-path scores, candidate pools and latencies. `OracleVS` is kept for ingestion and as the agent's tool. |
| **Linear weighted fusion, not RRF** | Explainable: `score = w·v̂ + (1−w)·k̂`, visible on screen and adjustable with one slider. RRF can come later. |
| **Min-max normalise each component before fusing** | Cosine similarity lives in a narrow band (~0.2–0.6) while keyword scores are 0–1, so a raw weighted sum silently favoured keyword and made the slider dishonest. Measured: raw fusion scored hit@5 0.92, normalised 1.00. |
| **Model fallback on 503, not just 429** | On the first live run, `gemini-3.8-flash` and `3.7` both returned 503 (high demand). A quota check alone would have failed the question while two healthy models sat unused. |
| **Local mode is first-class** | The free tier cannot be the only way in. A demo that dies at 20 requests/day is not a demo. |
| **Keep Gemini's native SDK** | The OpenAI-compatible endpoint drops thought signatures and breaks the ReAct loop on the second turn. Already proven the hard way. |
| **Notebook moved into `notebook/`** | So the whole project - lab and product - is versioned in one repository. |
