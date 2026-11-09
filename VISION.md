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
- Streams the answer text token by token, with step-level progress (searching →
  reading → writing) above it, not a dead spinner.
- Every answer carries a trace: tool calls, the exact queries sent, sources with
  relevance, model used, API calls spent, wall-clock time.
- Search results are numbered for the whole question, and the answer cites those
  numbers inline. Clicking `[1]` opens the exact passage and date beside the
  sentence; hover and keyboard focus do the same. The source list repeats the
  numbers.
- **Local mode** toggle: hybrid retrieval with no LLM at all. Zero quota, still
  useful, and the honest fallback when the free tier is empty.
- **Follow-up context** toggle: the last few turns travel with the question, so
  "who funded them?" still makes sense. The trace says how many were sent.
- Discussions are saved in Oracle and restored on open. The sidebar lists recent
  threads, searches their text, can rename them, downloads them as Markdown, and
  puts deletion behind a confirmation.
- Empty state teaches: four example questions, one click each.

*Success:* a non-technical person gets a sourced answer in under 30 seconds and
can tell where it came from.
*Failure:* it answers without searching - the trace makes that obvious immediately.

### Search Lab - *the proof*
One query, four retrieval paths, one screen.
- Four columns: keyword (Oracle Text), vector (HNSW), hybrid (weighted fusion),
  and reranked (local cross-encoder over the hybrid candidates).
- Each hit: rank, source, published date, score bar, snippet, and badges showing
  which *other* paths found it too.
- A weight slider that moves fusion from keyword-only to vector-only, live.
- A source filter and a published-date range that apply to every path, so one
  document or one time window can be inspected on its own. Undated rows are left
  out when a range is set.
- Per-path latency in milliseconds.
- A consensus line: how many documents the four methods agree on.

*Success:* you can answer "is reranking actually better here?" by looking.
*Failure:* the four columns return identical results - which is itself a finding.

### Corpus - *the control*
- Document and chunk counts, average chunk size, table size, index status.
- Every source listed with its chunk count and published date.
- Ingest: paste text or upload `.txt` / `.md` / `.pdf`, with chunk size control.
- Delete by source, behind an explicit confirmation.
- Re-index: rebuild the HNSW or Oracle Text index if either went missing.

*Success:* you can add a document and query it thirty seconds later.
*Failure:* double-ingest - so an existing source is detected and reported, never
silently duplicated.

### Quality - *the honesty*
A 40-question labelled set, run against all four paths.
- 24 single-document, 8 cross-document, 8 unanswerable.
- Cross-document hits require every expected string in the top k.
- hit@k and MRR per method, plus a single-versus-cross breakdown.
- Unanswerable questions are not scored as hits. Their vector top score is
  compared with answerable questions; keyword and reranked scores are normalised
  per query, so they are not a confidence measure.
- Results saved to `data/eval-last.json` so runs are comparable over time. A
  changed question file **or corpus fingerprint** invalidates the stored run.
- Each run rebuilds the Oracle Text index first, so keyword scores do not depend
  on what was ingested or deleted earlier.

The answer gate sits beside the retrieval benchmark. It scores 28 labelled
answers: 19 single-document, 4 cross-document and 5 unanswerable. A case passes
only when every expected source is cited, every expected fact appears in the
text, an unanswerable question is refused, and every citation names a real
document and a number the tool returned. The scorer is local and costs no quota;
a live run is opt-in and small by default. Results go to `data/answers-last.json`.

*Success:* changing a parameter has a number attached to it, and a wrong answer
has a named reason.
*Failure:* a question nobody can retrieve - visible, not swept away.

### Health - *the plumbing*
Database version, container state, table/index status, embedding model,
API key presence (never the value), the per-model quota table with reset time,
and discussion history counts with a clear-all action.

*Success:* "why is it broken?" is answered on one screen.
*Failure:* the page lies about state - so every value is read live, never cached
beyond a few seconds.

---

## 6. Design direction

**A query console, not a chatbot skin.**

- Graphite ground, one brand accent (Oracle red `#C74634`), everything else quiet.
- **Colour carries meaning:** amber = keyword, blue = vector, green = hybrid,
  purple = reranked. The same four colours appear in the lab, the trace and the
  eval table.
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
                 connect · search · ingest · agent · quota · history
                  /            |              \
        app.py (Streamlit)   ask.py      mcp_server.py
        five UI surfaces     terminal     read-only MCP tools
```

`rag_core.py` public surface (stable - the app and tests both use it):

| Area | Functions |
|---|---|
| Config | `Settings.from_env()`, `Settings.models()` |
| Connection | `RagEngine.connect()`, `.ensure()`, `.stats()` |
| Retrieval | `.keyword_search(q,k,source=,date_from=,date_to=)`, `.vector_search(...)`, `.hybrid_search(q,k,w,...)`, `.rerank_hits(q,hits,k)`, `.reranked_search(...)`, `.search_all(q,k,w,rerank=,source=,date_from=,date_to=)` |
| Corpus | `.sources()`, `.published_range()`, `.source_passages(source)`, `.ingest(text, source, metadata=)`, `.delete_source(s)`, `.reindex()`, `.export_corpus()`, `.import_documents(docs, replace=)` |
| Citations | `sources_from_tool(text)`, `citation_numbers(items)` |
| Agent | `Agent.ask(q, on_step=, on_token=, on_reset=, context=)`, `ask_local(q, context=)` |
| Budget | `Quota.status()`, `.record()`, `.mark_exhausted()` |
| Cache | `AnswerCache.key()`, `.get()`, `.put()`, `.stats()`, `.clear()` |
| History | `ChatHistory.list()`, `.search()`, `.messages()`, `.title()`, `.export_markdown()`, `.append()`, `.rename()`, `.delete()`, `.clear()` |
| MCP | `mcp_server.py`: `search_corpus`, `list_sources`, `corpus_info` (read-only, stdio) |
| CLI | `ask.py "question"`, `--local`, `--json`, `--no-cache`, `--chat`, `--no-history` |

Types: `Hit` (key, source, text, score, method, detail, published) and `Answer`
(text, model, steps, sources, calls, elapsed, cached). Each `Answer.sources` item
can carry `snippet`, `published`, `citation`, `citations` and `passages`, so the
UI can open the exact chunk behind each number without another query. No
dicts-of-dicts leaking into the UI.

---

## 8. Data and privacy

- Documents, embeddings and indexes never leave the machine.
- Discussion history (questions, answers, traces) is stored in the local Oracle
  database, not in a browser profile or a cloud service.
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
| 8–16 GB RAM, CPU only | 384-dim MiniLM embeddings; a small MiniLM cross-encoder reranks at most 30 candidates; no GPU assumptions; no torch re-download |
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

**v1.3**
A 40-question benchmark: 24 single-document, 8 cross-document, 8 unanswerable.
Cross-document hits require every expected string in the top k. Unanswerable
questions are compared with answerable ones instead of being scored as hits.

**v1.4**
A 46-chunk corpus re-split from the same ten articles, plus local cross-encoder
reranking of the hybrid candidates. Judged against the v1.3 benchmark: reranking
lifts all-answerable MRR from 0.68 to 0.87 and puts every single-document answer
at rank 1. The reranker runs on the CPU, so it costs no API quota.

**v1.5**
A read-only MCP server over stdio: `search_corpus`, `list_sources` and
`corpus_info`. Other agents can query the corpus without a Gemini key or the
Streamlit UI. No write tools, no network listener, no quota.

**v1.6**
Discussion history persisted in Oracle. One table, one row per message. The
sidebar lists recent threads, reopening the app returns to the latest one, and
each answer rebuilds its trace, sources and retrieval hits. History is separate
from the corpus, costs no quota, and is never dropped automatically.

**v1.7**
Answers stream token by token into the page, with the trace still filling in
above. The sidebar searches saved discussions by text, one result per thread,
with the latest matching message as the snippet. Streaming adds an optional
callbacks pair to `Agent.ask`; the cache, sources and model fallback are
unchanged.

**v1.8**
Follow-up questions carry the last few turns, behind a sidebar toggle, and the
cache key includes that context. Discussions can be renamed and downloaded as
Markdown. Search shows a count, highlights the matching words and pages past the
first four. The Search lab filters every path to one source.

**v1.9**
An interactive terminal chat, `ask.py --chat`, over the same `Agent` and
`ChatHistory` as the app. It restores the latest discussion, streams answers,
keeps follow-up context, and supports `/history`, `/find`, `/open`, `/sources`,
`/trace`, `/title` and `/delete`. `--no-history` keeps a test session in memory.
No new dependency: Rich and prompt_toolkit were already installed.

**v2.0**
Published dates in the corpus metadata, read from each article's dateline. The
Search lab filters every retrieval path by an inclusive date range, and every hit
shows its date. Existing corpora gain dates in place with `JSON_MERGEPATCH`, so
nothing is re-embedded. Answers keep the passage behind each citation, and the
Ask page turns each source into a clickable expander. A 28-case answer gate
checks expected sources, expected facts, refusals and invented citations, with
zero-quota scoring and an opt-in live run.

**v2.1**
Search results carry a number for the whole question, even across several tool
calls, and the answer cites that number inline. Each `[1]` is a self-targeting
link that opens the exact passage and date without JavaScript; hover and keyboard
focus work too. A document keeps every number that cited it, plus the passage for
each number. `tests/run_all.py` runs all seven zero-quota suites and prints one
summary.

**Next**
A larger and more varied labelled set, multi-collection support, and a live
answer run that can resume where quota stopped it.

**Later**
Longer discussions with summarised memory and a shareable static report. A
full-screen Textual TUI only if the terminal chat proves it needs panes and mouse
support; it would add a dependency and a second rendering surface.

---

## 12. Definition of done (v1)

- [x] App starts and renders with the database **down**, explaining what to do.
- [x] All five surfaces render with a healthy database.
- [x] Keyword, vector and hybrid each return scored hits for the same query.
- [x] Every displayed score is reproducible with the SQL shown in `rag_core.py`.
- [x] Ingest → search → delete round-trip works without dropping the table.
- [x] Quota is counted, persisted, and visible; exhausted models are skipped.
- [x] Agent answers carry a trace with sources and call count.
- [x] A remote answer streams in token by token and still ends with the full trace.
- [x] A follow-up sends the earlier turns and the cache key includes them.
- [x] A discussion survives a Streamlit restart, with its trace and sources.
- [x] A word from a saved discussion finds that thread from the sidebar.
- [x] A saved discussion can be renamed and downloaded with its sources.
- [x] The Search lab can restrict every retrieval path to one source.
- [x] The Search lab can restrict every path to a published-date range.
- [x] A citation opens to the passage and date behind it.
- [x] A numbered citation in the answer opens the exact passage it points at.
- [x] The answer gate scores sources, facts, refusals and invented citations without quota.
- [x] One command runs every zero-quota suite and prints one summary.
- [x] `ask.py --chat` restores a saved discussion, streams an answer, and keeps context.
- [x] An MCP client can list three tools and search the corpus over stdio.
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
| **Re-chunk the ten articles instead of fetching more** | The benchmark is labelled against this text. Unrelated articles could produce false-positive hits and contaminate the ground truth. Re-chunking keeps the facts identical and changes only retrieval granularity, while making k = 5 a real test (5 of 46). |
| **Local cross-encoder, not an API reranker** | Zero quota, no extra key, and about a second per query on CPU. Retrieval still belongs to Oracle; the reranker only reorders the hybrid pool. |
| **Stable keyword benchmark** | Oracle Text score ties and index-level statistics made repeated runs disagree (cross hit@5 moved between 0.25 and 0.50). `ORDER BY kw DESC, id` removes tie drift; the Quality run rebuilds the text index first, which changes no data. |
| **Read-only MCP tools, not an LLM-backed ask** | Retrieval is the reusable capability. An `ask` tool would spend quota, hide the retrieval step, and duplicate the app's agent. The server exposes search and corpus metadata only, over stdio, with no network listener. |
| **Discussions in Oracle, not browser storage or a JSON file** | The corpus already lives in Oracle, so history survives a browser reset and a machine restart, needs no second backup file, and can be queried with SQL. Browser storage is fragile, and a JSON file would be one more source of truth. The table is created on first use and never dropped automatically. |
| **Substring search over history, not full-text indexing** | A second Oracle Text index would complicate setup for a table that holds one person's chats. A case-insensitive substring match is predictable, needs no index maintenance, and stays fast at this size. Wildcards are escaped so `%` means the character, not "anything". |
| **Token callbacks on `Agent.ask`, not a second agent** | `on_token` and `on_reset` are optional, so the CLI, the MCP server and every test keep the same call. Streaming reuses the same graph and the same quota counter; only the event mode changes. The final `Answer` still arrives whole and is what gets cached and saved. |
| **Follow-up context as messages, not a rewritten prompt** | The graph already accepts a message list, so earlier turns are sent as normal user/assistant messages. The prompt stays one constant, and the CLI keeps asking standalone questions. Only the last six turns are sent, capped per turn, so retrieval still has room. The cache key includes them. |
| **Source filter in the retrieval SQL, not after ranking** | Filtering after `FETCH APPROX` would silently shrink the candidate pool and change what each path can return. The predicate sits inside every query, so keyword, vector, hybrid and rerank all see the same one-document corpus. |
| **Markdown export from the stored answer, not a second renderer** | The export is built by `ChatHistory` from the same rows the UI restores, so it carries the same text, sources and trace. No separate template can drift from what the screen shows. |
| **Interactive terminal chat, not a full-screen TUI** | `ask.py --chat` reuses `Agent`, `ChatHistory` and the streaming callbacks, so it adds no SQL, no prompt text and no dependency. A Textual TUI would duplicate chat, history, search and settings in a second rendering surface, and `textual` is not installed. The terminal chat covers streaming, follow-ups, saved history and search; a TUI can wait until panes and mouse support are actually needed. |
| **Published dates in metadata, not a new column** | The corpus metadata is already JSON, and `published` is optional. A new column would force a schema migration and touch every ingest path. ISO strings compare lexically, so the date range is a plain `JSON_VALUE` predicate inside every retrieval query. Rows without a date are excluded from a range instead of being guessed into it. |
| **Patch dates in place, not re-embed** | `JSON_MERGEPATCH` changes metadata only. Re-ingesting would recompute every embedding for no retrieval gain and would change row ids, which would invalidate saved answers and evaluation runs. The fingerprint is based on row count and id hash, so it stays stable. |
| **Citations keep the tool's passage** | The retrieval tool already returns the exact chunk the model saw. Parsing that block into `snippet` and `published` means a click can show the evidence without a second search, and an older saved answer falls back to `source_passages()`. |
| **Answer quality as a separate scorer** | Retrieval hit@k asks whether the right chunk was found. It cannot tell whether the answer cited it, used the fact, refused an unanswerable question, or invented a source. A separate labelled set and pure scoring function keep those failures named, and fake answers let the scorer run with zero quota. |
| **Numbered tool results, not source names in the text** | A short marker is easier for the model to repeat exactly than a file name, and it stays stable across several tool calls because one counter runs for the whole question. The tool block carries the number, so the renderer never has to guess which source the model meant. A document seen in several chunks keeps every number and the passage for each. |
| **Self-targeting citation links, not JavaScript** | `[1]` is an anchor whose `href` points at its own `id`. The `:target` rule opens the passage, and hover or keyboard focus opens it too. This needs no script, no component iframe and no second request, and it still works if a browser blocks focus on a plain span. |
| **One-command runner captures each suite** | Seven separate commands are easy to forget and easy to run with the wrong interpreter. `tests/run_all.py` uses `sys.executable`, runs only the zero-quota suites, keeps full output in `logs/tests/`, and prints one summary. `--llm` and the live answer check stay manual because they spend quota. |
