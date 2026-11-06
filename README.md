# Oracle RAG Console

A local web app for the Oracle 26ai hybrid-search corpus: **ask questions, compare
keyword vs vector vs hybrid vs reranked retrieval, manage the corpus, measure
quality, check the plumbing** - without touching a notebook cell.

Vision, principles and roadmap: [`VISION.md`](VISION.md).

---

## Quick start

```bat
START-GUI.bat
```

That checks the venv, installs the app dependencies if needed, starts Streamlit on
port 8501 and opens the browser. The Oracle container must be running.

The container, the `vector` user and the Python environment are created by the
companion package [`rag-oneclick`](https://github.com/Azizfois2/rag-oneclick).

Manual start:

```bat
%USERPROFILE%\rag-venv\Scripts\streamlit.exe run app.py --server.port 8501
```

Then open <http://localhost:8501>.

---

## Command line

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe ask.py "Which company raised a Series B?"
%USERPROFILE%\rag-venv\Scripts\python.exe ask.py --local "hybrid search"
%USERPROFILE%\rag-venv\Scripts\python.exe ask.py --json "Who partnered with NVIDIA?" > answer.json
```

`--local` prints the top hybrid hits and spends no quota. `--no-cache` forces a
fresh model run. The command exits 3 when no model is usable, so a script can
fall back to local mode.

---

## Requirements

- Oracle 26ai Free container `oracle-26ai-free` running and healthy (port 1521)
- The `rag` Python environment at `%USERPROFILE%\rag-venv` (Python 3.11)
- A Gemini API key in `.env` - **optional**; without it the app runs in local mode

`.env` is looked up, in order:

1. `oracle-rag\.env`
2. `Desktop\.env`  ← the notebook's existing file, picked up automatically
3. `rag-oneclick\.env`
4. `%USERPROFILE%\.env`

Start from the template:

```bat
Copy-Item .env.example .env
```

`.env.example` documents every variable. The essential ones:

```ini
ORACLE_USER=vector
ORACLE_PWD=vector
ORACLE_DSN=localhost:1521/FREEPDB1
GEMINI_API_KEY=
GEMINI_MODEL=          # optional pin; otherwise models are tried in order
```

`RAG_TABLE`, `RAG_EMBEDDING_MODEL`, `RAG_RERANK_MODEL` and
`GEMINI_DAILY_LIMIT` are optional; the template shows their defaults.

---

## The notebook

`notebook/oracle_agentic_rag_hybrid_search.ipynb` is the original lab and the
source of the ten sample articles. It is safe to re-run:

- it ingests only when the table is empty, and otherwise keeps what is there;
- it skips HNSW index creation when the index already exists;
- it never drops a populated table unless you set `REBUILD_SAMPLES = True` in
  the ingest cell;
- the cleanup section only *documents* the manual `DROP TABLE` statement, so
  running the notebook cannot delete your corpus.

The notebook stores the ten articles whole, which is right for teaching. The
app's benchmark runs against the same ten articles split into **46 overlapping
chunks** (900 characters, 120 overlap). That gives retrieval more than ten
candidates without inventing new facts. To rebuild that corpus:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tools\rechunk_corpus.py
```

The tool refuses to run if chunking would break an expected answer in
`quality/questions.json`. `tools/restore_corpus.py` puts the ten whole articles
back, so after a restore run `tools/rechunk_corpus.py` again if you want the
46-chunk benchmark corpus. Either way, the corpus changes, so an old evaluation
is no longer comparable: run Quality again.

---

## The five surfaces

| Surface | What it is for |
|---|---|
| **Ask** | ReAct agent over the corpus, with a trace: tool calls, queries, sources, API calls, latency. Local mode = no LLM, zero quota. A repeated question is served from cache with no call at all. |
| **Search lab** | The same query through keyword, vector, hybrid and the local reranker at once - scores, snippets, overlap. The proof that each step earns its keep. |
| **Corpus** | Source list, chunk counts, ingest text/files, delete a source, rebuild indexes. |
| **Quality** | 40 labelled questions (single, cross-document, unanswerable) → hit@k and MRR per retrieval path, including the reranker, saved per run. |
| **Health** | Database version, container state, indexes, embedding model, key presence, per-model quota with reset time, answer cache, corpus backup. |

---

## Layout

```
oracle-rag/
  app.py                  Streamlit UI - no SQL, no prompts
  rag_core.py             the only contract: Oracle, retrieval, agent, quota
  ask.py                  terminal CLI over the same contract
  START-GUI.bat           launcher (checks venv, deps, port)
  notebook/               the original lab notebook (canonical copy)
  tools/restore_corpus.py re-ingest the ten samples from the notebook
  tools/rechunk_corpus.py split those samples into 46 retrieval chunks
  tools/export_corpus.py  write the corpus to one JSON file
  tools/import_corpus.py  restore that file, re-embedding the text
  quality/questions.json  40 labelled questions: single, cross, unanswerable
  quality/run_eval.py     hit@k / MRR harness (CLI + used by the Quality tab)
  tests/smoke_test.py     engine tests, spend zero API quota
  tests/test_app.py       headless render + live search via streamlit.testing
  tests/test_cli.py       ask.py local mode, no quota
  tests/test_db_down.py   the offline screen starts without a traceback
  data/                   quota.json, model.json, answers.json, eval-last.json (gitignored)
  logs/                   launcher logs (gitignored)
```

**Rule:** `app.py` never imports `oracledb` and never builds a prompt. If the UI
needs data, `rag_core.py` grows a function.

---

## Tests

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py --llm   REM spends 2-6 API calls
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_app.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_cli.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_db_down.py
```

`smoke_test.py` covers: settings discovery, connect, stats, the four retrieval
paths and the local reranker, score ranges, ingest, export, import, search and
delete round-trips, chunking, quota bookkeeping. It writes only to a scratch
source name and removes it afterwards. `--llm` adds one real agent question and
is the only test that spends quota. `test_cli.py` runs `ask.py` in local mode,
so it costs nothing.

---

## How retrieval works

| Path | Mechanism | Score shown |
|---|---|---|
| Keyword | Oracle Text `CONTAINS(text, :q, 1)`, ranked by `SCORE(1)` then `id` | normalised 0-1 vs best hit |
| Vector | HNSW index, `1 - VECTOR_DISTANCE(embedding, :q, COSINE)` | cosine similarity 0-1 |
| Hybrid | candidate pool from both paths, each component min-max normalised over the pool, then `w·v̂ + (1-w)·k̂` | fused 0-1 |
| Reranked | top 30 hybrid candidates rescored by a local cross-encoder | min-max normalised within the candidate pool |

`w` is the slider in the Search lab (0 = keyword only, 1 = vector only).
At either extreme the ranking is exactly the pure path, not an approximation.
Scores are computed in `rag_core.py` exactly as shown - copy the SQL and re-run it.

The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded on first use and
run on the CPU. It costs no API quota. Its displayed score is normalised within
the candidate pool, so the top hit is always 1.00; that number is for reading the
ordering, not for judging confidence. The raw logit is kept in each hit's
`detail`.

**Measured on 40 labelled questions, k = 5** (Quality tab, weight 0.6), against
46 chunks split from the ten articles. The set has 24 single-document questions,
8 cross-document questions and 8 whose answer is not in the corpus. A
cross-document hit requires every expected string to be inside the top k, not
just one of them.

| Path | hit@5 single | hit@5 cross | MRR single | MRR cross |
|---|---|---|---|---|
| Keyword | 0.79 | 0.50 | 0.57 | 0.19 |
| Vector | 1.00 | 0.75 | 0.85 | 0.27 |
| Hybrid | 1.00 | 1.00 | 0.79 | 0.33 |
| Reranked | 1.00 | 1.00 | 1.00 | 0.47 |

Each Quality run rebuilds the Oracle Text index first (about 2 seconds on this
corpus). Oracle Text scoring uses index-level statistics, so without that step
two runs over the same rows could disagree. The rebuild changes no data; it makes
the keyword numbers comparable across runs.

At k = 5 from 46 chunks, hit@5 is 5 out of 46, so it is a real test. MRR is the
more discriminating number. Cross-document questions are still the hard case:
the second required chunk often lands later in the list.

Reranking is the clear win on ordering. It puts every single-document answer at
rank 1 (MRR 1.00) and lifts cross-document MRR from 0.33 to 0.47, while hybrid
already finds all 32 answers within the top 5. In other words, fusion gets the
right chunks into the pool; the cross-encoder decides their order.

Keyword trails on cross-document questions (hit@5 0.50, MRR 0.19) because both
expected strings must appear in the top 5, and a keyword score ranks individual
chunks, not the pair.

The unanswerable questions are not scored as hits. Their vector top score
averages **0.43** (range 0.22-0.54), against **0.60** for answerable questions
(range 0.34-0.77). Those ranges overlap, so cosine similarity alone is not a safe
"I don't know" signal. Keyword scores are normalised per query, so the best match
is always 1.00 and cannot be used as confidence. Reranked scores are normalised
the same way and are also not confidence.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Database unreachable" panel | Container stopped | `docker start oracle-26ai-free`, then click **Re-check** |
| "Table is empty" | Corpus never ingested | Corpus → ingest a file, or run the notebook |
| Ask says "no model available" | Free-tier quota spent | Wait for midnight Pacific, or use **Local mode** |
| Port 8501 busy | Another Streamlit is running | Close it, or `--server.port 8502` |
| `torch` import error | Venv damaged | Re-run `START-GUI.bat` (it verifies imports) |
| First rerank query takes about a minute | The cross-encoder is downloaded once | Wait; later queries take about a second |
| Reranker will not load | First use needs network, or the model name is wrong | Connect once, or point `RAG_RERANK_MODEL` at a cached model |

---

## Data safety

The database lives in the Docker volume **`oracle-26ai-data`**, mounted at
`/opt/oracle/oradata`, and the container is set to `--restart unless-stopped`, so
it comes back with Docker. Stopping, restarting or even removing the container
keeps that volume; deleting the volume itself is what destroys the data.

The app never drops tables on its own - deleting a source removes only that
source's rows, behind a confirmation.

If the volume is ever lost, `START.bat /check` rebuilds the user and the vector
memory setting, then this puts the ten sample articles back:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tools\restore_corpus.py
```

That restores only the ten samples, as ten whole articles. For the 46-chunk
benchmark corpus, run `tools\rechunk_corpus.py` afterwards. Anything you ingest
yourself needs a backup.

The Quality tab keys a run to the corpus fingerprint, so restoring or re-chunking
marks an old evaluation stale instead of showing numbers from a different corpus.

### Backup and restore

Easiest: **Health -> Prepare corpus backup -> Download backup**. The file lands
in your Downloads folder, ready to copy somewhere else.

From the terminal, the same export:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tools\export_corpus.py
```

It writes `data/corpus-<timestamp>.json` with the text and source of every
chunk. Embeddings are not stored; they are recomputed on import, so the file
stays small. To restore:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tools\import_corpus.py data\corpus-20261105-120000.json
```

Existing sources are skipped by default, so importing twice cannot duplicate a
source. Add `--replace` to overwrite them.

---

## Free tier

Roughly **20 requests per day per model**; one agent question costs 2–6. The app
counts every call, marks exhausted models, and falls through to the next one.
Local mode costs nothing and always works.

Answers are cached in `data/answers.json`, keyed by the question **and a
fingerprint of the corpus**, so asking the same thing twice costs zero calls and
adding a document does not serve an answer that predates it. The Health tab shows
the cache size and can clear it.
