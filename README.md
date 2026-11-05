# Oracle RAG Console

A local web app for the Oracle 26ai hybrid-search corpus: **ask questions, compare
vector vs keyword vs hybrid retrieval, manage the corpus, measure quality, check
the plumbing** - without touching a notebook cell.

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

`RAG_TABLE`, `RAG_EMBEDDING_MODEL` and `GEMINI_DAILY_LIMIT` are optional; the
template shows their defaults.

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

---

## The five surfaces

| Surface | What it is for |
|---|---|
| **Ask** | ReAct agent over the corpus, with a trace: tool calls, queries, sources, API calls, latency. Local mode = no LLM, zero quota. A repeated question is served from cache with no call at all. |
| **Search lab** | The same query through keyword, vector and hybrid at once - scores, snippets, overlap. The proof that hybrid earns its keep. |
| **Corpus** | Source list, chunk counts, ingest text/files, delete a source, rebuild indexes. |
| **Quality** | 40 labelled questions (single, cross-document, unanswerable) → hit@k and MRR per retrieval path, saved per run. |
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

`smoke_test.py` covers: settings discovery, connect, stats, the three retrieval
paths, score ranges, ingest, export, import, search and delete round-trips,
chunking, quota bookkeeping. It writes only to a scratch source name and removes
it afterwards. `--llm` adds one real agent question and is the only test that
spends quota. `test_cli.py` runs `ask.py` in local mode, so it costs nothing.

---

## How retrieval works

| Path | Mechanism | Score shown |
|---|---|---|
| Keyword | Oracle Text `CONTAINS(text, :q, 1)`, ranked by `SCORE(1)` | normalised 0–1 vs best hit |
| Vector | HNSW index, `1 − VECTOR_DISTANCE(embedding, :q, COSINE)` | cosine similarity 0–1 |
| Hybrid | candidate pool from both paths, each component min-max normalised over the pool, then `w·v̂ + (1−w)·k̂` | fused 0–1 |

`w` is the slider in the Search lab (0 = keyword only, 1 = vector only).
At either extreme the ranking is exactly the pure path, not an approximation.
Scores are computed in `rag_core.py` exactly as shown - copy the SQL and re-run it.

**Measured on 40 labelled questions, k = 5** (Quality tab, weight 0.6). The set
has 24 single-document questions, 8 cross-document questions and 8 whose answer
is not in the corpus. A cross-document hit requires every expected string to be
inside the top k, not just one of them.

| Path | hit@5 single | hit@5 cross | MRR single | MRR cross |
|---|---|---|---|---|
| Keyword | 0.96 | 0.88 | 0.84 | 0.36 |
| Vector | 1.00 | 1.00 | 0.92 | 0.40 |
| Hybrid | 1.00 | 1.00 | 0.98 | 0.44 |

At k = 5 from ten documents, hit@5 is a low bar: half the corpus is returned.
MRR is the more discriminating number, and it shows where the difficulty is.
Cross-document questions are much harder: the second required chunk often lands
later in the list. Hybrid leads on MRR in both sets, but the margin over vector
is small.

The unanswerable questions are not scored as hits. Their vector top score
averages **0.41** (range 0.20-0.53), against **0.53** for answerable questions
(range 0.27-0.76). Those ranges overlap, so cosine similarity alone is not a
safe "I don't know" signal. Keyword scores are normalised per query, so the best
match is always 1.00 and cannot be used as confidence.

Hybrid does not win by a large margin on this ten-document sample. That is the
honest result, and the Search lab shows why: at this size the vector pool already
holds the answer, so fusion has more to do with ordering than with finding. The
keyword path is what fusion rescues when it misses.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Database unreachable" panel | Container stopped | `docker start oracle-26ai-free`, then click **Re-check** |
| "Table is empty" | Corpus never ingested | Corpus → ingest a file, or run the notebook |
| Ask says "no model available" | Free-tier quota spent | Wait for midnight Pacific, or use **Local mode** |
| Port 8501 busy | Another Streamlit is running | Close it, or `--server.port 8502` |
| `torch` import error | Venv damaged | Re-run `START-GUI.bat` (it verifies imports) |

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

That restores only the ten samples. Anything you ingest yourself needs a backup.

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
