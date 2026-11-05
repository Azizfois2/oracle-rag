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
| **Quality** | Labelled question set → hit@k and MRR per retrieval path, saved per run. |
| **Health** | Database version, container state, indexes, embedding model, key presence, per-model quota with reset time. |

---

## Layout

```
oracle-rag/
├─ app.py                 Streamlit UI - no SQL, no prompts
├─ rag_core.py            the only contract: Oracle, retrieval, agent, quota
├─ START-GUI.bat          launcher (checks venv, deps, port)
├─ notebook/              the original lab notebook (canonical copy)
├─ tools/restore_corpus.py re-ingest the ten samples from the notebook
├─ quality/questions.json labelled questions + expected source
├─ quality/run_eval.py    hit@k / MRR harness (CLI + used by the Quality tab)
├─ tests/smoke_test.py    engine tests, spend zero API quota
├─ tests/test_app.py      headless render + live search via streamlit.testing
├─ tests/test_db_down.py  the offline screen starts without a traceback
├─ data/                  quota.json, model.json, answers.json, eval-last.json (gitignored)
└─ logs/                  launcher logs (gitignored)
```

**Rule:** `app.py` never imports `oracledb` and never builds a prompt. If the UI
needs data, `rag_core.py` grows a function.

---

## Tests

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py --llm   REM spends 2-6 API calls
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_app.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_db_down.py
```

`smoke_test.py` covers: settings discovery, connect, stats, the three retrieval
paths, score ranges, ingest → search → delete round-trip, chunking, quota
bookkeeping. It writes only to a scratch source name and removes it afterwards.
`--llm` adds one real agent question and is the only test that spends quota.

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

**Measured on the 12 labelled questions, k = 5** (Quality tab, weight 0.6):

| Path | hit@5 | MRR | notes |
|---|---|---|---|
| Keyword | 0.92 | 0.83 | misses the Oracle OCI question entirely |
| Vector | 1.00 | 0.96 | finds everything, ranks first 11/12 |
| Hybrid | 1.00 | 0.96 | recovers the keyword miss, matches vector on rank |

Hybrid does not beat vector on this ten-document sample. That is the honest
result, and the Search lab shows why: at this size the vector pool already holds
the answer, so fusion has nothing extra to add. The keyword path is what the
fusion rescues, and the numbers say so.

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

---

## Free tier

Roughly **20 requests per day per model**; one agent question costs 2–6. The app
counts every call, marks exhausted models, and falls through to the next one.
Local mode costs nothing and always works.

Answers are cached in `data/answers.json`, keyed by the question **and a
fingerprint of the corpus**, so asking the same thing twice costs zero calls and
adding a document does not serve an answer that predates it. The Health tab shows
the cache size and can clear it.
