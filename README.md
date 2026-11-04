# Oracle RAG Console

A local web app for the Oracle 26ai hybrid-search corpus: **ask questions, compare
vector vs keyword vs hybrid retrieval, manage the corpus, measure quality, check
the plumbing** — without touching a notebook cell.

Vision, principles and roadmap: [`VISION.md`](VISION.md).

---

## Quick start

```bat
START-GUI.bat
```

That checks the venv, installs the app dependencies if needed, starts Streamlit on
port 8501 and opens the browser. The Oracle container must be running.

Manual start:

```bat
%USERPROFILE%\rag-venv\Scripts\streamlit.exe run app.py --server.port 8501
```

Then open <http://localhost:8501>.

---

## Requirements

- Oracle 26ai Free container `oracle-26ai-free` running and healthy (port 1521)
- The `rag` Python environment at `%USERPROFILE%\rag-venv` (Python 3.11)
- A Gemini API key in `.env` — **optional**; without it the app runs in local mode

`.env` is looked up, in order:

1. `oracle-rag\.env`
2. `Desktop\.env`  ← the notebook's existing file, picked up automatically
3. `rag-oneclick\.env`
4. `%USERPROFILE%\.env`

Contents:

```ini
ORACLE_USER=vector
ORACLE_PWD=vector
ORACLE_DSN=localhost:1521/FREEPDB1
GEMINI_API_KEY=
GEMINI_MODEL=          # optional pin; otherwise models are tried in order
```

---

## The five surfaces

| Tab | What it is for |
|---|---|
| **Ask** | ReAct agent over the corpus, with a trace: tool calls, queries, sources, API calls, latency. Local mode = no LLM, zero quota. |
| **Search lab** | The same query through keyword, vector and hybrid at once — scores, snippets, overlap. The proof that hybrid earns its keep. |
| **Corpus** | Source list, chunk counts, ingest text/files, delete a source, rebuild indexes. |
| **Quality** | Labelled question set → hit@k and MRR per retrieval path, saved per run. |
| **Health** | Database version, container state, indexes, embedding model, key presence, per-model quota with reset time. |

---

## Layout

```
oracle-rag/
├─ app.py                 Streamlit UI — no SQL, no prompts
├─ rag_core.py            the only contract: Oracle, retrieval, agent, quota
├─ START-GUI.bat          launcher (checks venv, deps, port)
├─ notebook/              the original lab notebook (canonical copy)
├─ quality/questions.json labelled questions + expected source
├─ quality/run_eval.py    hit@k / MRR harness (CLI + used by the Quality tab)
├─ tests/smoke_test.py    engine tests, spend zero API quota
├─ tests/test_app.py      headless render test via streamlit.testing
├─ data/                  quota.json, model.json, eval-last.json (gitignored)
└─ logs/                  launcher logs (gitignored)
```

**Rule:** `app.py` never imports `oracledb` and never builds a prompt. If the UI
needs data, `rag_core.py` grows a function.

---

## Tests

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py --llm   REM spends 1-5 API calls
%USERPROFILE%\rag-venv\Scripts\python.exe -m pytest tests -q
```

`smoke_test.py` covers: settings discovery, connect, stats, the three retrieval
paths, score ranges, ingest → search → delete round-trip, chunking, quota
bookkeeping. It writes only to a scratch source name and removes it afterwards.

---

## How retrieval works

| Path | Mechanism | Score shown |
|---|---|---|
| Keyword | Oracle Text `CONTAINS(text, :q, 1)`, ranked by `SCORE(1)` | normalised 0–1 vs best hit |
| Vector | HNSW index, `1 − VECTOR_DISTANCE(embedding, :q, COSINE)` | cosine similarity 0–1 |
| Hybrid | candidate pool from both paths, fused: `w·vector + (1−w)·keyword` | fused 0–1 |

`w` is the slider in the Search lab (0 = keyword only, 1 = vector only).
Scores are computed in `rag_core.py` exactly as shown — copy the SQL and re-run it.

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

The Oracle container has **no mounted volume**. `docker stop/start/restart` is
safe; `docker rm` or deleting the container in Docker Desktop destroys the
database. The app never drops tables on its own — deleting a source removes only
that source's rows, behind a confirmation.

---

## Free tier

Roughly **20 requests per day per model**; one agent question costs 2–6. The app
counts every call, marks exhausted models, and falls through to the next one.
Local mode costs nothing and always works.
