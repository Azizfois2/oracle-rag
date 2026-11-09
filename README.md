# Oracle RAG Console

A local web app for the Oracle 26ai hybrid-search corpus: **ask questions, compare
keyword vs vector vs hybrid vs reranked retrieval, manage the corpus, measure
quality, check the plumbing, expose the corpus to other agents over MCP, and keep
your discussions in Oracle** - without touching a notebook cell.

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
%USERPROFILE%\rag-venv\Scripts\python.exe ask.py --chat
```

`--local` prints the top hybrid hits and spends no quota. `--no-cache` forces a
fresh model run. The command exits 3 when no model is usable, so a script can
fall back to local mode.

`--chat` opens an interactive terminal session. It uses the same `Agent` and
`ChatHistory` as the app, so a discussion started here appears in the Streamlit
sidebar and can be reopened there. Answers stream in, follow-up context is on,
and a discussion is saved after every turn. `/help` lists the commands:

| Command | What it does |
|---|---|
| `/local` / `/remote` | switch between retrieval only and the Gemini agent |
| `/new` | start a fresh discussion |
| `/history` | list the ten most recent discussions |
| `/open N` | reopen discussion N from that list |
| `/find words` | search saved questions and answers |
| `/sources` | sources of the last answer |
| `/trace` | trace of the last answer |
| `/title [name]` | show or rename this discussion |
| `/delete` | delete this discussion, twice to confirm |
| `/quit` | leave the chat |

`--chat --local` runs the session with no model and no quota. `--no-history`
keeps the session in memory only, so a quick test leaves no rows behind.
`--steps` prints the trace after each answer. `--json` stays one-shot only.

---

## MCP server

`mcp_server.py` lets other agents query the corpus over the Model Context
Protocol. It is **read-only** and **uses no Gemini quota**. The server speaks
stdio, so it opens no network port.

| Tool | What it does |
|---|---|
| `search_corpus` | Retrieve passages. `mode` is `keyword`, `vector`, `hybrid` (default) or `rerank`. `k` is 1-10. |
| `list_sources` | Every indexed source with its chunk count. |
| `corpus_info` | Rows, sources, database version, models, indexes and corpus fingerprint. |

Check the server and the database before wiring it into a client:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe mcp_server.py --check
```

That prints corpus information and exits. A normal start looks like:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe mcp_server.py
```

Usually the MCP client starts it for you. The first search loads the local
embedding model, and `rerank` mode also loads the cross-encoder, so the first
call can take about a minute. Later calls are fast.

### OpenCode

Add this to `opencode.jsonc` in the project, or to your global OpenCode config.
Replace `<you>` with your Windows user name:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "oracle-rag": {
        "type": "local",
        "command": [
          "C:\\Users\\<you>\\rag-venv\\Scripts\\python.exe",
          "C:\\Users\\<you>\\Desktop\\oracle-rag\\mcp_server.py"
        ],
        "cwd": "C:\\Users\\<you>\\Desktop\\oracle-rag"
      }
    }
  }
}
```

Then run `opencode mcp list` and look for `oracle-rag connected`.

### Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json`, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "oracle-rag": {
      "command": "C:\\Users\\<you>\\rag-venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\<you>\\Desktop\\oracle-rag\\mcp_server.py"
      ]
    }
  }
}
```

Both clients should list `search_corpus`, `list_sources` and `corpus_info`.
The server only reads the corpus. It has no ingest, delete or ask tool, so an
agent cannot spend Gemini quota or change your data.

---

## Discussion history

Every question and answer is saved in the Oracle table `RAG_CHAT_LOG`, one row
per message. Closing the browser, restarting Streamlit or rebooting the machine
does not lose the discussion. The sidebar lists the six most recent, and opening
the app again returns to the latest one.

- **New discussion** starts a clean thread.
- Click a title to reopen it. The answer text, trace, sources and retrieval hits
  all come back.
- **Find in discussions** searches the saved questions, answers and error text.
  Matching is case-insensitive, one result per discussion, with a count, the
  matching words highlighted in the snippet, and **Show more** past the first
  four. Typed `%` and `_` are treated as ordinary text.
- **Manage this discussion** can rename the thread or download it as Markdown,
  including every source and trace line.
- **Delete** removes that thread after a confirmation.
- Health shows the discussion and message counts and can clear all of them.

History uses no Gemini quota and stays in your local database. Clearing it does
not touch the corpus, the evaluation or the answer cache. The table is created
automatically on first use.

## Streaming answers and follow-ups

Remote answers are written into the page token by token, so the text starts to
appear as the model produces it. The trace keeps filling in above, then collapses
to the elapsed time when the answer is done. A cursor marks the text while it is
still arriving, and the final message replaces it with the normal formatted
answer, including its numbered inline citations. Sources, the trace, the answer
cache and model fallback behave exactly as before. Local mode is unchanged: it
shows ranked hits, with no model and no quota.

**Follow-up context** is on by default. The last few turns of the open discussion
travel with the new question, so "who funded them?" still makes sense. The trace
records how many earlier messages were sent. Switch it off in the sidebar when
each question should stand alone. In local mode the previous question is added to
the retrieval query, which needs no model. The answer cache includes the context
in its key, so the same words asked in two different discussions cannot share an
answer by mistake.

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
| **Ask** | ReAct agent over the corpus, with the answer streaming in and a trace: tool calls, queries, sources, API calls, latency. Every source is a numbered inline citation like [1]; click one to open the exact passage and its date. Follow-up context carries the last few turns. Local mode = no LLM, zero quota. A repeated question is served from cache with no call at all. Discussions are saved in Oracle, restored on open, searchable, renameable and downloadable. |
| **Search lab** | The same query through keyword, vector, hybrid and the local reranker at once - scores, snippets, overlap. Filter every path to one source or one published-date range. The proof that each step earns its keep. |
| **Corpus** | Source list with published dates, chunk counts, ingest text/files, delete a source, rebuild indexes. |
| **Quality** | 40 labelled retrieval questions (single, cross-document, unanswerable) with hit@k and MRR per path, plus 28 labelled answer cases that check expected sources, expected facts, refusals and invented citations or citation numbers. Retrieval runs cost no quota; the live answer check is opt-in. |
| **Health** | Database version, container state, indexes, embedding model, key presence, per-model quota with reset time, answer cache, discussion history, corpus backup. |

---

## Layout

```
oracle-rag/
  app.py                  Streamlit UI - no SQL, no prompts
  rag_core.py             the only contract: Oracle, retrieval, agent, history
  ask.py                  terminal CLI over the same contract
  mcp_server.py           read-only MCP server for other agents (stdio)
  START-GUI.bat           launcher (checks venv, deps, port)
  notebook/               the original lab notebook (canonical copy)
  tools/restore_corpus.py re-ingest the ten samples from the notebook
  tools/rechunk_corpus.py split those samples into 46 retrieval chunks
  tools/add_published_dates.py add article dates to an existing corpus, in place
  tools/export_corpus.py  write the corpus to one JSON file
  tools/import_corpus.py  restore that file, re-embedding the text
  quality/questions.json  40 labelled retrieval questions
  quality/run_eval.py     hit@k / MRR harness (CLI + used by the Quality tab)
  quality/answers.json    28 labelled answer cases: source, facts, refusal
  quality/answer_eval.py  zero-quota scoring for answer quality
  quality/check_answers.py live answer check, saved to data/answers-last.json
  tests/smoke_test.py     engine tests, spend zero API quota
  tests/test_app.py       headless render + live search via streamlit.testing
  tests/test_answers.py   answer gate scoring with fake answers, no quota
  tests/test_cli.py       ask.py one-shot and interactive chat, no quota
  tests/test_mcp.py       MCP server over real stdio, no quota
  tests/test_history.py   discussions persist and rebuild, scratch table only
  tests/test_db_down.py   the offline screen starts without a traceback
  tests/run_all.py        run every zero-quota suite, one summary
  data/                   quota.json, model.json, answers.json, eval-last.json,
                          answers-last.json (gitignored)
  logs/                   launcher and test logs (gitignored)
```

**Rule:** `app.py` never imports `oracledb` and never builds a prompt. If the UI
needs data, `rag_core.py` grows a function.

---

## Tests

One command runs every zero-quota suite and prints one summary:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tests\run_all.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\run_all.py --list
%USERPROFILE%\rag-venv\Scripts\python.exe tests\run_all.py --only test_answers test_cli
```

It never runs `smoke_test.py --llm` and never runs the live answer check, so it
spends no Gemini quota. Full output for each suite is kept in
`logs/tests/<name>.log`.

The same suites can be run one at a time:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py
REM the next one spends 2-6 API calls
%USERPROFILE%\rag-venv\Scripts\python.exe tests\smoke_test.py --llm
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_app.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_answers.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_cli.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_mcp.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_history.py
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_db_down.py
```

`smoke_test.py` covers: settings discovery, connect, stats, the four retrieval
paths and the local reranker, score ranges, source and published-date filtering
(including undated rows, an inverted range and a single-day boundary),
ingest, export, import, search and delete round-trips, chunking, quota
bookkeeping, and the streaming and follow-up contract with a fake graph. It
writes only to a scratch source name and removes it afterwards. `--llm` adds one
real agent question and is the only test that spends quota. `test_answers.py`
scores fake answers against the 28 labelled cases, including missing facts,
missing sources, refusals, invented source names and invented citation numbers,
so the gate itself costs nothing. `test_cli.py` runs `ask.py` one-shot and
drives the interactive chat with piped input and a fake agent, so it costs
nothing and leaves no discussion behind. `test_mcp.py` starts the MCP server as
a child process and calls every tool over stdio, so it also costs nothing.
`test_history.py` uses the
scratch table `RAG_CHAT_LOG_TEST`, checks persistence, search, rename and export,
and drops it, so your real discussions are never touched. `test_app.py` also
asks one local question, searches and renames it, checks the download button,
drives a fake streaming answer with follow-up context and a numbered inline
citation that opens its own passage, runs the answer gate with a fake model,
reopens the app in a second session, checks that the discussion came back, and
deletes that test thread. `run_all.py` runs all seven suites in turn and prints
one pass or fail summary.

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
The source filter adds `JSON_VALUE(metadata, '$.source') = :src` inside every
query, so all four paths rank the same one-document corpus. The published-date
filter adds `JSON_VALUE(metadata, '$.published') >= :pub_from` and the matching
upper bound inside every query too. Rows without a date are left out when a
range is set, because an undated document cannot be shown to fall inside it. The
range is inclusive at both ends, so a single-day filter includes that whole day.
An inverted range is rejected with a clear error instead of returning an empty
result. Scores are computed in `rag_core.py` exactly as shown - copy the SQL and
re-run it.

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

## Published dates, citations and the answer gate

The ten sample articles carry a dateline. `published_from_text()` reads it once
and stores an ISO date such as `2026-02-14` in the chunk metadata. An existing
corpus can gain the dates without re-embedding:

```bat
REM dry run, writes nothing
%USERPROFILE%\rag-venv\Scripts\python.exe tools\add_published_dates.py
REM patch the metadata in place
%USERPROFILE%\rag-venv\Scripts\python.exe tools\add_published_dates.py --apply
```

The tool patches only the metadata column with `JSON_MERGEPATCH`, so ids,
embeddings and the corpus fingerprint stay the same. It is safe to run twice.
New ingests also read a dateline automatically, and `tools/restore_corpus.py`
and `tools/rechunk_corpus.py` carry the date forward.

Every search result the tool returns is numbered: `[1]`, `[2]`, and so on for
the whole question, even across several tool calls. The prompt tells the model
to cite that same number inline, so `[1]` in the answer always points at the
first returned chunk. `sources_from_tool()` keeps the number, the passage and
the published date, and `citation_numbers()` maps each number back to its
document. A document that appears in several chunks keeps one entry with every
number that cited it and the exact passage for each number.

In the Ask page a marker like `[1]` is a small clickable citation. Click it to
open the passage and its date right beside the sentence; hover or keyboard focus
also opens it. The source list below repeats the same numbers, so a reader can
go from the sentence to the evidence without leaving the answer. If an older
saved answer has no stored passage, the app reads the first chunks for that
source from Oracle instead.

The answer gate is separate from the retrieval benchmark. `quality/answers.json`
holds 28 labelled cases: 19 single-document, 4 cross-document and 5
unanswerable. A case passes only when every expected source is cited, every
expected fact appears in the text, unanswerable questions are refused, and every
citation names a real document and a number the tool returned. Scoring is local
and costs no quota:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe tests\test_answers.py
```

A live run asks Gemini, so it is opt-in and small by default:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe quality\check_answers.py --dry-run
%USERPROFILE%\rag-venv\Scripts\python.exe quality\check_answers.py --limit 5
%USERPROFILE%\rag-venv\Scripts\python.exe quality\check_answers.py --all
```

Progress is saved to `data/answers-last.json` after every answer, so a quota
stop keeps what was already scored. `--resume` skips the questions in that file
and asks only the rest, which lets the full 28-case gate run across several
quota days:

```bat
%USERPROFILE%\rag-venv\Scripts\python.exe quality\check_answers.py --resume
%USERPROFILE%\rag-venv\Scripts\python.exe quality\check_answers.py --all --resume
```

`--resume` refuses a saved run from a different corpus, so results from two
corpora are never mixed. The Quality tab offers the same check with a
questions-per-run limit and shows the estimated call cost above the button
(about 2 to 4 calls per question). Each run is saved to `data/answers-last.json`,
which is gitignored and contains no key.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Database unreachable" panel | Container stopped | `docker start oracle-26ai-free`, then click **Re-check** |
| "Table is empty" | Corpus never ingested | Corpus → ingest a file, or run the notebook |
| Ask says "no model available" | Free-tier quota spent | Wait for midnight Pacific, or use **Local mode** |
| Ask fails with "timed out" | A Gemini request stalled, or a database read stalled | Try again in a minute; the agent already falls back to the next model. If it repeats, check the container or use **Local mode** |
| Port 8501 busy | Another Streamlit is running | Close it, or `--server.port 8502` |
| `torch` import error | Venv damaged | Re-run `START-GUI.bat` (it verifies imports) |
| First rerank query takes about a minute | The cross-encoder is downloaded once | Wait; later queries take about a second |
| Reranker will not load | First use needs network, or the model name is wrong | Connect once, or point `RAG_RERANK_MODEL` at a cached model |
| MCP client lists no tools | Wrong Python path, or the server cannot start | Run `mcp_server.py --check`; then use the full path in the client config |
| First MCP search is slow | The local embedding model loads once | Wait; later searches are fast |
| Discussions do not appear | The history table could not be created | Use the `vector` user, which owns the corpus table, and keep the database reachable |

---

## Data safety

The database lives in the Docker volume **`oracle-26ai-data`**, mounted at
`/opt/oracle/oradata`, and the container is set to `--restart unless-stopped`, so
it comes back with Docker. Stopping, restarting or even removing the container
keeps that volume; deleting the volume itself is what destroys the data.

The app never drops tables on its own - deleting a source removes only that
source's rows, behind a confirmation. Discussion history lives in the same
volume, in `RAG_CHAT_LOG`. It is separate from the corpus backup: restoring the
corpus does not restore old discussions, and clearing discussions does not touch
the corpus.

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
