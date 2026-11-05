"""app.py - Oracle RAG Console.

Streamlit UI. Rule: this file contains no SQL and no prompt strings. Everything
that touches Oracle or Gemini lives in rag_core.py.
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st

import rag_core as rc
from quality.run_eval import evaluate, load_last, load_questions, save_run

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
METHODS = ("keyword", "vector", "hybrid")
LABEL = {"keyword": "Keyword", "vector": "Vector", "hybrid": "Hybrid"}
NOTE = {
    "keyword": "Oracle Text · CONTAINS",
    "vector": "HNSW · cosine",
    "hybrid": "weighted fusion",
}
GLYPH = {"search": "search", "sources": "read", "answer": "write", "note": "note"}
LAB_SQL = """-- keyword: Oracle Text, ranked by its own score
SELECT id, text, metadata, SCORE(1) AS kw
  FROM TECH_NEWS_COLLECTION
 WHERE CONTAINS(text, :expr, 1) > 0
 ORDER BY kw DESC
 FETCH FIRST 20 ROWS ONLY;

-- vector: HNSW index, cosine distance
SELECT id, text, metadata, 1 - VECTOR_DISTANCE(embedding, :q, COSINE) AS sim
  FROM TECH_NEWS_COLLECTION
 ORDER BY VECTOR_DISTANCE(embedding, :q, COSINE)
 FETCH APPROX FIRST 20 ROWS ONLY;

-- hybrid: computed in rag_core.hybrid_search
--   score = w * sim + (1 - w) * normalized_keyword_score"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --ink:#1B1816; --ink2:#6B615A; --ink3:#948A82;
  --line:#E4DBD4; --panel:#F3EDE8; --paper:#FAF7F4;
  --accent:#B23A2B; --kw:#8A5A0B; --vec:#20609C; --hyb:#1E6E48;
}

html, body, .stApp, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"]{
  font-family: Inter, "Segoe UI", system-ui, sans-serif;
}
.stApp{ background:var(--paper); color:var(--ink); }
.block-container{ padding-top:1.9rem; padding-bottom:4rem; max-width:1240px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"]{ display:none; }

.appbar{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
  border-bottom:1px solid var(--line); padding-bottom:.7rem; }
.appbar .eyebrow{ font-family:"IBM Plex Mono",monospace; font-size:.66rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ink3); }
.appbar h1{ font-size:1.45rem; font-weight:600; margin:.1rem 0 0; letter-spacing:-.02em; }
.appbar .meta{ font-family:"IBM Plex Mono",monospace; font-size:.71rem;
  color:var(--ink2); text-align:right; line-height:1.6; font-variant-numeric:tabular-nums; }

.dot{ display:inline-block; width:6px; height:6px; border-radius:50%;
  background:var(--ink3); margin-right:.4rem; vertical-align:1px; }
.dot.ok{ background:var(--hyb); } .dot.bad{ background:var(--accent); }
.dot.warn{ background:#9A6A00; }

h2{ font-size:1.08rem; font-weight:600; letter-spacing:-.01em; margin:.1rem 0 .45rem; }
h3{ font-size:.88rem; font-weight:600; margin:.2rem 0 .35rem; }
p, li{ font-size:.89rem; line-height:1.6; }
.small{ font-size:.77rem; color:var(--ink2); line-height:1.55; }
.mono{ font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; }
.kw-t{ color:var(--kw); } .vec-t{ color:var(--vec); } .hyb-t{ color:var(--hyb); }

button[data-baseweb="tab"]{ font-size:.82rem; font-weight:500; color:var(--ink2); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
div[data-baseweb="tab-highlight"]{ background:var(--accent); height:2px; }
div[data-baseweb="tab-border"]{ background:var(--line); }

.lab-head{ display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:1px solid var(--line); padding-bottom:.35rem; }
.lab-head .name{ font-weight:600; font-size:.86rem; }
.lab-head .name i{ display:inline-block; width:7px; height:7px; border-radius:2px;
  margin-right:.4rem; vertical-align:1px; }
.lab-head .lat{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; color:var(--ink3); }

.row{ padding:.55rem 0 .5rem; border-bottom:1px solid var(--line); }
.row-top{ display:flex; align-items:baseline; gap:.45rem; }
.rank{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; font-weight:500; width:.9rem; }
.src{ font-family:"IBM Plex Mono",monospace; font-size:.7rem; color:var(--ink2);
  flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.score{ font-family:"IBM Plex Mono",monospace; font-size:.73rem; font-variant-numeric:tabular-nums; }
.track{ height:2px; background:var(--line); margin:.35rem 0 .4rem; }
.track>i{ display:block; height:100%; }
.snip{ font-size:.79rem; line-height:1.5; }
.tags{ margin-top:.3rem; display:flex; gap:.28rem; flex-wrap:wrap; }
.tag{ font-family:"IBM Plex Mono",monospace; font-size:.6rem; letter-spacing:.05em;
  text-transform:uppercase; color:var(--ink2); border:1px solid var(--line);
  padding:.04rem .3rem; border-radius:2px; }

i.kw{ background:var(--kw); } i.vec{ background:var(--vec); } i.hyb{ background:var(--hyb); }

.panel{ border:1px solid var(--line); background:#FFFDFB; padding:.7rem .85rem; border-radius:3px; }
.panel .k{ font-family:"IBM Plex Mono",monospace; font-size:.62rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink3); }
.panel .v{ font-size:.85rem; }
.strip{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:1rem; }

section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.5rem; }
.side-title{ font-size:1rem; font-weight:600; letter-spacing:-.01em; }
.side-sub{ font-family:"IBM Plex Mono",monospace; font-size:.62rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink3); margin-top:.15rem; }
.kv{ display:flex; justify-content:space-between; gap:.6rem; font-size:.73rem; padding:.13rem 0; }
.kv span{ color:var(--ink3); } .kv b{ font-weight:500; font-family:"IBM Plex Mono",monospace; }
.bar{ height:3px; background:#E4DBD4; margin:.28rem 0 .1rem; }
.bar>i{ display:block; height:100%; background:var(--accent); }

[data-testid="stChatMessage"]{ background:transparent; padding:.15rem 0; }
[data-testid="stChatMessageContent"]{ font-size:.89rem; }

.stButton>button{ border-radius:3px; border:1px solid var(--line); background:#FFFDFB;
  color:var(--ink); font-size:.8rem; font-weight:500; padding:.28rem .7rem; }
.stButton>button:hover{ border-color:#C9BDB4; background:#F7F1EC; }
.stButton>button[kind="primary"]{ background:var(--accent); border-color:var(--accent); color:#FFF8F6; }
.stButton>button[kind="primary"]:hover{ background:#9C3123; border-color:#9C3123; }
[data-testid="stExpander"] details{ border:1px solid var(--line); border-radius:3px; background:#FFFDFB; }
hr{ border-color:var(--line); }
"""


# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def boot() -> tuple[rc.Settings, rc.RagEngine, rc.Agent]:
    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    agent = rc.Agent(engine, settings)
    engine.warm_async()
    return settings, engine, agent


# --------------------------------------------------------------------------- #
# markup helpers
# --------------------------------------------------------------------------- #
def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def row_html(rank: int, hit: rc.Hit, method: str, tags: tuple[str, ...] = ()) -> str:
    width = max(2, min(100, round(hit.score * 100)))
    tags_block = ""
    if tags:
        inner = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags)
        tags_block = f'<div class="tags">{inner}</div>'
    return (
        f'<div class="row">'
        f'<div class="row-top">'
        f'<span class="rank {method}-t">{rank}</span>'
        f'<span class="src">{esc(hit.source)}</span>'
        f'<span class="score {method}-t">{hit.score:.3f}</span>'
        f"</div>"
        f'<div class="track"><i class="{method}"></i></div>'
        f'<div class="snip">{esc(hit.snippet(190))}</div>'
        f"{tags_block}"
        f"</div>"
    )


def lab_head(method: str, ms: float, count: int) -> str:
    return (
        f'<div class="lab-head"><div class="name"><i class="{method}"></i>{LABEL[method]}</div>'
        f'<div class="lat">{count} hits · {ms:.0f} ms</div></div>'
    )


def panel(label: str, value: str) -> str:
    return f'<div><div class="k">{esc(label)}</div><div class="v mono">{esc(value)}</div></div>'


def dot(state: str) -> str:
    return {"ok": "ok", "warn": "warn"}.get(state, "bad")


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(settings: rc.Settings, engine: rc.RagEngine, agent: rc.Agent,
                   stats: dict, db: dict) -> None:
    quota = agent.quota
    models = settings.models()
    status = quota.status()
    spent = status["spent"]
    budget = settings.daily_limit * max(1, len(models))

    with st.sidebar:
        st.markdown(
            '<div class="side-title">Oracle RAG Console</div>'
            '<div class="side-sub">26ai · local</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

        db_state = "ok" if db.get("ok") else "bad"
        db_label = "connected" if db.get("ok") else "offline"
        rows = stats.get("rows", 0)
        st.markdown(
            f'<div class="kv"><span>database</span><b><span class="dot {dot(db_state)}"></span>{db_label}</b></div>'
            f'<div class="kv"><span>table</span><b>{esc(settings.table)}</b></div>'
            f'<div class="kv"><span>chunks</span><b>{rows}</b></div>'
            f'<div class="kv"><span>ping</span><b>{db.get("ms", "?")} ms</b></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('<div class="k mono" style="font-size:.62rem;letter-spacing:.11em;'
                    'text-transform:uppercase;color:var(--ink3)">mode</div>',
                    unsafe_allow_html=True)
        st.toggle(
            "Local mode",
            key="local_mode",
            help="Retrieval only, no model call, zero quota. Useful when the free tier is spent.",
        )
        if not settings.has_llm:
            st.markdown('<p class="small">No Gemini key found, so only local mode is available. '
                        'Add <span class="mono">GEMINI_API_KEY</span> to '
                        f'<span class="mono">{esc(settings.env_path or ".env")}</span>.</p>',
                        unsafe_allow_html=True)

        st.divider()
        pct = min(1.0, spent / budget) if budget else 0.0
        st.markdown(
            f'<div class="kv"><span>quota today</span><b>{spent} / {budget}</b></div>'
            f'<div class="bar"><i style="width:{pct*100:.0f}%"></i></div>'
            f'<div class="kv"><span>free tier</span><b>{settings.daily_limit} / model</b></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Per model"):
            for name in models:
                entry = status["models"].get(name, {"calls": 0, "exhausted": False})
                flag = "exhausted" if entry["exhausted"] else f'{entry["calls"]} calls'
                st.markdown(
                    f'<div class="kv"><span class="mono">{esc(name)}</span><b>{flag}</b></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Reset counters", key="reset_quota"):
                quota.clear()
                st.rerun()

        st.divider()
        st.markdown(
            '<div class="kv"><span>model</span><b>'
            f'{esc(agent.current_model() or "none available")}</b></div>'
            f'<div class="kv"><span>embeddings</span><b>MiniLM-L12</b></div>'
            f'<div class="kv"><span>k</span><b>384 dims</b></div>'
            f'<div class="kv"><span>cache</span><b>{agent.cache.stats()["entries"]} answers</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p class="small" style="margin-top:.6rem">'
            '<a href="http://localhost:8888/lab?token=rag" target="_blank">JupyterLab</a>'
            " · "
            '<a href="http://localhost:8888/lab/tree/notebook/oracle_agentic_rag_hybrid_search.ipynb?token=rag" '
            'target="_blank">lab notebook</a></p>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #
SUGGESTIONS = (
    "Which companies are putting money into hybrid search, and why?",
    "What did AWS announce for multi-agent orchestration?",
    "How many monthly downloads did LangGraph reach?",
    "Which open-weight models landed within 48 hours of each other?",
)


def render_answer_body(answer: rc.Answer) -> None:
    if answer.local:
        st.markdown('<p class="small">Retrieval only. No model was called.</p>',
                    unsafe_allow_html=True)
        if not answer.hits:
            st.markdown('<p class="small">Nothing matched. Try different words.</p>',
                        unsafe_allow_html=True)
        for index, hit in enumerate(answer.hits, start=1):
            st.markdown(row_html(index, hit, "hybrid"), unsafe_allow_html=True)
    else:
        st.markdown(answer.text)
        if answer.cached:
            st.markdown(
                '<p class="small">Cached: the same question against the same corpus, '
                "answered earlier. No model was called.</p>",
                unsafe_allow_html=True,
            )

    if answer.cached:
        label = f"cached · 0 API calls · {answer.model}"
    elif answer.local:
        label = f"0 API calls · {answer.elapsed:.1f}s"
    else:
        label = f"{answer.calls} API calls · {answer.elapsed:.1f}s · {answer.model}"
    with st.expander(label):
        for step in answer.steps:
            glyph = GLYPH.get(step.kind, step.kind)
            detail = f" {esc(step.detail)}" if step.detail else ""
            st.markdown(
                f'<div class="mono small"><b>{esc(step.label)}</b>'
                f'<span style="color:var(--ink3)"> · {esc(glyph)}</span>{detail}</div>',
                unsafe_allow_html=True,
            )
        if answer.sources:
            st.markdown('<p class="small">Sources returned by the retrieval tool</p>',
                        unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(answer.sources).rename(
                    columns={"source": "source", "relevance": "relevance"}),
                hide_index=True,
                column_config={
                    "source": st.column_config.TextColumn("source", width="large"),
                    "relevance": st.column_config.ProgressColumn(
                        "relevance", min_value=0.0, max_value=1.0, format="%.3f"),
                },
            )


def answer_flow(question: str, settings: rc.Settings, agent: rc.Agent) -> None:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    local = st.session_state.local_mode or not settings.has_llm
    with st.chat_message("assistant"):
        status = st.status(
            "Reading the corpus" if local else "Searching the corpus",
            expanded=True,
        )

        def on_step(kind: str, text: str) -> None:
            status.write(f"`{GLYPH.get(kind, kind)}` {text}")

        try:
            answer = (agent.ask_local(question) if local
                      else agent.ask(question, on_step=on_step))
            status.update(label=f"Answered in {answer.elapsed:.1f}s",
                          state="complete", expanded=False)
        except rc.QuotaExhausted as exc:
            status.update(label="Quota exhausted", state="error", expanded=False)
            st.warning(f"{exc}. Switch on Local mode in the sidebar to keep working.")
            st.session_state.chat.append({"role": "error", "content": str(exc)})
            return
        except rc.RagError as exc:
            status.update(label="Failed", state="error", expanded=False)
            st.error(str(exc))
            st.session_state.chat.append({"role": "error", "content": str(exc)})
            return

        render_answer_body(answer)
    st.session_state.chat.append({"role": "assistant", "answer": answer})


def tab_ask(settings: rc.Settings, agent: rc.Agent, stats: dict) -> None:
    chat = st.session_state.chat
    if not chat and not st.session_state.get("pending"):
        st.markdown(
            f'<p class="small">{stats.get("sources", 0)} documents, '
            f'{stats.get("rows", 0)} chunks, indexed in Oracle 26ai. '
            "Answers cite the document they came from.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
        columns = st.columns(2)
        for index, suggestion in enumerate(SUGGESTIONS):
            if columns[index % 2].button(suggestion, key=f"sug{index}"):
                st.session_state.pending = suggestion
                st.rerun()

    for item in chat:
        if item["role"] == "user":
            with st.chat_message("user"):
                st.markdown(item["content"])
        elif item["role"] == "error":
            with st.chat_message("assistant"):
                st.error(item["content"])
        else:
            with st.chat_message("assistant"):
                render_answer_body(item["answer"])

    queued = st.session_state.pop("pending", None)
    if queued:
        answer_flow(queued, settings, agent)

    prompt = st.chat_input("Ask about the corpus")
    if prompt:
        answer_flow(prompt, settings, agent)


# --------------------------------------------------------------------------- #
# search lab
# --------------------------------------------------------------------------- #
def tab_lab(engine: rc.RagEngine) -> None:
    with st.form("lab", border=False):
        query = st.text_input(
            "Query",
            value=st.session_state.get("lab_query", "companies building hybrid search"),
            placeholder="e.g. which startup is challenging Pinecone?",
        )
        left, middle, right = st.columns([1, 1.6, 1])
        k = left.slider("Results per path", 3, 10, 5)
        weight = middle.slider(
            "Hybrid weight", 0.0, 1.0, 0.6, 0.05,
            help="0 = keyword only, 1 = vector only.",
        )
        right.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        submitted = right.form_submit_button("Compare paths", type="primary")

    if submitted and query.strip():
        st.session_state.lab_query = query.strip()
        with st.spinner("Querying keyword, vector and hybrid"):
            try:
                st.session_state.bundle = engine.search_all(
                    query.strip(), k=int(k), weight=float(weight))
                st.session_state.bundle_at = time.strftime("%H:%M:%S")
            except rc.RagError as exc:
                st.error(str(exc))
                return

    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.markdown(
            '<p class="small">One query, three retrieval paths, side by side. '
            "The same scores the app uses everywhere else.</p>",
            unsafe_allow_html=True,
        )
        return

    memberships = bundle.all_keys()
    shared = bundle.consensus()
    st.markdown(
        f'<p class="small"><span class="mono">{esc(bundle.query)}</span> · '
        f'{bundle.total} documents across the three paths · '
        f'{len(shared)} found by at least two · '
        f'fusion took <span class="mono">{bundle.timings["hybrid"]:.0f} ms</span></p>',
        unsafe_allow_html=True,
    )

    columns = st.columns(3, gap="large")
    for column, method in zip(columns, METHODS):
        hits = getattr(bundle, method)
        with column:
            st.markdown(lab_head(method, bundle.timings[method], len(hits)),
                        unsafe_allow_html=True)
            if not hits:
                st.markdown(
                    '<p class="small" style="padding-top:.6rem">No matches on this path. '
                    "That is a result too.</p>",
                    unsafe_allow_html=True,
                )
            for rank, hit in enumerate(hits, start=1):
                others = tuple(
                    f"also {other}" for other in METHODS
                    if other != method and hit.key in memberships[other]
                )
                st.markdown(row_html(rank, hit, method, others), unsafe_allow_html=True)

    with st.expander("How these scores are computed"):
        st.markdown(
            '<p class="small">Keyword scores are normalised against the best match in the '
            "same pool. Vector scores are cosine similarity. Hybrid is a weighted sum of "
            "both, so a document found by only one path can still rank.</p>",
            unsafe_allow_html=True,
        )
        st.code(LAB_SQL, language="sql")


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def tab_corpus(engine: rc.RagEngine, settings: rc.Settings) -> None:
    flash = st.session_state.pop("corpus_flash", None)
    if flash:
        st.success(flash)

    stats = engine.stats()
    has_table = bool(stats["columns"])
    sources = engine.sources()

    st.markdown(
        '<div class="panel"><div class="strip">'
        + panel("sources", stats["sources"])
        + panel("chunks", stats["rows"])
        + panel("avg chunk", f'{stats["avg_chars"]} ch')
        + panel("table size", f'{stats["size_mb"]} MB')
        + panel("indexes", ", ".join(sorted({i["type"] for i in stats["indexes"]})) or "none")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    if has_table:
        hnsw = next((i for i in stats["indexes"] if i["name"].upper() == rc.HNSW_INDEX.upper()), None)
        text = next((i for i in stats["indexes"] if i["name"].upper() == rc.TEXT_INDEX.upper()), None)
        hnsw_text = f'present ({hnsw["status"]})' if hnsw else "missing"
        text_text = f'present ({text["status"]})' if text else "missing"
        st.markdown(
            f'<p class="small"><span class="mono">{esc(rc.HNSW_INDEX)}</span> '
            f"{esc(hnsw_text)} · <span class=\"mono\">{esc(rc.TEXT_INDEX)}</span> "
            f"{esc(text_text)}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.markdown('<h3>Sources</h3>', unsafe_allow_html=True)
    if sources:
        st.dataframe(
            pd.DataFrame(sources).rename(
                columns={"source": "source", "chunks": "chunks", "avg_chars": "avg chars"}),
            hide_index=True,
            column_config={
                "source": st.column_config.TextColumn("source", width="large"),
                "chunks": st.column_config.NumberColumn("chunks", width="small"),
                "avg chars": st.column_config.NumberColumn("avg chars", width="small"),
            },
        )
    else:
        st.markdown(
            '<p class="small">No documents yet. Add one below, or run the notebook once to '
            "load the ten sample tech-news articles.</p>",
            unsafe_allow_html=True,
        )

    with st.expander("Add documents", expanded=not sources):
        files = st.file_uploader(
            "Files", type=["txt", "md", "pdf"], accept_multiple_files=True,
            help="Plain text, markdown or PDF. PDFs need pypdf, already installed.",
        )
        pasted = st.text_area("Or paste text", height=130,
                              placeholder="Paste an article, a note, anything worth searching.")
        col_a, col_b, col_c = st.columns([2, 1, 1])
        name = col_a.text_input("Source name", placeholder="my_article.md")
        size = col_b.number_input("Chunk size", 300, 2000, 900, 50)
        replace = col_c.checkbox("Replace existing", value=False)

        if st.button("Ingest", type="primary", key="ingest"):
            jobs: list[tuple[str, str]] = []
            for upload in files or []:
                try:
                    jobs.append((upload.name, rc.extract_text(upload.name, upload.getvalue())))
                except rc.RagError as exc:
                    st.error(f"{upload.name}: {exc}")
            if pasted.strip():
                jobs.append((name.strip() or "pasted.md", pasted))
            if not jobs:
                st.warning("Nothing to ingest: add a file or paste some text.")
                return

            added, failed = [], []
            for source_name, body in jobs:
                try:
                    result = engine.ingest(
                        body, source_name, chunk_size=int(size), replace=replace)
                    added.append(f'{result["source"]} ({result["chunks"]} chunks)')
                except rc.RagError as exc:
                    failed.append(f"{source_name}: {exc}")
            if added:
                st.session_state.corpus_flash = "Ingested " + ", ".join(added)
            for message in failed:
                st.error(message)
            if added:
                st.rerun()

    if sources:
        with st.expander("Remove a source"):
            target = st.selectbox("Source", [s["source"] for s in sources], key="delete_src")
            chunks = next((s["chunks"] for s in sources if s["source"] == target), 0)
            confirm = st.checkbox(
                f"I understand this deletes {chunks} chunks of {target}", key="delete_ok")
            if st.button("Delete", disabled=not confirm, key="delete_btn"):
                removed = engine.delete_source(target)
                st.session_state.corpus_flash = f"Deleted {removed} chunks of {target}"
                st.rerun()

    with st.expander("Rebuild indexes"):
        st.markdown(
            '<p class="small">Recreates the HNSW or Oracle Text index if either went '
            "missing. Never touches the documents.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Reindex", key="reindex"):
            try:
                report = engine.reindex()
                st.session_state.corpus_flash = " · ".join(report)
                st.rerun()
            except rc.RagError as exc:
                st.error(str(exc))


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #
def tab_quality(engine: rc.RagEngine) -> None:
    questions = load_questions()
    st.markdown(
        f'<p class="small">{len(questions)} labelled questions, run against all three paths. '
        "A hit means the expected string appears in the retrieved chunk or its source. "
        "No model is called, so a full run costs nothing.</p>",
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns([1, 2, 1])
    k = left.number_input("k", 3, 10, 5, key="eval_k")
    weight = middle.slider("Hybrid weight", 0.0, 1.0, 0.6, 0.05, key="eval_w")
    right.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    if right.button("Run evaluation", type="primary"):
        with st.spinner("Retrieving all questions through all paths"):
            try:
                result = evaluate(engine, questions, k=int(k), weight=float(weight))
                save_run(result)
                st.session_state.eval = result
            except rc.RagError as exc:
                st.error(str(exc))
                return

    result = st.session_state.get("eval") or load_last()
    if not result:
        st.markdown('<p class="small">No run yet. Press Run evaluation.</p>',
                    unsafe_allow_html=True)
        return

    summary = pd.DataFrame([
        {"path": LABEL[m], "hit@k": result["summary"][m]["hit"],
         "MRR": result["summary"][m]["mrr"],
         "found": f"{result['summary'][m]['found']}/{len(result['rows'])}"}
        for m in METHODS
    ])
    st.dataframe(
        summary, hide_index=True,
        column_config={
            "path": st.column_config.TextColumn("path", width="small"),
            "hit@k": st.column_config.ProgressColumn(
                "hit@k", min_value=0.0, max_value=1.0, format="%.2f"),
            "MRR": st.column_config.ProgressColumn(
                "MRR", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    st.markdown(
        f'<p class="small">k = {result["k"]} · hybrid weight {result["weight"]} · '
        f'{result["elapsed"]}s · run at {result["at"].replace("T", " ")}</p>',
        unsafe_allow_html=True,
    )

    rows = []
    for row in result["rows"]:
        rows.append({
            "question": row["question"],
            "keyword": "miss" if row["keyword"] is None else f'#{row["keyword"]}',
            "vector": "miss" if row["vector"] is None else f'#{row["vector"]}',
            "hybrid": "miss" if row["hybrid"] is None else f'#{row["hybrid"]}',
            "expected": row["expect"],
        })
    with st.expander("Per question"):
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
def tab_health(settings: rc.Settings, engine: rc.RagEngine, agent: rc.Agent,
               stats: dict, db: dict) -> None:
    flash = st.session_state.pop("cache_flash", None)
    if flash:
        st.success(flash)
    docker = rc.container_status()
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<h3>Database</h3>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel"><div class="strip">'
            + panel("state", "connected" if db.get("ok") else "offline")
            + panel("ping", f'{db.get("ms", "?")} ms')
            + panel("rows", stats.get("rows", 0))
            + "</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="small mono" style="margin-top:.5rem">{esc(stats.get("version", "?"))}</p>'
            f'<p class="small mono">{esc(settings.dsn)} · user {esc(settings.user)}</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<h3 style="margin-top:1.1rem">Container</h3>', unsafe_allow_html=True)
        state = docker.get("state", "unknown")
        literal = {"running": "ok", "exited": "warn", "missing": "bad"}.get(state, "warn")
        st.markdown(
            f'<div class="kv"><span>name</span><b>{esc(docker.get("name"))}</b></div>'
            f'<div class="kv"><span>state</span><b><span class="dot {dot(literal)}"></span>{esc(state)}</b></div>'
            f'<div class="kv"><span>health</span><b>{esc(docker.get("health"))}</b></div>'
            f'<div class="kv"><span>image</span><b>{esc(docker.get("image") or "unknown")}</b></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<h3 style="margin-top:1.1rem">Configuration</h3>', unsafe_allow_html=True)
        snapshot = settings.snapshot()
        st.markdown(
            f'<div class="kv"><span>env file</span><b>{esc(snapshot["env_file"] or "not found")}</b></div>'
            f'<div class="kv"><span>gemini key</span><b>{esc(snapshot["gemini_key_hint"] or "missing")}</b></div>'
            f'<div class="kv"><span>pinned model</span><b>{esc(snapshot["gemini_model_pinned"] or "auto")}</b></div>'
            f'<div class="kv"><span>embeddings</span><b>{esc(settings.embedding_model)}</b></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<h3>Indexes</h3>', unsafe_allow_html=True)
        if stats["indexes"]:
            st.dataframe(
                pd.DataFrame(stats["indexes"]).rename(
                    columns={"name": "index", "type": "type", "status": "status"}),
                hide_index=True,
            )
        else:
            st.markdown('<p class="small">No table or no indexes yet.</p>',
                        unsafe_allow_html=True)

        st.markdown('<h3 style="margin-top:1.1rem">Quota today</h3>', unsafe_allow_html=True)
        status = agent.quota.status()
        rows = []
        for name in settings.models():
            entry = status["models"].get(name, {"calls": 0, "exhausted": False})
            rows.append({"model": name, "calls": entry["calls"],
                         "state": "exhausted" if entry["exhausted"] else "ok"})
        st.dataframe(
            pd.DataFrame(rows), hide_index=True,
            column_config={
                "model": st.column_config.TextColumn("model", width="medium"),
                "calls": st.column_config.NumberColumn(
                    "calls", width="small", max_value=settings.daily_limit),
                "state": st.column_config.TextColumn("state", width="small"),
            },
        )
        st.markdown(
            f'<p class="small">Resets at midnight Pacific. Day counted: '
            f'<span class="mono">{esc(status["date"])}</span></p>',
            unsafe_allow_html=True,
        )

        st.markdown('<h3 style="margin-top:1.1rem">Answer cache</h3>', unsafe_allow_html=True)
        cache_info = agent.cache.stats()
        st.markdown(
            f'<div class="kv"><span>entries</span><b>{cache_info["entries"]}</b></div>'
            f'<div class="kv"><span>size</span><b>{cache_info["size_kb"]} KB</b></div>'
            f'<div class="kv"><span>file</span><b>{esc(cache_info["path"].name)}</b></div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear answer cache", key="clear_cache"):
            removed = agent.cache.clear()
            st.session_state.cache_flash = f"Cleared {removed} cached answers."
            st.rerun()

        st.markdown('<h3 style="margin-top:1.1rem">Corpus backup</h3>', unsafe_allow_html=True)
        backup = st.session_state.get("backup")
        if st.button("Prepare corpus backup", key="prepare_backup"):
            try:
                payload = engine.export_corpus()
            except rc.RagError as exc:
                st.error(str(exc))
            else:
                data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
                sources = {item["source"] for item in payload["documents"]}
                backup = {
                    "data": data,
                    "name": f"corpus-{time.strftime('%Y%m%d-%H%M%S')}.json",
                    "chunks": len(payload["documents"]),
                    "sources": len(sources),
                }
                st.session_state.backup = backup
        if backup:
            st.download_button(
                "Download backup",
                data=backup["data"],
                file_name=backup["name"],
                mime="application/json",
                key="download_backup",
            )
            st.markdown(
                f'<p class="small">{backup["chunks"]} chunks from {backup["sources"]} '
                f'sources, {len(backup["data"]) / 1024:.1f} KB. Embeddings are '
                "recomputed on import.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="small">One JSON file with the text of every chunk. '
                "Prepare it again after adding documents.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.button("Re-check everything", key="recheck"):
            st.rerun()

    with st.expander("If something is wrong"):
        st.markdown(
            """
- **Database offline**: `docker start oracle-26ai-free`, wait for healthy, then re-check.
- **Table missing**: ingest a document in the Corpus tab, or run the notebook once.
- **HNSW index rejected (ORA-51962)**: the vector memory area is off. Run
  `START.bat /check` from the `rag-oneclick` folder; it sets the parameter and restarts.
- **Quota exhausted**: switch on Local mode in the sidebar; it never calls a model.
- **Port already in use**: close the other Streamlit, or run with `--server.port 8502`.
"""
        )


# --------------------------------------------------------------------------- #
# database down
# --------------------------------------------------------------------------- #
def render_db_down(db: dict, settings: rc.Settings) -> None:
    docker = rc.container_status()
    st.markdown(
        f'<div class="appbar"><div><div class="eyebrow">Oracle 26ai · local</div>'
        f"<h1>RAG Console</h1></div>"
        f'<div class="meta"><div><span class="dot bad"></span>database offline</div>'
        f'<div>{esc(settings.dsn)}</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<h2>The database is not reachable</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="small mono">{esc(db.get("error", "unknown error"))}</p>',
                unsafe_allow_html=True)

    if docker.get("state") == "running":
        advice = ("The container is running but the database is not answering yet. "
                  "It may still be opening the pluggable database, which takes about a minute "
                  "after a restart.")
    elif docker.get("state") == "missing":
        advice = ("The container does not exist. Create it with the one-click package, "
                  "or run the command below.")
    else:
        advice = f"The container is {docker.get('state')}. Start it with the command below."

    st.markdown(f'<p class="small">{esc(advice)}</p>', unsafe_allow_html=True)
    st.code("docker start oracle-26ai-free\ndocker ps --filter name=oracle-26ai-free", language="bash")
    st.markdown(
        '<p class="small">Or double-click <span class="mono">START.bat /check</span> in the '
        f'<span class="mono">rag-oneclick</span> folder, which repairs the container, the user '
        "and the vector memory setting.</p>",
        unsafe_allow_html=True,
    )
    if st.button("Re-check", type="primary"):
        st.rerun()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Oracle RAG Console",
        page_icon="◍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    settings, engine, agent = boot()
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("local_mode", not settings.has_llm)
    st.session_state.setdefault("bundle", None)

    db = engine.ping()
    if not db.get("ok"):
        render_db_down(db, settings)
        st.stop()

    stats = engine.stats()
    render_sidebar(settings, engine, agent, stats, db)

    st.markdown(
        f'<div class="appbar"><div><div class="eyebrow">Oracle 26ai · local</div>'
        f"<h1>RAG Console</h1></div>"
        f'<div class="meta">'
        f'<div><span class="dot ok"></span>connected · ping {db.get("ms", "?")} ms</div>'
        f'<div>{stats.get("sources", 0)} sources · {stats.get("rows", 0)} chunks</div>'
        f'<div>quota {agent.quota.status()["spent"]} / '
        f'{settings.daily_limit * max(1, len(settings.models()))} today</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    page = st.segmented_control(
        "Section",
        ["Ask", "Search lab", "Corpus", "Quality", "Health"],
        default="Ask",
        key="page",
        label_visibility="collapsed",
    ) or "Ask"
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    if page == "Ask":
        tab_ask(settings, agent, stats)
    elif page == "Search lab":
        tab_lab(engine)
    elif page == "Corpus":
        tab_corpus(engine, settings)
    elif page == "Quality":
        tab_quality(engine)
    else:
        tab_health(settings, engine, agent, stats, db)


main()
