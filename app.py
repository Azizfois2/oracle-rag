"""app.py - Oracle RAG Console.

Streamlit UI. Rule: this file contains no SQL and no prompt strings. Everything
that touches Oracle or Gemini lives in rag_core.py.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import time
import uuid
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st

import artifacts as arts
import rag_core as rc
from quality.answer_eval import load_cases as load_answer_cases
from quality.answer_eval import pending_cases, score_case
from quality.answer_store import (append_run, clear_progress, load_runs,
                                  progress_payload, read_progress,
                                  save_progress, saved_rows)
from quality.run_eval import (evaluate, load_last, load_questions, question_type,
                              questions_digest, save_run)

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
METHODS = ("keyword", "vector", "hybrid", "rerank")
LABEL = {"keyword": "Keyword", "vector": "Vector", "hybrid": "Hybrid",
         "rerank": "Reranked"}
NOTE = {
    "keyword": "Oracle Text · CONTAINS",
    "vector": "HNSW · cosine",
    "hybrid": "weighted fusion",
    "rerank": "local cross-encoder",
}
METHOD_CLASS = {"keyword": "kw", "vector": "vec", "hybrid": "hyb", "rerank": "rer"}
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
--   score = w * sim + (1 - w) * normalized_keyword_score

-- rerank: local cross-encoder over the top hybrid candidates
--   shown score is min-max normalised within that candidate pool"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  color-scheme:light;
  --ink:#201D1B; --ink2:#665E58; --ink3:#91867D;
  --line:#E6DFD9; --line-strong:#CFC3BA; --panel:#F1ECE7; --paper:#F8F5F2;
  --surface:#FFFEFC; --surface-2:#F4EFEB; --surface-3:#ECE4DE; --mark:#F4DCC4;
  --accent:#B23A2B; --accent-hover:#9C3123; --accent-2:#D5835E;
  --accent-soft:rgba(178,58,43,.09); --on-accent:#FFF9F6;
  --kw:#8A5A0B; --vec:#20609C; --hyb:#1E6E48; --rer:#6B4B8A;
  --warn:#9A6A00; --danger:#B84632; --ok:#247451;
  --ok-soft:rgba(36,116,81,.09); --ambient:rgba(213,131,94,.12);
  --chat-user:#F1EAE4; --focus:rgba(178,58,43,.18);
  --shadow:rgba(45,34,28,.08); --shadow-lg:rgba(45,34,28,.14);
}

html, body, .stApp, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"]{
  font-family:Inter,"Segoe UI",system-ui,sans-serif;
}
.stApp{
  background:radial-gradient(circle at 82% -8%,var(--ambient),transparent 32rem),var(--paper);
  color:var(--ink);
}
.block-container{ padding-top:2rem; padding-bottom:5rem; max-width:1280px; }
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"]{ display:none; }
[data-testid="stToolbar"]{ background:transparent; }
[data-testid="stExpandSidebarButton"]{ display:inline-flex; }
[data-testid="stExpandSidebarButton"] button{ color:var(--ink2); }
[data-testid="stExpandSidebarButton"] button:hover{ color:var(--accent); }

.appbar{ display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
  padding:.25rem 0 1.35rem; }
.appbar .brand{ display:flex; align-items:center; gap:.85rem; min-width:0; }
.brand-mark{ display:grid; place-items:center; width:42px; height:42px; flex:0 0 42px;
  border-radius:13px; background:linear-gradient(145deg,var(--accent-2),var(--accent));
  color:var(--on-accent); font-size:1.15rem; font-weight:700; letter-spacing:-.08em;
  box-shadow:0 8px 22px var(--shadow),inset 0 1px 0 rgba(255,255,255,.28); }
.appbar .eyebrow{ font-family:"IBM Plex Mono",monospace; font-size:.61rem; font-weight:500;
  letter-spacing:.17em; text-transform:uppercase; color:var(--accent); }
.appbar h1{ font-size:1.65rem; font-weight:650; margin:.08rem 0 0; letter-spacing:-.045em;
  line-height:1.15; }
.appbar .meta{ font-family:"IBM Plex Mono",monospace; font-size:.67rem;
  color:var(--ink2); text-align:right; line-height:1.65; font-variant-numeric:tabular-nums; }
.status-pill{ display:inline-flex; align-items:center; padding:.18rem .55rem;
  border:1px solid var(--line); border-radius:999px; background:var(--ok-soft);
  color:var(--ok); font-family:Inter,"Segoe UI",system-ui,sans-serif; font-size:.68rem;
  font-weight:600; letter-spacing:.01em; margin-bottom:.28rem; }

.dot{ display:inline-block; width:6px; height:6px; border-radius:50%;
  background:var(--ink3); margin-right:.4rem; vertical-align:1px; }
.dot.ok{ background:var(--ok); box-shadow:0 0 0 3px var(--ok-soft); }
.dot.bad{ background:var(--danger); } .dot.warn{ background:var(--warn); }

h2{ font-size:1.12rem; font-weight:650; letter-spacing:-.025em; margin:.1rem 0 .55rem; }
h3{ font-size:.9rem; font-weight:650; letter-spacing:-.015em; margin:.2rem 0 .4rem; }
p, li{ font-size:.89rem; line-height:1.65; }
.small{ font-size:.77rem; color:var(--ink2); line-height:1.6; }
mark{ background:var(--mark); color:var(--ink); padding:.02em .16em; border-radius:4px; }
.mono{ font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; }
.kw-t{ color:var(--kw); } .vec-t{ color:var(--vec); } .hyb-t{ color:var(--hyb); }
.rer-t{ color:var(--rer); }

.welcome{ position:relative; overflow:hidden; margin:.35rem 0 1.35rem; padding:2rem 2.1rem;
  border:1px solid var(--line); border-radius:20px;
  background:linear-gradient(135deg,var(--surface),var(--surface-2));
  box-shadow:0 18px 48px var(--shadow); }
.welcome::after{ content:""; position:absolute; width:250px; height:250px; right:-90px; top:-125px;
  border:42px solid var(--accent-soft); border-radius:50%; pointer-events:none; }
.welcome>*{ position:relative; z-index:1; }
.welcome .kicker{ font-family:"IBM Plex Mono",monospace; font-size:.62rem; font-weight:500;
  letter-spacing:.15em; text-transform:uppercase; color:var(--accent); margin-bottom:.65rem; }
.welcome h2{ max-width:38rem; font-size:clamp(1.65rem,3vw,2.25rem); font-weight:650;
  line-height:1.15; letter-spacing:-.045em; margin:0 0 .65rem; }
.welcome .lede{ max-width:43rem; color:var(--ink2); font-size:.91rem; margin:0; }
.hero-stats{ display:flex; gap:.45rem; flex-wrap:wrap; margin-top:1.15rem; }
.hero-stats span{ padding:.28rem .58rem; border:1px solid var(--line); border-radius:999px;
  background:var(--surface); color:var(--ink2); font-family:"IBM Plex Mono",monospace;
  font-size:.62rem; box-shadow:0 2px 8px var(--shadow); }
.section-label{ margin:0 0 .65rem; color:var(--ink3); font-family:"IBM Plex Mono",monospace;
  font-size:.62rem; font-weight:500; letter-spacing:.13em; text-transform:uppercase; }

button[data-baseweb="tab"]{ font-size:.82rem; font-weight:500; color:var(--ink2); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
div[data-baseweb="tab-highlight"]{ background:var(--accent); height:2px; }
div[data-baseweb="tab-border"]{ background:var(--line); }

.lab-head{ display:flex; justify-content:space-between; align-items:baseline;
  border-bottom:1px solid var(--line); padding-bottom:.5rem; }
.lab-head .name{ font-weight:650; font-size:.86rem; letter-spacing:-.015em; }
.lab-head .name i{ display:inline-block; width:7px; height:7px; border-radius:50%;
  margin-right:.45rem; vertical-align:1px; box-shadow:0 0 0 3px var(--surface-2); }
.lab-head .lat{ font-family:"IBM Plex Mono",monospace; font-size:.65rem; color:var(--ink3); }

.row{ padding:.85rem .95rem; margin-bottom:.6rem; border:1px solid var(--line);
  border-radius:12px; background:var(--surface); box-shadow:0 3px 12px var(--shadow);
  transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease; }
.row:hover{ border-color:var(--line-strong); box-shadow:0 8px 22px var(--shadow);
  transform:translateY(-1px); }
.row-top{ display:flex; align-items:baseline; gap:.5rem; }
.rank{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; font-weight:500; width:.9rem; }
.src{ font-family:"IBM Plex Mono",monospace; font-size:.7rem; color:var(--ink2);
  flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.date{ font-family:"IBM Plex Mono",monospace; font-size:.62rem; color:var(--ink3);
  white-space:nowrap; }
.score{ font-family:"IBM Plex Mono",monospace; font-size:.73rem; font-variant-numeric:tabular-nums; }
.track{ height:3px; background:var(--surface-3); margin:.45rem 0 .5rem; border-radius:99px;
  overflow:hidden; }
.track>i{ display:block; height:100%; border-radius:inherit; }
.snip{ font-size:.8rem; line-height:1.55; }
.tags{ margin-top:.45rem; display:flex; gap:.3rem; flex-wrap:wrap; }
.tag{ font-family:"IBM Plex Mono",monospace; font-size:.58rem; letter-spacing:.04em;
  text-transform:uppercase; color:var(--ink2); border:1px solid var(--line);
  background:var(--surface-2); padding:.12rem .38rem; border-radius:999px; }

i.kw{ background:var(--kw); } i.vec{ background:var(--vec); } i.hyb{ background:var(--hyb); }
i.rer{ background:var(--rer); }

.panel{ border:1px solid var(--line); background:var(--surface); padding:1.05rem 1.15rem;
  border-radius:14px; box-shadow:0 8px 26px var(--shadow); }
.panel .k{ font-family:"IBM Plex Mono",monospace; font-size:.6rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink3); }
.panel .v{ font-size:.88rem; font-weight:500; margin-top:.18rem; }
.strip{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:1.1rem; }

section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }
.side-brand{ display:flex; align-items:center; gap:.65rem; margin-bottom:.3rem; }
.side-brand .brand-mark{ width:32px; height:32px; flex-basis:32px; border-radius:10px;
  font-size:.85rem; box-shadow:0 5px 16px var(--shadow); }
.side-title{ font-size:.95rem; font-weight:650; letter-spacing:-.025em; }
.side-sub{ font-family:"IBM Plex Mono",monospace; font-size:.58rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink3); margin-top:.12rem; }
.kv{ display:flex; justify-content:space-between; align-items:center; gap:.6rem;
  font-size:.73rem; padding:.18rem 0; }
.kv span{ color:var(--ink3); } .kv b{ font-weight:500; font-family:"IBM Plex Mono",monospace; }
.bar{ height:4px; background:var(--surface-3); margin:.35rem 0 .2rem; border-radius:99px;
  overflow:hidden; }
.bar>i{ display:block; height:100%; border-radius:inherit;
  background:linear-gradient(90deg,var(--accent),var(--accent-2)); }

[data-testid="stChatMessage"]{ background:transparent; padding:.85rem 1rem; margin:.45rem 0;
  border:1px solid transparent; border-radius:15px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:var(--chat-user); border-color:var(--line); }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{
  border-radius:10px; }
[data-testid="stChatMessageAvatarUser"]{ background:var(--surface-3)!important;
  color:var(--ink2)!important; }
[data-testid="stChatMessageAvatarAssistant"]{ background:var(--accent-soft)!important;
  color:var(--accent)!important; }
[data-testid="stChatMessageContent"]{ font-size:.89rem; }

.stApp button[data-testid^="stBaseButton-"], .stButton>button,
.stDownloadButton>button{ border-radius:10px; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); font-size:.79rem; font-weight:550;
  padding:.4rem .8rem; min-height:2.35rem; box-shadow:0 2px 5px var(--shadow);
  transition:background .16s ease,border-color .16s ease,box-shadow .16s ease,transform .16s ease; }
.stApp button[data-testid^="stBaseButton-"]:hover, .stButton>button:hover,
.stDownloadButton>button:hover{ border-color:var(--line-strong); background:var(--surface-2);
  box-shadow:0 5px 14px var(--shadow); transform:translateY(-1px); }
.stApp button[data-testid^="stBaseButton-"]:active, .stButton>button:active{ transform:translateY(0); }
.stApp button[data-testid="stBaseButton-primary"], .stButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--accent),var(--accent-hover));
  border-color:var(--accent); color:var(--on-accent); box-shadow:0 6px 16px var(--shadow); }
.stApp button[data-testid="stBaseButton-primary"]:hover, .stButton>button[kind="primary"]:hover{
  background:linear-gradient(135deg,var(--accent-2),var(--accent));
  border-color:var(--accent-hover); color:var(--on-accent); }
.stApp button[data-testid^="stBaseButton-"]:focus-visible,
.stApp [data-testid="stChatInputSubmitButton"]:focus-visible{
  outline:3px solid var(--focus); outline-offset:2px; }

.stApp [data-testid="stButtonGroup"]{ display:inline-flex; gap:.2rem; padding:.25rem;
  border:1px solid var(--line); border-radius:12px; background:var(--surface-2); }
.stApp [data-testid="stButtonGroup"] button{ border:1px solid transparent; border-radius:8px;
  background:transparent; color:var(--ink2); box-shadow:none; font-size:.76rem;
  min-height:2rem; padding:.3rem .75rem; }
.stApp [data-testid="stButtonGroup"] button[data-selected],
.stApp [data-testid="stButtonGroup"] button[aria-selected="true"],
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"]{
  background:var(--surface); color:var(--ink); border-color:var(--line);
  box-shadow:0 3px 10px var(--shadow); }

.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"]>div{
  border-radius:10px; box-shadow:0 2px 6px var(--shadow); }
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="base-input"]:focus-within,
.stApp [data-baseweb="textarea"]:focus-within, .stApp [data-baseweb="select"]>div:focus-within{
  border-color:var(--accent)!important; box-shadow:0 0 0 3px var(--focus)!important; }
.stApp [data-testid="stCheckbox"] label:has(input[role="switch"]) > div:first-of-type{
  background:var(--line-strong); border:1px solid var(--line-strong); }
.stApp [data-testid="stCheckbox"] label:has(input[role="switch"]:checked) > div:first-of-type{
  background:var(--accent); border-color:var(--accent); }
.stApp [data-testid="stCheckbox"] label:has(input[role="switch"]) > div:first-of-type > div{
  background:var(--surface); box-shadow:0 1px 4px var(--shadow-lg); }

.stApp [data-testid="stBottom"]{ background:var(--paper); border-top:1px solid var(--line); }
.stApp [data-testid="stChatInput"]>div{ border-radius:14px; box-shadow:0 8px 24px var(--shadow); }
.stApp [data-testid="stChatInputSubmitButton"]{ border-radius:9px; }
.stApp [data-testid="stExpander"] details{ border:1px solid var(--line); border-radius:12px;
  background:var(--surface); overflow:hidden; box-shadow:0 3px 12px var(--shadow); }
.stApp [data-testid="stExpander"] summary{ padding:.15rem .2rem; }
.stApp [data-testid="stAlert"]{ border-radius:12px; }
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"],
.stApp [data-testid="stCodeBlock"]{ border:1px solid var(--line); border-radius:12px;
  overflow:hidden; box-shadow:0 3px 12px var(--shadow); }
hr{ border-color:var(--line); }

/* Inline citation: [1] opens the passage on hover, focus or click. */
a.cite{ position:relative; display:inline; color:var(--accent);
  font-family:"IBM Plex Mono",monospace; font-size:.72rem; font-weight:500;
  padding:0 .08rem; text-decoration:none; }
a.cite:hover, a.cite:focus{ text-decoration:underline; outline:none; }
a.cite .cite-body{ display:none; position:absolute; left:0; top:1.45em;
  width:min(30rem,78vw); text-align:left; background:var(--surface); color:var(--ink);
  border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:10px;
  padding:.7rem .8rem; box-shadow:0 12px 32px var(--shadow-lg); z-index:60; }
a.cite:hover .cite-body, a.cite:focus .cite-body,
a.cite:target .cite-body{ display:block; }
a.cite .cite-meta{ display:block; font-size:.62rem; color:var(--ink3);
  margin-bottom:.25rem; }
a.cite .cite-text{ display:block; font-family:Inter,"Segoe UI",system-ui,sans-serif;
  font-size:.78rem; line-height:1.55; white-space:normal; }

@media (max-width:700px){
  .block-container{ padding:1.25rem 1rem 4rem; }
  .appbar{ align-items:flex-start; flex-direction:column; gap:.85rem; }
  .appbar .meta{ text-align:left; }
  .welcome{ padding:1.45rem 1.25rem; border-radius:16px; }
  .row{ padding:.75rem; }
  .date{ display:none; }
}
@media (prefers-reduced-motion:reduce){
  .stApp *, .stApp *::before, .stApp *::after{ transition:none!important; }
}
"""

DARK_CSS = """
:root{
  color-scheme:dark;
  /* Warm graphite, not black. Every neutral keeps a trace of the terracotta
     accent, so the dark ground belongs to the same product as the light one. */
  --ink:#EDE5DC; --ink2:#B5A79B; --ink3:#9A8D81;
  --line:#3B322B; --line-strong:#5C4E44; --panel:#1C1714; --paper:#15120F;
  --surface:#2A2420; --surface-2:#342D28; --surface-3:#41372F; --mark:#5A3D24;
  --accent:#D9765E; --accent-hover:#E88C72; --accent-2:#E8A184;
  --accent-soft:rgba(217,118,94,.12); --on-accent:#1B0F0B;
  --kw:#D9A75A; --vec:#7FB6E8; --hyb:#6FCFA1; --rer:#C2A2E4;
  --warn:#D8A64A; --danger:#F0937F; --ok:#6FCFA1;
  --ok-soft:rgba(111,207,161,.1); --ambient:rgba(217,118,94,.12);
  --chat-user:#332A24; --focus:rgba(217,118,94,.34);
  --shadow:rgba(0,0,0,.5); --shadow-lg:rgba(0,0,0,.7);
  --hairline:rgba(255,255,255,.06);
}

/* ---- shell -------------------------------------------------------------- */
.stApp{
  background:
    radial-gradient(circle at 78% -10%,var(--ambient),transparent 38rem),
    radial-gradient(circle at 4% 108%,rgba(232,161,132,.06),transparent 32rem),
    linear-gradient(180deg,#1B1613 0%,var(--paper) 42rem,#120F0D 100%)!important;
  color:var(--ink)!important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stMainBlockContainer"]{
  background:transparent!important; color:var(--ink)!important;
}
[data-testid="stHeader"], [data-testid="stToolbar"]{ background:transparent!important; }
[data-testid="stToolbar"]{ color:var(--ink2)!important; }

section[data-testid="stSidebar"], [data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"]{
  background:linear-gradient(180deg,#1A1513,var(--panel))!important;
  color:var(--ink)!important;
}
section[data-testid="stSidebar"]{ border-right:1px solid var(--line)!important; }

[data-testid="stBottom"], [data-testid="stBottomBlockContainer"]{
  background:var(--paper)!important;
}
[data-testid="stBottom"]{ border-top:1px solid var(--line)!important; }

/* ---- text --------------------------------------------------------------- */
/* Streamlit paints most copy from its own light palette, so the base colour
   has to be forced here rather than inherited from .stApp alone. */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stChatMessageContent"]{ color:var(--ink)!important; }
.stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp h5, .stApp h6, .stApp label, .stApp legend, .stApp summary,
.stApp blockquote, .stApp td, .stApp th, .stApp dt, .stApp dd,
.stApp figcaption, .stApp caption{ color:var(--ink)!important; }
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *{ color:var(--ink)!important; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *{ color:var(--ink3)!important; }
.stApp a{ color:var(--accent)!important; }
.stApp a:hover{ color:var(--accent-hover)!important; }
.stApp ::selection{ background:var(--accent); color:var(--on-accent); }

/* the muted roles the custom HTML relies on */
.stApp .small, .stApp .appbar .meta, .stApp .src, .stApp .tag,
.stApp .welcome .lede{ color:var(--ink2)!important; }
.stApp .appbar .eyebrow, .stApp .side-sub, .stApp .kv span,
.stApp .lab-head .lat, .stApp .date, .stApp .panel .k,
.stApp .section-label, .stApp .cite-meta{ color:var(--ink3)!important; }
.stApp .kw-t{ color:var(--kw)!important; }
.stApp .vec-t{ color:var(--vec)!important; }
.stApp .hyb-t{ color:var(--hyb)!important; }
.stApp .rer-t{ color:var(--rer)!important; }
.stApp mark{ background:var(--mark)!important; color:var(--ink)!important; }

/* ---- buttons ------------------------------------------------------------ */
/* Match buttons by their test id: st.button(..., help=...) wraps the button in
   a tooltip span, so the old .stButton > button child selector misses it. */
.stApp button[data-testid^="stBaseButton-"]{
  background:var(--surface)!important; color:var(--ink)!important;
  border:1px solid var(--line)!important; box-shadow:none!important;
}
.stApp button[data-testid^="stBaseButton-"]:hover{
  background:var(--surface-2)!important; border-color:var(--line-strong)!important;
  color:var(--ink)!important;
}
.stApp button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--accent),var(--accent-hover))!important;
  border-color:var(--accent)!important; color:var(--on-accent)!important;
  box-shadow:0 6px 18px var(--shadow)!important;
}
.stApp button[data-testid="stBaseButton-primary"]:hover{
  background:linear-gradient(135deg,var(--accent-2),var(--accent))!important;
  border-color:var(--accent-hover)!important; color:var(--on-accent)!important;
}
.stApp button[data-testid="stBaseButton-header"],
.stApp button[data-testid="stBaseButton-headerNoPadding"]{
  background:transparent!important; border-color:transparent!important;
  color:var(--ink2)!important;
}
.stApp button[data-testid^="stBaseButton-"]:disabled{
  background:var(--surface)!important; color:var(--ink3)!important;
  border-color:var(--line)!important; opacity:.55;
}
.stApp button[data-testid^="stBaseButton-"]:focus-visible,
.stApp [data-testid="stChatInputSubmitButton"]:focus-visible{
  outline:2px solid var(--focus)!important; outline-offset:1px;
}

/* section switcher (st.segmented_control) */
.stApp [data-testid="stButtonGroup"]{
  background:var(--surface-2)!important; border:1px solid var(--line)!important;
}
.stApp [data-testid="stButtonGroup"] button{
  background:transparent!important; color:var(--ink2)!important;
  border:1px solid transparent!important; box-shadow:none!important;
}
.stApp [data-testid="stButtonGroup"] button:hover{
  background:var(--surface-3)!important; color:var(--ink)!important;
}
.stApp [data-testid="stButtonGroup"] button[data-selected],
.stApp [data-testid="stButtonGroup"] button[aria-selected="true"],
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"]{
  background:var(--surface)!important; color:var(--ink)!important;
  border-color:var(--line)!important; box-shadow:0 3px 10px var(--shadow)!important;
}

/* ---- fields and menus --------------------------------------------------- */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"]>div{
  background:var(--surface)!important; color:var(--ink)!important;
  border-color:var(--line)!important;
}
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="base-input"] input,
.stApp [data-baseweb="textarea"] textarea, .stApp [data-baseweb="select"] input{
  background:transparent!important; color:var(--ink)!important;
  caret-color:var(--accent);
}
.stApp input::placeholder, .stApp textarea::placeholder{
  color:var(--ink3)!important; opacity:1;
}
[data-baseweb="menu"], [data-baseweb="popover"]>div,
[data-baseweb="popover"] ul{
  background:var(--surface)!important; color:var(--ink)!important;
  border-color:var(--line)!important;
}
[data-baseweb="menu"] li{ color:var(--ink)!important; }
[data-baseweb="menu"] li:hover, [data-baseweb="menu"] li[aria-selected="true"]{
  background:var(--surface-2)!important;
}
[data-baseweb="tooltip"]{ background:var(--surface-2)!important; color:var(--ink)!important; }

/* ---- materials ---------------------------------------------------------- */
/* A drop shadow alone disappears on a near-black ground. Every raised plane
   gets a hairline of light along its top edge, so the ambient glow has
   something to catch and the surface reads as a material instead of a hole. */
.stApp .welcome{
  box-shadow:0 24px 60px var(--shadow-lg),inset 0 1px 0 var(--hairline)!important;
}
.stApp .panel{
  box-shadow:0 10px 30px var(--shadow),inset 0 1px 0 var(--hairline)!important;
}
.stApp .row{
  box-shadow:0 4px 14px var(--shadow),inset 0 1px 0 var(--hairline)!important;
}
.stApp .row:hover{
  box-shadow:0 10px 26px var(--shadow-lg),inset 0 1px 0 var(--hairline)!important;
}
.stApp .hero-stats span, .stApp .tag{
  box-shadow:inset 0 1px 0 var(--hairline)!important;
}
.stApp .brand-mark{
  box-shadow:0 10px 26px var(--shadow-lg),inset 0 1px 0 rgba(255,255,255,.18)!important;
}
.stApp button[data-testid^="stBaseButton-"]{
  box-shadow:inset 0 1px 0 var(--hairline)!important;
}
.stApp button[data-testid^="stBaseButton-"]:hover{
  box-shadow:0 6px 16px var(--shadow),inset 0 1px 0 var(--hairline)!important;
}
.stApp button[data-testid="stBaseButton-header"],
.stApp button[data-testid="stBaseButton-headerNoPadding"]{ box-shadow:none!important; }
.stApp button[data-testid="stBaseButton-primary"]{
  box-shadow:0 10px 24px var(--shadow-lg),inset 0 1px 0 rgba(255,255,255,.16)!important;
}
.stApp button[data-testid="stBaseButton-primary"]:hover{
  box-shadow:0 12px 28px var(--shadow-lg),inset 0 1px 0 rgba(255,255,255,.18)!important;
}
.stApp [data-testid="stButtonGroup"]{
  box-shadow:inset 0 1px 0 var(--hairline)!important;
}
.stApp [data-testid="stButtonGroup"] button[data-selected],
.stApp [data-testid="stButtonGroup"] button[aria-selected="true"],
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"]{
  box-shadow:0 4px 12px var(--shadow),inset 0 1px 0 var(--hairline)!important;
}
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"]>div{
  box-shadow:0 2px 8px var(--shadow),inset 0 1px 0 var(--hairline)!important;
}
.stApp [data-testid="stChatInput"]>div{
  box-shadow:0 10px 28px var(--shadow-lg),inset 0 1px 0 var(--hairline)!important;
}
.stApp [data-testid="stExpander"] details,
.stApp [data-testid="stCodeBlock"],
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"]{
  box-shadow:0 4px 14px var(--shadow),inset 0 1px 0 var(--hairline)!important;
}
.stApp a.cite .cite-body{
  box-shadow:0 16px 40px var(--shadow-lg),inset 0 1px 0 var(--hairline)!important;
}

/* ---- chat --------------------------------------------------------------- */
.stApp [data-testid="stChatMessage"]{ background:transparent!important; }
.stApp [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:var(--chat-user)!important;
}
.stApp [data-testid="stChatInput"]{ color:var(--ink)!important; }
.stApp [data-testid="stChatInput"]>div{
  background:var(--surface)!important; border-color:var(--line)!important;
}
.stApp [data-testid="stChatInputTextArea"]{
  background:transparent!important; color:var(--ink)!important;
}
.stApp [data-testid="stChatInputTextArea"]::placeholder{ color:var(--ink3)!important; }
.stApp [data-testid="stChatInputSubmitButton"]{
  background:var(--accent)!important; color:var(--on-accent)!important;
}
.stApp [data-testid="stChatInputSubmitButton"]:hover{
  background:var(--accent-hover)!important;
}
.stApp [data-testid="stChatInputSubmitButton"]:disabled{
  background:var(--surface-2)!important; color:var(--ink3)!important;
}

/* ---- containers --------------------------------------------------------- */
.stApp [data-testid="stExpander"] details{
  background:var(--surface)!important; border:1px solid var(--line)!important;
  color:var(--ink)!important;
}
.stApp [data-testid="stExpander"] summary{
  background:transparent!important; color:var(--ink)!important;
}
.stApp [data-testid="stExpander"] summary:hover{ background:var(--surface-2)!important; }
.stApp [data-testid="stForm"]{ background:transparent!important; border-color:var(--line)!important; }
.stApp hr, .stApp [data-testid="stDivider"]{
  border-color:var(--line)!important; background:var(--line)!important;
}
.stApp [data-testid="stToast"]{
  background:var(--surface)!important; color:var(--ink)!important;
  border:1px solid var(--line)!important;
}
.stApp [data-testid="stFileUploader"] section,
.stApp [data-testid="stFileUploaderDropzone"]{
  background:var(--surface)!important; border-color:var(--line)!important;
  color:var(--ink)!important;
}
.stApp [data-testid="stFileUploaderDropzone"]:hover{ background:var(--surface-2)!important; }

/* ---- status, alerts and code ------------------------------------------- */
.stApp [data-testid="stAlertContainer"], .stApp [data-testid="stAlert"]{
  background:var(--surface-2)!important; border:1px solid var(--line)!important;
  color:var(--ink)!important;
}
.stApp [data-testid="stAlertContentError"],
.stApp [data-testid="stAlertContentError"] *{ color:var(--danger)!important; }
.stApp [data-testid="stAlertContentWarning"],
.stApp [data-testid="stAlertContentWarning"] *{ color:var(--warn)!important; }
.stApp [data-testid="stAlertContentSuccess"],
.stApp [data-testid="stAlertContentSuccess"] *{ color:var(--ok)!important; }
.stApp [data-testid="stAlertContentInfo"],
.stApp [data-testid="stAlertContentInfo"] *{ color:var(--vec)!important; }
.stApp pre, .stApp code, .stApp [data-testid="stCodeBlock"] pre{
  background:var(--surface-2)!important; color:var(--ink)!important;
  border-color:var(--line)!important;
}
.stApp [data-testid="stCodeBlock"] code{ background:transparent!important; }
.stApp .token.comment, .stApp .token.prolog, .stApp .token.doctype,
.stApp .token.cdata{ color:var(--ink3)!important; }
.stApp .token.punctuation{ color:var(--ink2)!important; }
.stApp .token.property, .stApp .token.tag, .stApp .token.constant,
.stApp .token.symbol, .stApp .token.deleted{ color:var(--danger)!important; }
.stApp .token.boolean, .stApp .token.number{ color:var(--kw)!important; }
.stApp .token.selector, .stApp .token.attr-name, .stApp .token.string,
.stApp .token.char, .stApp .token.builtin, .stApp .token.inserted{ color:var(--hyb)!important; }
.stApp .token.operator, .stApp .token.entity, .stApp .token.url{ color:var(--ink2)!important; }
.stApp .token.atrule, .stApp .token.attr-value, .stApp .token.keyword{ color:var(--vec)!important; }
.stApp .token.function, .stApp .token.class-name{ color:var(--rer)!important; }

/* ---- tables, metrics and progress -------------------------------------- */
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"]{
  background:var(--surface)!important; color:var(--ink)!important;
  border-color:var(--line)!important;
}
/* The data grid paints its own canvas, so CSS colours never reach the cells.
   A gentle inversion maps Streamlit's light grid onto the graphite palette. */
.stApp [data-testid="stDataFrame"] canvas{
  filter:invert(.92) hue-rotate(180deg) saturate(.9);
}
.stApp [data-testid="stDataFrame"] [role="columnheader"]{ background:var(--surface-2)!important; }
.stApp [data-testid="stTable"] th{ background:var(--surface-2)!important; }
.stApp [data-testid="stTable"] td, .stApp [data-testid="stTable"] th{
  border-color:var(--line)!important;
}
.stApp [data-testid="stMetric"] *{ color:var(--ink)!important; }
.stApp [data-testid="stMetricLabel"]{ color:var(--ink2)!important; }
.stApp [data-testid="stProgress"]>div{ background:var(--surface-2)!important; }
.stApp [data-testid="stProgress"]>div>div{ background:var(--accent)!important; }

/* ---- tabs, chrome and scrollbars --------------------------------------- */
.stApp button[data-baseweb="tab"]{
  background:transparent!important; color:var(--ink2)!important;
}
.stApp button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink)!important; }
.stApp [data-baseweb="tab-highlight"]{ background:var(--accent)!important; }
.stApp [data-baseweb="tab-border"]{ background:var(--line)!important; }
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button{ color:var(--ink2)!important; }
[data-testid="stExpandSidebarButton"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover{ color:var(--accent)!important; }
.stApp *::-webkit-scrollbar{ width:10px; height:10px; }
.stApp *::-webkit-scrollbar-track{ background:var(--paper); }
.stApp *::-webkit-scrollbar-thumb{
  background:var(--line-strong); border-radius:8px; border:2px solid var(--paper);
}
.stApp *::-webkit-scrollbar-thumb:hover{ background:var(--ink3); }
"""


# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def boot() -> tuple[rc.Settings, rc.RagEngine, rc.Agent, rc.ChatHistory]:
    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    agent = rc.Agent(engine, settings)
    history = rc.ChatHistory(engine)
    engine.warm_async()
    return settings, engine, agent, history


# --------------------------------------------------------------------------- #
# markup helpers
# --------------------------------------------------------------------------- #
def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def row_html(rank: int, hit: rc.Hit, method: str, tags: tuple[str, ...] = ()) -> str:
    width = max(2, min(100, round(hit.score * 100)))
    css = METHOD_CLASS[method]
    tags_block = ""
    if tags:
        inner = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags)
        tags_block = f'<div class="tags">{inner}</div>'
    date_block = (f'<span class="date">{esc(hit.published)}</span>'
                  if hit.published else "")
    return (
        f'<div class="row">'
        f'<div class="row-top">'
        f'<span class="rank {css}-t">{rank}</span>'
        f'<span class="src">{esc(hit.source)}</span>'
        f'{date_block}'
        f'<span class="score {css}-t">{hit.score:.3f}</span>'
        f"</div>"
        f'<div class="track"><i class="{css}"></i></div>'
        f'<div class="snip">{esc(hit.snippet(190))}</div>'
        f"{tags_block}"
        f"</div>"
    )


def lab_head(method: str, ms: float, count: int) -> str:
    return (
        f'<div class="lab-head"><div class="name"><i class="{METHOD_CLASS[method]}"></i>{LABEL[method]}</div>'
        f'<div class="lat">{count} hits · {ms:.0f} ms</div></div>'
    )


def panel(label: str, value: str) -> str:
    return f'<div><div class="k">{esc(label)}</div><div class="v mono">{esc(value)}</div></div>'


def dot(state: str) -> str:
    return {"ok": "ok", "warn": "warn"}.get(state, "bad")


def short_label(text: object, limit: int = 30) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean or "Untitled discussion"
    return clean[: limit - 3].rstrip() + "..."


def when(value: object) -> str:
    try:
        stamp = dt.datetime.fromisoformat(str(value))
    except Exception:
        return ""
    now = dt.datetime.now().astimezone()
    if stamp.date() == now.date():
        return stamp.strftime("%H:%M")
    if stamp.year == now.year:
        return stamp.strftime("%b %d")
    return stamp.strftime("%b %d, %Y")


def highlight(text: object, query: str) -> str:
    """Escape a snippet, then wrap each searched word in a mark."""
    body = str(text or "")
    terms = sorted({word for word in str(query or "").split() if len(word) > 1},
                   key=len, reverse=True)
    if not terms:
        return esc(body)
    pattern = re.compile("(" + "|".join(re.escape(word) for word in terms) + ")",
                         re.IGNORECASE)
    parts = pattern.split(body)
    return "".join(
        f"<mark>{html.escape(part)}</mark>" if index % 2 else html.escape(part)
        for index, part in enumerate(parts)
    )


def more_history() -> None:
    limit = int(st.session_state.get("history_limit", 4))
    st.session_state.history_limit = min(50, limit + 4)


def fresh_chat() -> None:
    st.session_state.chat_id = uuid.uuid4().hex
    st.session_state.chat = []
    st.session_state.pending = None
    st.session_state.pop("confirm_delete", None)


def clear_history_search() -> None:
    st.session_state.history_search = ""
    st.session_state.history_limit = 4


def open_chat(history: rc.ChatHistory, chat_id: str) -> None:
    try:
        messages = history.messages(chat_id)
    except rc.RagError as exc:
        st.session_state.history_flash = f"Could not open that discussion: {exc}"
        return
    st.session_state.chat_id = chat_id
    st.session_state.chat = messages
    st.session_state.pop("confirm_delete", None)


def init_chat(history: rc.ChatHistory) -> None:
    """On a fresh session, reopen the most recent discussion."""
    if "chat_id" in st.session_state:
        return
    try:
        chats = history.list(limit=1)
        if chats:
            st.session_state.chat_id = chats[0]["id"]
            st.session_state.chat = history.messages(chats[0]["id"])
        else:
            fresh_chat()
    except rc.RagError as exc:
        fresh_chat()
        st.session_state.history_flash = f"History is unavailable: {exc}"


def save_message(history: rc.ChatHistory, role: str, content: str,
                 answer: rc.Answer | None = None) -> None:
    try:
        history.append(st.session_state.get("chat_id"), role, content, answer=answer)
    except rc.RagError as exc:
        st.session_state.history_flash = f"This discussion was not saved: {exc}"


def follow_context() -> list[dict]:
    """Earlier turns, excluding the question that was just added."""
    if not st.session_state.get("follow_context", True):
        return []
    chat = st.session_state.get("chat") or []
    return list(chat[:-1]) if chat else []


def question_before(chat: list[dict], index: int) -> str:
    """The user question that this answer belongs to."""
    for item in reversed(chat[:index]):
        if item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def save_answer_artefact(chat: list[dict], index: int, answer: rc.Answer) -> None:
    """Write the answer shown at `index` to the artefacts directory."""
    question = question_before(chat, index)
    try:
        saved = arts.save_answer(
            question, answer, chat_id=st.session_state.get("chat_id") or "")
    except (OSError, ValueError) as exc:
        st.session_state.artifact_flash = f"Could not save this answer: {exc}"
    else:
        st.session_state.artifact_flash = (
            f"Saved {len(saved)} artefacts to {saved[0].path.parent}")
    st.rerun()


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(settings: rc.Settings, engine: rc.RagEngine, agent: rc.Agent,
                   history: rc.ChatHistory, stats: dict, db: dict) -> None:
    quota = agent.quota
    models = settings.models()
    status = quota.status()
    spent = status["spent"]
    budget = settings.daily_limit * max(1, len(models))

    with st.sidebar:
        st.markdown(
            '<div class="side-brand"><div class="brand-mark">O</div><div>'
            '<div class="side-title">Oracle RAG Console</div>'
            '<div class="side-sub">26ai · private intelligence</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.toggle(
            "Dark mode",
            key="dark_mode",
            help="Switch the console to the dark palette. The choice lasts for this browser session.",
        )
        flash = st.session_state.pop("history_flash", None)
        if flash:
            st.markdown(f'<p class="small">{esc(flash)}</p>', unsafe_allow_html=True)
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
        st.toggle(
            "Follow-up context",
            key="follow_context",
            help="Send the last few turns with the question, so follow-ups like "
                 "'who funded them?' still make sense. Off means every question stands alone.",
        )
        method_labels = list(SEARCH_METHOD_LABELS)
        current_method = st.session_state.get("search_method", "hybrid")
        method_index = next(
            (index for index, method in enumerate(SEARCH_METHOD_LABELS.values())
             if method == current_method), 0)
        selected_method = st.selectbox(
            "Ask search method", method_labels, index=method_index,
            key="sidebar_search_method_label",
            help="Choose Hybrid, Vector, Keyword, or Reranked corpus retrieval.",
        )
        st.session_state.search_method = SEARCH_METHOD_LABELS[selected_method]
        st.caption("Ask commands: `/search`, `/mode`, `/context`, `/history`, "
                   "`/open`, `/find`, `/sources`, `/trace`, `/save`, `/title`, `/delete`.")
        if not settings.has_llm:
            st.markdown('<p class="small">No Gemini key found, so only local mode is available. '
                        'Add <span class="mono">GEMINI_API_KEY</span> to '
                        f'<span class="mono">{esc(settings.env_path or ".env")}</span>.</p>',
                        unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="k mono" style="font-size:.62rem;letter-spacing:.11em;'
                    'text-transform:uppercase;color:var(--ink3)">discussions</div>',
                    unsafe_allow_html=True)
        if st.button("New discussion", key="new_chat", use_container_width=True):
            fresh_chat()
            st.rerun()

        find = st.text_input("Find in discussions", key="history_search",
                             placeholder="words from a past chat")
        query = " ".join((find or "").split())
        if query:
            st.button("Clear search", key="clear_history_search",
                      on_click=clear_history_search, use_container_width=True)
        limit = int(st.session_state.get("history_limit", 4))
        found: list[dict] = []
        try:
            if query:
                found = history.search(query, limit=50)
                chats = found[:limit]
            else:
                chats = history.list(limit=6)
        except rc.RagError as exc:
            chats = []
            st.markdown(f'<p class="small">{esc(exc)}</p>', unsafe_allow_html=True)
        if query and found:
            noun = "discussion" if len(found) == 1 else "discussions"
            st.markdown(f'<p class="small">{len(found)} {noun} match.</p>',
                        unsafe_allow_html=True)
        if not chats:
            message = ("No discussion matches those words." if query
                       else "No saved discussions yet.")
            st.markdown(f'<p class="small">{message}</p>', unsafe_allow_html=True)
        for item in chats:
            active = item["id"] == st.session_state.get("chat_id")
            help_text = (f'{item.get("matches", 0)} matching messages'
                         if query else
                         f'{item["messages"]} messages · {when(item["updated"])}')
            if st.button(
                short_label(item["title"]),
                key=f'chat_{item["id"]}',
                help=help_text,
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                open_chat(history, item["id"])
                st.rerun()
            if query:
                st.markdown(
                    f'<p class="small">{highlight(item.get("snippet", ""), query)}</p>',
                    unsafe_allow_html=True,
                )
        if query and len(found) > limit:
            st.button("Show more", key="more_history", on_click=more_history,
                      use_container_width=True)

        if st.session_state.get("chat_id"):
            with st.expander("Manage this discussion"):
                chat_id = st.session_state.chat_id
                try:
                    current_title = history.title(chat_id)
                except rc.RagError:
                    current_title = "Untitled discussion"
                new_title = st.text_input("Discussion title", value=current_title,
                                          key=f"rename_title_{chat_id}")
                if st.button("Rename", key="rename_chat", use_container_width=True):
                    label = " ".join((new_title or "").split())
                    if not label:
                        st.warning("A discussion needs a title.")
                    elif label != current_title:
                        try:
                            history.rename(chat_id, label)
                        except rc.RagError as exc:
                            st.session_state.history_flash = str(exc)
                        else:
                            st.session_state.history_flash = "Discussion renamed."
                        st.rerun()

                try:
                    export = history.export_markdown(chat_id)
                except rc.RagError:
                    export = ""
                if export:
                    st.download_button(
                        "Download discussion",
                        data=export.encode("utf-8"),
                        file_name=f"discussion-{chat_id[:8]}-{dt.datetime.now():%Y%m%d-%H%M}.md",
                        mime="text/markdown",
                        key="download_discussion",
                        use_container_width=True,
                    )

                if st.button("Save to artefacts", key="save_discussion_artefact",
                             use_container_width=True,
                             help="Write this discussion to artifacts/ as Markdown and JSON."):
                    try:
                        saved = arts.save_discussion(history, chat_id)
                    except (OSError, ValueError, rc.RagError) as exc:
                        st.session_state.history_flash = f"Could not save the discussion: {exc}"
                    else:
                        st.session_state.artifact_flash = (
                            f"Saved {len(saved)} discussion artefacts to "
                            f"{saved[0].path.parent}")
                    st.rerun()

                if st.session_state.get("confirm_delete") == chat_id:
                    st.markdown('<p class="small">Delete this discussion and all of its '
                                'messages?</p>', unsafe_allow_html=True)
                    if st.button("Yes, delete", key="delete_yes", type="primary",
                                 use_container_width=True):
                        try:
                            removed = history.delete(chat_id)
                        except rc.RagError as exc:
                            st.session_state.history_flash = str(exc)
                        else:
                            st.session_state.history_flash = f"Deleted {removed} messages."
                            try:
                                remaining = history.list(limit=1)
                            except rc.RagError:
                                remaining = []
                            if remaining:
                                open_chat(history, remaining[0]["id"])
                            else:
                                fresh_chat()
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()
                    if st.button("Cancel", key="delete_no", use_container_width=True):
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()
                elif st.button("Delete this discussion", key="delete_chat",
                               use_container_width=True):
                    st.session_state.confirm_delete = chat_id
                    st.rerun()

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

SEARCH_METHOD_LABELS = {
    "Hybrid": "hybrid",
    "Vector": "vector",
    "Keyword": "keyword",
    "Reranked": "rerank",
}

GUI_SLASH_COMMANDS = (
    ("/help", "Show Ask commands"),
    ("/new", "Start a new discussion"),
    ("/local", "Switch to zero-quota retrieval"),
    ("/remote", "Switch to grounded model answers"),
    ("/search hybrid|vector|keyword|rerank", "Choose the retrieval method"),
    ("/mode local|remote", "Set the answer mode"),
    ("/context on|off", "Toggle follow-up context"),
    ("/history", "Show recent saved discussions in the sidebar"),
    ("/open N", "Open a recent saved discussion"),
    ("/find words", "Search saved discussions"),
    ("/sources", "Show sources for the latest answer"),
    ("/trace", "Show the latest answer trace"),
    ("/save", "Save the latest answer as an artefact"),
    ("/title [name]", "Show or rename this discussion"),
    ("/delete", "Open the discussion delete confirmation"),
    ("/clear", "Clear the discussion search"),
)


def _last_saved_answer() -> rc.Answer | None:
    for item in reversed(st.session_state.get("chat") or []):
        if item.get("role") == "assistant" and item.get("answer"):
            return item["answer"]
    return None


def handle_gui_slash(line: str, settings: rc.Settings,
                     history: rc.ChatHistory) -> str:
    """Execute an Ask slash command before sidebar widgets are created."""
    name, _, raw_arg = line.strip().partition(" ")
    name, arg = name.lower(), raw_arg.strip()
    if name == "/help":
        return "**Ask commands**\n\n" + "\n".join(
            f"- `{command}`: {description}"
            for command, description in GUI_SLASH_COMMANDS)
    if name in ("/new", "/clear-chat"):
        fresh_chat()
        return "Started a new discussion."
    if name == "/local":
        st.session_state.local_mode = True
        return "Local mode on. Retrieval only, no model call or quota."
    if name == "/remote":
        if not settings.has_llm:
            return "Remote mode is unavailable. Add GEMINI_API_KEY to the environment."
        st.session_state.local_mode = False
        return "Remote mode on. Answers must be supported by cited corpus documents."
    if name == "/mode":
        if arg.lower() in ("local", "remote"):
            return handle_gui_slash("/" + arg.lower(), settings, history)
        return "Usage: `/mode local` or `/mode remote`."
    if name == "/search":
        method = arg.lower().replace("-", "")
        if method == "reranked":
            method = "rerank"
        try:
            method = rc.normalise_search_method(method)
        except rc.RagError:
            return "Usage: `/search hybrid`, `/search vector`, `/search keyword`, or `/search rerank`."
        st.session_state.search_method = method
        label = next(label for label, key in SEARCH_METHOD_LABELS.items()
                     if key == method)
        st.session_state.sidebar_search_method_label = label
        return f"Ask retrieval set to **{label}**."
    if name == "/context":
        value = arg.lower()
        if value in ("on", "off"):
            st.session_state.follow_context = value == "on"
            return f"Follow-up context {'on' if value == 'on' else 'off'}."
        return "Usage: `/context on` or `/context off`."
    if name == "/history":
        st.session_state.history_search = ""
        st.session_state.history_limit = 6
        return "Showing recent saved discussions in the sidebar."
    if name == "/open":
        try:
            index = int(arg)
        except ValueError:
            return "Usage: `/open N`. Use `/history` to see recent discussions."
        try:
            chats = history.list(limit=10)
        except rc.RagError as exc:
            return f"Could not list discussions: {exc}"
        if not 1 <= index <= len(chats):
            return "No discussion at that number. Use `/history` to see the list."
        open_chat(history, chats[index - 1]["id"])
        st.session_state.history_search = ""
        return f"Opened **{esc(chats[index - 1]['title'])}**."
    if name == "/find":
        if not arg:
            return "Usage: `/find words from a saved discussion`."
        st.session_state.history_search = arg
        st.session_state.history_limit = 6
        return f"Searching saved discussions for **{esc(arg)}** in the sidebar."
    if name == "/clear":
        st.session_state.history_search = ""
        st.session_state.history_limit = 4
        return "Discussion search cleared."
    if name == "/sources":
        answer = _last_saved_answer()
        if answer is None or not answer.sources:
            return "The latest answer has no saved sources."
        return "**Latest answer sources**\n\n" + "\n".join(
            f"- {esc(str(item.get('source', 'unknown')))}"
            for item in answer.sources)
    if name == "/trace":
        answer = _last_saved_answer()
        if answer is None or not answer.steps:
            return "There is no answer trace yet."
        return "**Latest answer trace**\n\n" + "\n".join(
            f"- `{step.kind}` {esc(step.label)}"
            + (f": {esc(step.detail)}" if step.detail else "")
            for step in answer.steps)
    if name == "/save":
        chat = st.session_state.get("chat") or []
        for index in range(len(chat) - 1, -1, -1):
            item = chat[index]
            if item.get("role") == "assistant" and item.get("answer") is not None:
                save_answer_artefact(chat, index,
                                     rc.enforce_answer_grounding(item["answer"]))
                return "Saving the latest answer as an artefact."
        return "There is no answer to save yet."
    if name == "/title":
        chat_id = st.session_state.get("chat_id")
        if not chat_id:
            return "There is no active discussion to rename."
        if not arg:
            try:
                return f"Discussion title: **{esc(history.title(chat_id))}**"
            except rc.RagError as exc:
                return str(exc)
        try:
            history.rename(chat_id, arg)
        except rc.RagError as exc:
            return f"Could not rename discussion: {exc}"
        st.session_state.history_flash = "Discussion renamed."
        return f"Discussion renamed to **{esc(arg)}**."
    if name == "/delete":
        chat_id = st.session_state.get("chat_id")
        if not chat_id:
            return "There is no active discussion to delete."
        st.session_state.slash_delete_confirm = chat_id
        return "Review the delete confirmation below the Ask input."
    return f"Unknown command `{esc(name)}`. Type `/help` for the list."


_CITE_MARK = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
_SOURCE_MARK = re.compile(
    r"\[Source:\s*(?P<src>[^|\]\n]+?)\s*(?:\|[^\]]*)?\]")


def citation_passage(item: dict, engine: rc.RagEngine | None,
                     number: int | None = None) -> str:
    """The passage behind one citation, from the tool or from Oracle."""
    snippet = ""
    passages = item.get("passages")
    if isinstance(passages, dict) and number is not None:
        snippet = str(passages.get(str(number)) or passages.get(number) or "").strip()
    if not snippet:
        snippet = str(item.get("snippet") or "").strip()
    if not snippet and engine is not None:
        name = str(item.get("source") or "")
        try:
            passages = engine.source_passages(name, limit=1)
        except rc.RagError:
            passages = []
        if passages:
            snippet = passages[0].text
    return " ".join(snippet.split())[:600]


def citation_html(number: int, item: dict, passage: str, anchor: str) -> str:
    name = str(item.get("source") or "unknown")
    published = str(item.get("published") or "")
    meta = f"{name} · {published}" if published else name
    body = esc(passage) if passage else (
        "No passage stored for this source. Open the Corpus tab to read it.")
    return (
        f'<a class="cite" id="{anchor}" href="#{anchor}" '
        f'aria-label="Source {number}: {esc(meta)}" title="{esc(meta)}">'
        f'[{number}]'
        f'<span class="cite-body"><span class="cite-meta">{esc(meta)}</span>'
        f'<span class="cite-text">{body}</span></span>'
        f"</a>"
    )


def render_answer_text(answer: rc.Answer,
                       engine: rc.RagEngine | None = None) -> None:
    """Render answer text, turning [1] markers into clickable passages."""
    text = answer.text or ""
    if not text.strip():
        return
    lookup = rc.citation_numbers(answer.sources or [])
    if not lookup:
        st.markdown(text)
        return

    by_name: dict[str, dict] = {}
    for item in answer.sources:
        if isinstance(item, dict):
            by_name.setdefault(str(item.get("source") or "").strip().lower(), item)
    by_item: dict[int, int] = {}
    for number in sorted(lookup):
        by_item.setdefault(id(lookup[number]), number)

    def from_source(match: re.Match[str]) -> str:
        item = by_name.get(match.group("src").strip().lower())
        number = by_item.get(id(item)) if item is not None else None
        return f"[{number}]" if number is not None else match.group(0)

    text = _SOURCE_MARK.sub(from_source, text)
    seen = {"count": 0}
    token = uuid.uuid4().hex[:8]

    def from_number(match: re.Match[str]) -> str:
        chips: list[str] = []
        for part in re.split(r"\s*,\s*", match.group(1)):
            number = int(part)
            item = lookup.get(number)
            if item is None:
                chips.append(f"[{number}]")
                continue
            seen["count"] += 1
            anchor = f"cite-{token}-{number}-{seen['count']}"
            chips.append(citation_html(
                number, item, citation_passage(item, engine, number), anchor))
        return "".join(chips)

    rendered = _CITE_MARK.sub(from_number, text)
    st.markdown(rendered, unsafe_allow_html=seen["count"] > 0)


def render_answer_body(answer: rc.Answer, show_text: bool = True,
                       engine: rc.RagEngine | None = None) -> None:
    if answer.local:
        st.markdown('<p class="small">Retrieval only. No model was called.</p>',
                    unsafe_allow_html=True)
        if not answer.hits:
            st.markdown('<p class="small">Nothing matched. Try different words.</p>',
                        unsafe_allow_html=True)
        method = answer.hits[0].method if answer.hits else "hybrid"
        for index, hit in enumerate(answer.hits, start=1):
            st.markdown(row_html(index, hit, method), unsafe_allow_html=True)
    else:
        if show_text:
            render_answer_text(answer, engine=engine)
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
        st.markdown(
            '<p class="small">Click a numbered citation in the answer to read '
            "the passage, or open a source below.</p>",
            unsafe_allow_html=True,
        )
        for item in answer.sources:
            if not isinstance(item, dict):
                item = {"source": str(item)}
            name = str(item.get("source") or "unknown")
            published = str(item.get("published") or "")
            relevance = float(item.get("relevance", 0.0) or 0.0)
            numbers = item.get("citations")
            if not isinstance(numbers, (list, tuple)) or not numbers:
                numbers = [item["citation"]] if item.get("citation") else []
            prefix = f"[{', '.join(str(n) for n in numbers)}] " if numbers else ""
            label = f"{prefix}{name} · {published}" if published else f"{prefix}{name}"
            with st.expander(label):
                meta = f"Relevance {relevance:.3f}"
                meta += f" · published {published}" if published else " · undated"
                st.markdown(f'<p class="small">{esc(meta)}</p>',
                            unsafe_allow_html=True)
                snippet = str(item.get("snippet") or "").strip()
                passages: list[rc.Hit] = []
                if not snippet and engine is not None:
                    try:
                        passages = engine.source_passages(name, limit=3)
                    except rc.RagError:
                        passages = []
                if snippet:
                    st.markdown(snippet)
                elif passages:
                    for index, hit in enumerate(passages):
                        if index:
                            st.divider()
                        st.markdown(hit.text)
                else:
                    st.markdown(
                        '<p class="small">No passage stored for this source. '
                        "Open the Corpus tab to read the document.</p>",
                        unsafe_allow_html=True,
                    )


def answer_flow(question: str, settings: rc.Settings, agent: rc.Agent,
                history: rc.ChatHistory) -> None:
    st.session_state.chat.append({"role": "user", "content": question})
    save_message(history, "user", question)
    context = follow_context()
    with st.chat_message("user"):
        st.markdown(question)
    _run_answer(question, settings, agent, history, context)


def _run_answer(question: str, settings: rc.Settings, agent: rc.Agent,
                history: rc.ChatHistory, context: list[dict]) -> None:
    local = st.session_state.local_mode or not settings.has_llm
    search_method = st.session_state.get("search_method", "hybrid")
    with st.chat_message("assistant"):
        status = st.status(
            "Reading the corpus" if local else "Searching the corpus",
            expanded=True,
        )
        stream_box = st.empty()
        chunks: list[str] = []
        writing = {"on": False}

        def on_step(kind: str, text: str) -> None:
            status.write(f"`{GLYPH.get(kind, kind)}` {text}")

        def on_reset() -> None:
            chunks.clear()
            writing["on"] = False
            stream_box.empty()
            status.update(label="Searching the corpus", state="running",
                          expanded=True)

        def on_token(text: str) -> None:
            if not text:
                return
            if not writing["on"]:
                writing["on"] = True
                status.update(label="Writing the answer", state="running",
                              expanded=False)
            chunks.append(text)
            stream_box.markdown("".join(chunks) + "▌")

        try:
            if local:
                if search_method == "hybrid":
                    answer = agent.ask_local(question, context=context)
                else:
                    answer = agent.ask_local(question, context=context,
                                             search_method=search_method)
            else:
                ask_options = dict(on_step=on_step, on_token=on_token,
                                   on_reset=on_reset, context=context)
                if search_method != "hybrid":
                    ask_options["search_method"] = search_method
                answer = agent.ask(question, **ask_options)
            answer = rc.enforce_answer_grounding(answer)
            status.update(label=f"Answered in {answer.elapsed:.1f}s",
                          state="complete", expanded=False)
        except rc.QuotaExhausted as exc:
            stream_box.empty()
            status.update(label="Quota exhausted", state="error", expanded=False)
            st.warning(f"{exc}. Switch on Local mode in the sidebar to keep working.")
            st.session_state.chat.append({"role": "error", "content": str(exc)})
            save_message(history, "error", str(exc))
            st.rerun()
        except rc.RagError as exc:
            stream_box.empty()
            status.update(label="Failed", state="error", expanded=False)
            st.error(str(exc))
            st.session_state.chat.append({"role": "error", "content": str(exc)})
            save_message(history, "error", str(exc))
            st.rerun()

        if not local:
            with stream_box:
                render_answer_text(answer, engine=agent.engine)
        render_answer_body(answer, show_text=local, engine=agent.engine)
    st.session_state.chat.append({"role": "assistant", "answer": answer})
    save_message(history, "assistant", answer.text, answer=answer)


def retry_last(settings: rc.Settings, agent: rc.Agent,
               history: rc.ChatHistory) -> None:
    """Ask the failed question again, with the same earlier context."""
    chat = st.session_state.get("chat") or []
    if not chat or chat[-1].get("role") != "error":
        return
    question = ""
    for item in reversed(chat[:-1]):
        if item.get("role") == "user":
            question = str(item.get("content") or "")
            break
    if not question:
        return
    try:
        history.delete_last_error(st.session_state.get("chat_id"))
    except rc.RagError as exc:
        st.session_state.history_flash = (
            f"The failed message is still in Oracle: {exc}")
    st.session_state.chat.pop()
    context = list(st.session_state.chat[:-1])
    _run_answer(question, settings, agent, history, context)


def tab_ask(settings: rc.Settings, agent: rc.Agent, history: rc.ChatHistory,
            stats: dict) -> None:
    flash = st.session_state.pop("history_flash", None)
    if flash:
        st.warning(flash)
    saved = st.session_state.pop("artifact_flash", None)
    if saved:
        st.success(saved)
    command_result = st.session_state.pop("command_result", None)
    if command_result:
        st.markdown(command_result)

    chat = st.session_state.chat
    if not chat and not st.session_state.get("pending"):
        st.markdown(
            '<div class="welcome"><div class="kicker">Grounded answers, clear sources</div>'
            '<h2>Your corpus, ready to answer.</h2>'
            '<p class="lede">Ask across your private knowledge with Oracle 26ai. '
            'Every answer stays connected to the passages that support it.</p>'
            f'<div class="hero-stats"><span>{stats.get("sources", 0)} documents</span>'
            f'<span>{stats.get("rows", 0)} indexed chunks</span>'
            '<span>local &amp; private</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Start with a question</div>',
                    unsafe_allow_html=True)
        columns = st.columns(2)
        for index, suggestion in enumerate(SUGGESTIONS):
            if columns[index % 2].button(suggestion, key=f"sug{index}"):
                st.session_state.pending = suggestion
                st.rerun()
    elif chat:
        st.markdown('<p class="small">This discussion is saved in Oracle. '
                    'Pick another one in the sidebar, or start a new one.</p>',
                    unsafe_allow_html=True)

    for index, item in enumerate(chat):
        if item["role"] == "user":
            with st.chat_message("user"):
                st.markdown(item["content"])
        elif item["role"] == "error":
            with st.chat_message("assistant"):
                st.error(item["content"])
                if index == len(chat) - 1:
                    if st.button("Retry", key=f"retry_{index}",
                                 help="Ask the same question again, with the same context."):
                        st.session_state.retry_question = True
                        st.rerun()
        else:
            with st.chat_message("assistant"):
                answer = rc.enforce_answer_grounding(item["answer"])
                render_answer_body(answer, engine=agent.engine)
                if st.button(
                    "Save artefact",
                    key=f"save_artefact_{index}",
                    help="Write this answer to artifacts/ as Markdown, JSON and HTML.",
                ):
                    save_answer_artefact(chat, index, answer)

    if st.session_state.pop("retry_question", False):
        retry_last(settings, agent, history)

    queued = st.session_state.pop("pending", None)
    if queued:
        answer_flow(queued, settings, agent, history)

    prompt = st.chat_input("Ask about the corpus")
    if prompt:
        if prompt.lstrip().startswith("/"):
            st.session_state.pending_slash = prompt.strip()
            st.rerun()
        else:
            answer_flow(prompt, settings, agent, history)

    confirm_delete = (st.session_state.get("slash_delete_confirm")
                      or st.session_state.get("confirm_delete"))
    if confirm_delete == st.session_state.get("chat_id"):
        st.warning("Delete this discussion and all its messages?")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm delete", key="ask_confirm_delete",
                              type="primary"):
            try:
                removed = history.delete(confirm_delete)
            except rc.RagError as exc:
                st.error(f"Could not delete discussion: {exc}")
            else:
                st.session_state.history_flash = f"Deleted {removed} messages."
                try:
                    remaining = history.list(limit=1)
                except rc.RagError:
                    remaining = []
                if remaining:
                    open_chat(history, remaining[0]["id"])
                else:
                    fresh_chat()
                st.session_state.pop("confirm_delete", None)
                st.session_state.pop("slash_delete_confirm", None)
                st.rerun()
        if cancel_col.button("Cancel", key="ask_cancel_delete"):
            st.session_state.pop("confirm_delete", None)
            st.session_state.pop("slash_delete_confirm", None)
            st.rerun()


# --------------------------------------------------------------------------- #
# search lab
# --------------------------------------------------------------------------- #
def tab_lab(engine: rc.RagEngine) -> None:
    try:
        names = [row["source"] for row in engine.sources()]
        lo, hi = engine.published_range()
    except rc.RagError:
        names = []
        lo, hi = None, None
    lo_date = dt.date.fromisoformat(lo) if lo else None
    hi_date = dt.date.fromisoformat(hi) if hi else None
    options = ["All sources"] + names
    if st.session_state.get("lab_source") not in options:
        st.session_state["lab_source"] = "All sources"
    with st.form("lab", border=False):
        top_left, top_right = st.columns([2.2, 1])
        query = top_left.text_input(
            "Query",
            value=st.session_state.get("lab_query", "companies building hybrid search"),
            placeholder="e.g. which startup is challenging Pinecone?",
        )
        source = top_right.selectbox(
            "Source",
            options,
            key="lab_source",
            help="Filter every retrieval path to one document.",
        )
        date_left, date_right = st.columns(2)
        date_from = date_left.date_input(
            "Published from",
            value=None,
            min_value=lo_date,
            max_value=hi_date,
            key="lab_date_from",
            help="Inclusive. Documents without a published date are left out "
                 "when a range is set.",
        )
        date_to = date_right.date_input(
            "Published to",
            value=None,
            min_value=lo_date,
            max_value=hi_date,
            key="lab_date_to",
            help="Inclusive. Leave both dates empty to search the whole corpus.",
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
        if date_from and date_to and date_from > date_to:
            st.warning("The start date is after the end date. Swap them and try again.")
            return
        st.session_state.lab_query = query.strip()
        only = None if source == "All sources" else source
        since = date_from.isoformat() if date_from else None
        until = date_to.isoformat() if date_to else None
        with st.spinner("Querying keyword, vector, hybrid and the reranker"):
            try:
                st.session_state.bundle = engine.search_all(
                    query.strip(), k=int(k), weight=float(weight), rerank=True,
                    source=only, date_from=since, date_to=until)
                st.session_state.bundle_at = time.strftime("%H:%M:%S")
            except rc.RagError as exc:
                st.error(str(exc))
                return

    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.markdown(
            '<p class="small">One query, four retrieval paths, side by side. '
            "Filter by source or published date; both apply to every path. "
            "The reranker is a local model, so it costs no API quota.</p>",
            unsafe_allow_html=True,
        )
        return

    memberships = bundle.all_keys()
    shared = bundle.consensus()
    filter_note = (f' · source <span class="mono">{esc(bundle.source)}</span>'
                   if bundle.source else "")
    if bundle.date_from or bundle.date_to:
        span = f'{bundle.date_from or "start"} to {bundle.date_to or "latest"}'
        filter_note += f' · published <span class="mono">{esc(span)}</span>'
    st.markdown(
        f'<p class="small"><span class="mono">{esc(bundle.query)}</span> · '
        f'{bundle.total} documents across the four paths · '
        f'{len(shared)} found by at least two · '
        f'fusion took <span class="mono">{bundle.timings["hybrid"]:.0f} ms</span> · '
        f'rerank took <span class="mono">{bundle.timings["rerank"]:.0f} ms</span>'
        f'{filter_note}</p>',
        unsafe_allow_html=True,
    )

    columns = st.columns(4, gap="large")
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
            "both, so a document found by only one path can still rank. Reranked takes the "
            "top hybrid candidates and scores each one with a local cross-encoder; its "
            "displayed score is normalised within that candidate pool.</p>",
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
                columns={"source": "source", "chunks": "chunks",
                         "avg_chars": "avg chars", "published": "published"}),
            hide_index=True,
            column_config={
                "source": st.column_config.TextColumn("source", width="large"),
                "chunks": st.column_config.NumberColumn("chunks", width="small"),
                "avg chars": st.column_config.NumberColumn("avg chars", width="small"),
                "published": st.column_config.TextColumn("published", width="small"),
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
                    stamp = result.get("published") or "undated"
                    added.append(f'{result["source"]} ({result["chunks"]} chunks, {stamp})')
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
def tab_quality(engine: rc.RagEngine, agent: rc.Agent) -> None:
    questions = load_questions()
    counts = {kind: sum(1 for item in questions if question_type(item) == kind)
              for kind in ("single", "cross", "unanswerable")}
    st.markdown(
        f'<p class="small">{len(questions)} labelled questions: '
        f'{counts["single"]} single-document, {counts["cross"]} cross-document, '
        f'{counts["unanswerable"]} unanswerable. A hit means the expected string '
        "appears in the retrieved chunk or its source. Cross-document hits need "
        "every expected string in the top k. Unanswerable questions are not "
        "scored as hits; their vector similarity is compared with answerable "
        "questions. Reranked is a local cross-encoder. No LLM or API is called, "
        "so a full run costs no quota.</p>",
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns([1, 2, 1])
    k = left.number_input("k", 3, 10, 5, key="eval_k")
    weight = middle.slider("Hybrid weight", 0.0, 1.0, 0.6, 0.05, key="eval_w")
    right.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    if right.button("Run evaluation", type="primary"):
        with st.spinner("Rebuilding the keyword index, loading the reranker if needed, then retrieving all questions"):
            try:
                result = evaluate(engine, questions, k=int(k), weight=float(weight))
                save_run(result)
                st.session_state.eval = result
            except rc.RagError as exc:
                st.error(str(exc))
                return

    result = st.session_state.get("eval") or load_last()
    current = engine.corpus_fingerprint()
    if result and (result.get("version") != 3
                   or result.get("questions_hash") != questions_digest(questions)
                   or result.get("corpus_fingerprint") != current):
        st.info("The question set or corpus changed since this run. "
                "Press Run evaluation to refresh it.")
        return
    if not result:
        st.markdown('<p class="small">No run yet. Press Run evaluation.</p>',
                    unsafe_allow_html=True)
        return

    progress = {
        "path": st.column_config.TextColumn("path", width="small"),
        "hit@k": st.column_config.ProgressColumn(
            "hit@k", min_value=0.0, max_value=1.0, format="%.2f"),
        "MRR": st.column_config.ProgressColumn(
            "MRR", min_value=0.0, max_value=1.0, format="%.2f"),
    }
    summary = pd.DataFrame([
        {"path": LABEL[m], "hit@k": result["summary"][m]["hit"],
         "MRR": result["summary"][m]["mrr"],
         "found": f"{result['summary'][m]['found']}/{result['summary'][m]['n']}"}
        for m in METHODS
    ])
    st.dataframe(summary, hide_index=True, column_config=progress)
    rerank_load = float(result.get("rerank_load") or 0.0)
    index_rebuild = float(result.get("index_rebuild") or 0.0)
    st.markdown(
        f'<p class="small">k = {result["k"]} · hybrid weight {result["weight"]} · '
        f'{result["elapsed"]}s · {result.get("corpus_rows", "?")} chunks · '
        f'keyword index {index_rebuild:.1f}s · reranker load {rerank_load:.1f}s · '
        f'run at {result["at"].replace("T", " ")} · '
        f'{result["question_count"]} questions</p>',
        unsafe_allow_html=True,
    )

    breakdown = []
    for kind, label in (("single", "Single document"), ("cross", "Cross document")):
        block = (result.get("by_type") or {}).get(kind)
        if not block:
            continue
        for method in METHODS:
            breakdown.append({
                "set": label,
                "path": LABEL[method],
                "hit@k": block[method]["hit"],
                "MRR": block[method]["mrr"],
                "found": f"{block[method]['found']}/{block[method]['n']}",
            })
    if breakdown:
        st.markdown('<h3 style="margin-top:1.1rem">By question type</h3>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(breakdown), hide_index=True, column_config=progress)

    unanswerable = result.get("unanswerable") or {}
    if unanswerable:
        st.markdown('<h3 style="margin-top:1.1rem">Unanswerable questions</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="small">These are not scored as hits: retrieval always '
            "returns k chunks. Vector cosine is the only absolute similarity "
            "here. Keyword scores are normalised per query, so their best match "
            "is always 1.00, and hybrid scores are fused from them; neither is "
            "a confidence measure. The table compares the vector top score "
            "with answerable questions.</p>",
            unsafe_allow_html=True,
        )
        base = (result.get("answerable_tops") or {}).get("vector") or {}
        scores = unanswerable["vector"]
        st.dataframe(
            pd.DataFrame([
                {"set": "Answerable", "min top": base.get("min_top", 0.0),
                 "mean top": base.get("mean_top", 0.0),
                 "max top": base.get("max_top", 0.0)},
                {"set": "Unanswerable", "min top": scores["min_top"],
                 "mean top": scores["mean_top"], "max top": scores["max_top"]},
            ]),
            hide_index=True,
            column_config={
                "set": st.column_config.TextColumn("set", width="small"),
                "min top": st.column_config.NumberColumn(format="%.2f", width="small"),
                "mean top": st.column_config.NumberColumn(format="%.2f", width="small"),
                "max top": st.column_config.NumberColumn(format="%.2f", width="small"),
            },
        )

    rows = []
    for row in result["rows"]:
        kind = row.get("type", "single")
        if kind == "cross":
            expected = " + ".join(row.get("expect_all") or [])
        elif kind == "unanswerable":
            expected = "not in corpus"
        else:
            expected = row.get("expect")

        def cell(method: str) -> str:
            if kind == "unanswerable":
                return "n/a"
            value = row[method]
            return "miss" if value is None else f"#{value}"

        rows.append({
            "question": row["question"],
            "set": kind,
            "keyword": cell("keyword"),
            "vector": cell("vector"),
            "hybrid": cell("hybrid"),
            "rerank": cell("rerank"),
            "top cosine": f'{float(row.get("vector_top", 0.0)):.2f}',
            "expected": expected,
        })
    with st.expander("Per question"):
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    answer_cases = load_answer_cases()
    answer_counts = {kind: sum(1 for item in answer_cases
                               if item.get("type") == kind)
                     for kind in ("single", "cross", "unanswerable")}
    st.markdown('<h3 style="margin-top:1.4rem">Answer quality</h3>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p class="small">{len(answer_cases)} labelled answers: '
        f'{answer_counts["single"]} single-document, '
        f'{answer_counts["cross"]} cross-document, '
        f'{answer_counts["unanswerable"]} unanswerable. A case passes when every '
        "expected source is cited, every expected fact appears in the answer, "
        "unanswerable questions are refused, and every citation names a real "
        "document. The scoring itself is local; a live run calls Gemini once "
        "per question and may use more than one request.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("Cases"):
        st.dataframe(
            pd.DataFrame([
                {
                    "question": item.get("question"),
                    "set": item.get("type"),
                    "expected source": ", ".join(item.get("expect_sources") or [])
                                       or "none",
                    "expected facts": ", ".join(item.get("expect_facts") or [])
                                      or "refuse",
                }
                for item in answer_cases
            ]),
            hide_index=True,
        )

    current_fp = engine.corpus_fingerprint()
    corpus_rows = int(engine.stats().get("rows", 0) or 0)
    saved = read_progress()
    saved_fp = str(saved.get("corpus_fingerprint") or "")
    saved_cases = saved_rows(saved)
    if saved_cases and saved_fp and saved_fp != current_fp:
        st.warning(
            f"The saved answer run used a different corpus (saved {saved_fp}, "
            f"current {current_fp}). Start a fresh run so the scores are not mixed.")
        if st.button("Start a fresh answer run", key="clear_answer_progress"):
            clear_progress()
            st.session_state.pop("answer_check", None)
            st.session_state.pop("answer_check_at", None)
            st.rerun()
        saved_cases = []
    known = {str(case.get("question") or "") for case in answer_cases}
    rows = [row for row in saved_cases
            if str(row.get("question") or "") in known]
    remaining = pending_cases(answer_cases, rows)
    answered = len(rows)

    flash = st.session_state.pop("answer_flash", None)
    if flash:
        kind, text = flash
        if kind == "success":
            st.success(text)
        else:
            st.warning(text)

    if answered:
        when = str(saved.get("at") or "").replace("T", " ")
        st.markdown(
            f'<p class="small"><b>{answered} of {len(answer_cases)} cases '
            f'answered</b> · saved {esc(when)} · corpus {esc(current_fp)}</p>',
            unsafe_allow_html=True,
        )

    if remaining:
        left, right = st.columns([1, 2])
        answer_limit = left.number_input(
            "Questions this run", 1, len(remaining),
            min(5, len(remaining)), key="answer_limit",
            help="Only the cases that are still unanswered. The free tier allows "
                 "about 20 requests per day per model, so start small.",
        )
        asked = int(answer_limit)
        right.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        right.markdown(
            f'<p class="small">This will ask {asked} of the {len(remaining)} '
            f'{"case" if len(remaining) == 1 else "cases"} still unanswered, about '
            f'{asked * 2} to {asked * 4} API calls. The free tier is about '
            "20 requests per day per model.</p>",
            unsafe_allow_html=True,
        )
        run_answers = right.button("Check answers", type="primary",
                                   key="run_answer_check")
    else:
        st.success(f"All {len(answer_cases)} cases are answered on this corpus.")
        run_answers = False
        answer_limit = 0
        if st.button("Start a fresh answer run", key="fresh_answer_run"):
            clear_progress()
            st.session_state.pop("answer_check", None)
            st.session_state.pop("answer_check_at", None)
            st.rerun()

    if run_answers:
        selected = remaining[: int(answer_limit)]
        corpus = [row["source"] for row in engine.sources()]
        run_rows = list(rows)
        model = str(saved.get("model") or "")
        progress = st.progress(0.0, text="Starting the answer check")
        stopped = ""
        for index, case in enumerate(selected, start=1):
            progress.progress((index - 1) / len(selected),
                              text=f"Question {index} of {len(selected)}")
            try:
                answer = agent.ask(str(case.get("question") or ""))
            except rc.QuotaExhausted as exc:
                stopped = (f"Quota stopped the run after {len(run_rows)} of "
                           f"{len(answer_cases)} cases: {exc}")
                break
            except rc.RagError as exc:
                stopped = (f"The run stopped after {len(run_rows)} of "
                           f"{len(answer_cases)} cases: {exc}")
                break
            result = score_case(case, answer, corpus)
            run_rows.append(result.as_dict())
            model = answer.model
            save_progress(progress_payload(
                current_fp, corpus_rows, corpus, model, run_rows))
        progress.empty()

        if not run_rows:
            st.warning("No answers completed, so there is nothing to score.")
        else:
            payload = progress_payload(
                current_fp, corpus_rows, corpus, model, run_rows)
            save_progress(payload)
            st.session_state.answer_check = payload["summary"]
            st.session_state.answer_check_at = time.strftime("%H:%M:%S")
            if stopped:
                st.session_state.answer_flash = ("warning", stopped)
            elif len(run_rows) == len(answer_cases):
                if append_run(payload, expected_total=len(answer_cases)):
                    st.session_state.answer_flash = (
                        "success",
                        f"All {len(answer_cases)} cases answered. "
                        "This run was added to the trend history.")
                else:
                    st.session_state.answer_flash = (
                        "success", f"All {len(answer_cases)} cases answered.")
            st.rerun()

    answer_summary = (st.session_state.get("answer_check")
                      or (saved.get("summary") if rows else None))
    if answer_summary:
        at = (st.session_state.get("answer_check_at")
              or (str(saved.get("at") or "") if rows else ""))
        passed = int(answer_summary.get("passed", 0))
        total = int(answer_summary.get("total", 0))
        rate = float(answer_summary.get("score", 0.0)) * 100
        st.markdown(
            f'<p class="small"><b>{passed}/{total}</b> cases passed '
            f'({rate:.0f}%) · run at {esc(at)}</p>',
            unsafe_allow_html=True,
        )
        checks = [
            {
                "check": name,
                "passed": bucket["passed"],
                "total": bucket["total"],
                "rate": bucket["passed"] / bucket["total"] if bucket["total"] else 0.0,
            }
            for name, bucket in (answer_summary.get("checks") or {}).items()
        ]
        st.dataframe(
            pd.DataFrame(checks), hide_index=True,
            column_config={
                "check": st.column_config.TextColumn("check", width="small"),
                "passed": st.column_config.NumberColumn("passed", width="small"),
                "total": st.column_config.NumberColumn("total", width="small"),
                "rate": st.column_config.ProgressColumn(
                    "rate", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        case_rows = []
        for item in answer_summary.get("cases") or []:
            failed = ", ".join(check["name"] for check in item.get("checks") or []
                               if not check.get("passed")) or "none"
            case_rows.append({
                "question": item.get("question"),
                "set": item.get("type"),
                "result": "pass" if item.get("passed") else "fail",
                "failed checks": failed,
                "sources": ", ".join(item.get("sources") or []) or "none",
            })
        with st.expander("Per case", expanded=passed < total):
            st.dataframe(pd.DataFrame(case_rows), hide_index=True)

    runs = [run for run in load_runs()
            if str(run.get("corpus_fingerprint") or "") == current_fp]
    if runs:
        st.markdown('<h3 style="margin-top:1.4rem">Pass-rate trend</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="small">One point per completed run on this corpus, oldest '
            "first. A run joins the history only when every case is answered. "
            "Hover a point to read its rate.</p>",
            unsafe_allow_html=True,
        )
        labels = [str(run.get("at") or "").replace("T", " ")[:19]
                  for run in runs]

        def rate_frame(key: str) -> pd.DataFrame:
            names = sorted({str(name) for run in runs
                            for name in (run.get(key) or {})})
            body = []
            for run in runs:
                point = {}
                for name in names:
                    bucket = (run.get(key) or {}).get(name) or {}
                    total = int(bucket.get("total") or 0)
                    point[name] = (int(bucket.get("passed") or 0) / total
                                   if total else float("nan"))
                body.append(point)
            return pd.DataFrame(body, index=labels)

        by_check = rate_frame("checks")
        by_type = rate_frame("by_type")
        if len(by_check.columns):
            st.markdown('<p class="small"><b>By check</b></p>',
                        unsafe_allow_html=True)
            st.line_chart(by_check)
        if len(by_type.columns):
            st.markdown('<p class="small"><b>By case type</b></p>',
                        unsafe_allow_html=True)
            st.line_chart(by_type)
        history_rows = [
            {
                "run": label,
                "passed": int(run.get("passed") or 0),
                "total": int(run.get("total") or 0),
                "rate": float(run.get("score") or 0.0),
                "model": run.get("model") or "unknown",
            }
            for label, run in zip(labels, runs)
        ]
        with st.expander("Run history"):
            st.dataframe(
                pd.DataFrame(history_rows), hide_index=True,
                column_config={
                    "rate": st.column_config.ProgressColumn(
                        "rate", min_value=0.0, max_value=1.0, format="%.2f"),
                },
            )


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def tab_artifacts() -> None:
    flash = st.session_state.pop("artifact_flash", None)
    if flash:
        st.success(flash)
    counts = arts.stats()
    st.markdown(
        '<p class="small">Answers and discussions saved as files: Markdown to read, '
        "JSON to process again, HTML to open in a browser. Use <b>Save artefact</b> "
        "under any answer, or <b>Save to artefacts</b> in the sidebar. The folder "
        "is local and no model call is involved.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel"><div class="strip">'
        + panel("files", str(counts["count"]))
        + panel("size", arts.human_size(int(counts["bytes"])))
        + panel("latest", when(counts["latest"]) or "none")
        + panel("folder", Path(str(counts["directory"])).name)
        + "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="small mono" style="margin-top:.35rem">{esc(counts["directory"])}</p>',
        unsafe_allow_html=True,
    )

    limit = int(st.session_state.get("artifact_limit", 12))
    items = arts.list_artefacts(limit=limit)
    if not items:
        st.markdown(
            '<p class="small" style="margin-top:1rem">No artefacts yet. Ask a '
            "question, press <b>Save artefact</b> below the answer, then come back "
            "here to preview, download or delete the files.</p>",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<h3 style="margin-top:1rem">Saved files</h3>', unsafe_allow_html=True)
    for item in items:
        label = (f'{item.title} · {item.kind} · {arts.human_size(item.size)} · '
                 f'{when(item.created) or "unknown date"}')
        with st.expander(label):
            st.markdown(
                f'<p class="small mono">{esc(item.filename)}</p>',
                unsafe_allow_html=True,
            )
            body = item.preview()
            if item.kind == "markdown":
                st.markdown(body)
            elif item.kind == "json":
                st.code(body, language="json")
            elif item.kind == "html":
                st.code(body, language="html")
            else:
                st.code(body)
            download, remove = st.columns(2)
            mime = {"markdown": "text/markdown", "json": "application/json",
                    "html": "text/html", "text": "text/plain"}[item.kind]
            with download:
                st.download_button(
                    "Download",
                    data=item.path.read_bytes(),
                    file_name=item.filename,
                    mime=mime,
                    key=f"download_artifact_{item.filename}",
                    use_container_width=True,
                )
            with remove:
                confirm = st.session_state.get("artifact_delete")
                if confirm == item.filename:
                    if st.button("Yes, delete", key=f"artifact_delete_yes_{item.filename}",
                                 type="primary", use_container_width=True):
                        try:
                            arts.delete_artefact(item)
                        except (OSError, ValueError) as exc:
                            st.session_state.artifact_flash = f"Could not delete: {exc}"
                        else:
                            st.session_state.artifact_flash = (
                                f"Deleted {item.filename}.")
                        st.session_state.pop("artifact_delete", None)
                        st.rerun()
                    if st.button("Cancel", key=f"artifact_delete_no_{item.filename}",
                                 use_container_width=True):
                        st.session_state.pop("artifact_delete", None)
                        st.rerun()
                elif st.button("Delete", key=f"artifact_delete_{item.filename}",
                               use_container_width=True):
                    st.session_state.artifact_delete = item.filename
                    st.rerun()

    total = int(counts["count"])
    if total > len(items):
        if st.button(f"Show more ({total - len(items)} left)", key="more_artifacts",
                     use_container_width=True):
            st.session_state.artifact_limit = min(60, limit + 12)
            st.rerun()


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
def tab_health(settings: rc.Settings, engine: rc.RagEngine, agent: rc.Agent,
               history: rc.ChatHistory, stats: dict, db: dict) -> None:
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

        st.markdown('<h3 style="margin-top:1.1rem">Discussion history</h3>',
                    unsafe_allow_html=True)
        try:
            history_info = history.stats()
        except rc.RagError as exc:
            st.markdown(f'<p class="small">{esc(exc)}</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="kv"><span>discussions</span><b>{history_info["chats"]}</b></div>'
                f'<div class="kv"><span>messages</span><b>{history_info["messages"]}</b></div>'
                f'<div class="kv"><span>table</span><b>{esc(history_info["table"])}</b></div>',
                unsafe_allow_html=True,
            )
            if st.session_state.get("confirm_clear_history"):
                st.warning("This deletes every saved discussion. The corpus is not touched.")
                yes, no = st.columns(2)
                if yes.button("Delete all", key="clear_history_yes", type="primary",
                              use_container_width=True):
                    try:
                        removed = history.clear()
                    except rc.RagError as exc:
                        st.session_state.history_flash = str(exc)
                    else:
                        st.session_state.history_flash = f"Deleted {removed} messages."
                        fresh_chat()
                    st.session_state.pop("confirm_clear_history", None)
                    st.rerun()
                if no.button("Cancel", key="clear_history_no", use_container_width=True):
                    st.session_state.pop("confirm_clear_history", None)
                    st.rerun()
            elif st.button("Clear all discussions", key="clear_history"):
                st.session_state.confirm_clear_history = True
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
        f'<div class="appbar"><div class="brand"><div class="brand-mark">O</div><div>'
        f'<div class="eyebrow">Oracle 26ai · private intelligence</div>'
        f"<h1>RAG Console</h1></div></div>"
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
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("artifact_limit", 12)
    theme = DARK_CSS if st.session_state.get("dark_mode") else ""
    st.markdown(f"<style>{CSS}{theme}</style>", unsafe_allow_html=True)

    settings, engine, agent, history = boot()
    st.session_state.setdefault("local_mode", not settings.has_llm)
    st.session_state.setdefault("follow_context", True)
    st.session_state.setdefault("search_method", "hybrid")
    st.session_state.setdefault("history_limit", 4)
    st.session_state.setdefault("bundle", None)

    db = engine.ping()
    if not db.get("ok"):
        render_db_down(db, settings)
        st.stop()

    init_chat(history)
    pending_slash = st.session_state.pop("pending_slash", None)
    if pending_slash:
        st.session_state.command_result = handle_gui_slash(
            pending_slash, settings, history)
    stats = engine.stats()
    render_sidebar(settings, engine, agent, history, stats, db)

    st.markdown(
        f'<div class="appbar"><div class="brand"><div class="brand-mark">O</div><div>'
        f'<div class="eyebrow">Oracle 26ai · private intelligence</div>'
        f"<h1>RAG Console</h1></div></div>"
        f'<div class="meta">'
        f'<div class="status-pill"><span class="dot ok"></span>Connected</div>'
        f'<div>{stats.get("sources", 0)} sources · {stats.get("rows", 0)} chunks · '
        f'ping {db.get("ms", "?")} ms</div>'
        f'<div>quota {agent.quota.status()["spent"]} / '
        f'{settings.daily_limit * max(1, len(settings.models()))} today</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    page = st.segmented_control(
        "Section",
        ["Ask", "Search lab", "Corpus", "Quality", "Artefacts", "Health"],
        default="Ask",
        key="page",
        label_visibility="collapsed",
    ) or "Ask"
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    if page == "Ask":
        tab_ask(settings, agent, history, stats)
    elif page == "Search lab":
        tab_lab(engine)
    elif page == "Corpus":
        tab_corpus(engine, settings)
    elif page == "Quality":
        tab_quality(engine, agent)
    elif page == "Artefacts":
        tab_artifacts()
    else:
        tab_health(settings, engine, agent, history, stats, db)


main()
