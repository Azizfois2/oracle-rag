"""Headless render test for app.py.

    python tests/test_app.py

Uses streamlit.testing.AppTest, which executes the real script against the real
database. It spends zero Gemini quota: the chat is never submitted, only the
retrieval form.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def markdown_text(at: AppTest) -> str:
    return "\n".join(str(getattr(element, "value", "")) for element in at.markdown)


def alert_text(at: AppTest) -> str:
    return "\n".join(str(getattr(element, "value", "")) for element in at.info)


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()

    print("\n== first render ==")
    check("no exception", not at.exception, str(at.exception))
    body = markdown_text(at)
    check("app bar rendered", "RAG Console" in body)
    check("sidebar rendered", "Oracle RAG Console" in body)
    check("quota block rendered", "quota today" in body)
    check("ask empty state rendered", "indexed in Oracle 26ai" in body or "documents" in body)
    check("no database-down screen", "database is not reachable" not in body)
    check("sidebar can be reopened after collapsing",
          "stExpandSidebarButton" in body
          and '[data-testid="stToolbar"]{ display:none; }' not in body,
          "the expand button is styled and its toolbar stays visible")

    print("\n== search lab ==")
    at.segmented_control[0].set_value("Search lab").run()
    check("lab renders without exception", not at.exception, str(at.exception))
    check("lab has a query field", len(at.text_input) >= 1)
    submit = [b for b in at.button if "Compare" in str(b.label)]
    check("lab has a submit button", bool(submit), str([str(b.label) for b in at.button]))
    if submit:
        submit[0].click().run()
        check("compare runs without exception", not at.exception, str(at.exception))
        bundle = at.session_state.get("bundle")
        check("bundle stored in session", bundle is not None)
        if bundle is not None:
            check("four paths returned",
                  bool(bundle.keyword) and bool(bundle.vector)
                  and bool(bundle.hybrid) and bool(bundle.rerank),
                  f"kw {len(bundle.keyword)} vec {len(bundle.vector)} "
                  f"hyb {len(bundle.hybrid)} rer {len(bundle.rerank)}")
            check("reranked hits come from the reranker",
                  all(h.method == "rerank" for h in bundle.rerank))
            check("timings present",
                  set(bundle.timings) == {"keyword", "vector", "hybrid", "rerank"},
                  str({k: round(v, 1) for k, v in bundle.timings.items()}))
        selectors = [s for s in at.selectbox if s.key == "lab_source"]
        check("lab offers a source filter", bool(selectors))
        if selectors:
            options = list(selectors[0].options)
            check("source filter lists the documents", len(options) > 1,
                  str(len(options)))
            if len(options) > 1:
                selectors[0].set_value(options[1])
                submits = [b for b in at.button if "Compare" in str(b.label)]
                if submits:
                    submits[0].click().run()
                filtered = at.session_state.get("bundle")
                check("source filter is stored in the bundle",
                      filtered is not None and filtered.source == options[1],
                      getattr(filtered, "source", "none"))
                if filtered is not None:
                    every = filtered.keyword + filtered.vector + filtered.hybrid
                    check("filtered hits stay in the chosen source",
                          bool(every) and all(h.source == options[1] for h in every),
                          f"{len(every)} hits")
        dates = [d for d in at.date_input
                 if d.key in ("lab_date_from", "lab_date_to")]
        check("lab offers published date filters", len(dates) == 2,
              str([d.key for d in at.date_input]))
        if len(dates) == 2 and selectors:
            query_fields = [t for t in at.text_input if t.label == "Query"]
            if query_fields:
                query_fields[0].set_value("hybrid search")
            at.selectbox(key="lab_source").set_value("All sources")
            by_key = {d.key: d for d in at.date_input}
            by_key["lab_date_from"].set_value(dt.date(2026, 2, 14))
            by_key["lab_date_to"].set_value(dt.date(2026, 2, 14))
            submits = [b for b in at.button if "Compare" in str(b.label)]
            if submits:
                submits[0].click().run()
            dated = at.session_state.get("bundle")
            check("date filter is stored in the bundle",
                  dated is not None and dated.date_from == "2026-02-14"
                  and dated.date_to == "2026-02-14"
                  and not dated.source,
                  f"{getattr(dated, 'date_from', 'none')} to "
                  f"{getattr(dated, 'date_to', 'none')}, "
                  f"source {getattr(dated, 'source', 'none')}")
            if dated is not None:
                every = dated.keyword + dated.vector + dated.hybrid + dated.rerank
                check("date filtered hits all fall on that day",
                      bool(every) and all(h.published == "2026-02-14" for h in every),
                      f"{len(every)} hits, "
                      f"{sorted({h.published for h in every})}")
            check("rows show the published date",
                  "2026-02-14" in markdown_text(at))
        body = markdown_text(at)
        check("result rows rendered", 'class="row"' in body)
        check("latency rendered", " ms</div>" in body or "ms" in body)

    print("\n== other surfaces ==")
    for page, expected in (("Corpus", "Sources"), ("Quality", "labelled questions"), ("Health", "Container")):
        at.segmented_control[0].set_value(page).run()
        check(f"{page} renders without exception", not at.exception, str(at.exception))
        check(f"{page} content present", expected.lower() in markdown_text(at).lower(), expected)

    at.segmented_control[0].set_value("Corpus").run()
    frames = [frame.value for frame in at.dataframe]
    check("corpus table shows published dates",
          any("published" in frame.columns for frame in frames))

    print("\n== quality tab ==")
    at.segmented_control[0].set_value("Quality").run()
    body = markdown_text(at)
    check("quality states the question mix",
          "24 single-document" in body and "8 cross-document" in body
          and "8 unanswerable" in body)
    run_eval = [b for b in at.button if "Run evaluation" in str(b.label)]
    check("quality offers running the evaluation", bool(run_eval))
    if run_eval:
        run_eval[0].click().run()
        check("evaluation runs without exception", not at.exception, str(at.exception))
        result = at.session_state.get("eval")
        check("evaluation stored in session", isinstance(result, dict), type(result).__name__)
        if isinstance(result, dict):
            check("evaluation is schema 3", result.get("version") == 3,
                  str(result.get("version")))
            check("evaluation covers 40 questions",
                  result.get("question_count") == 40, str(result.get("question_count")))
            check("evaluation has 40 rows", len(result.get("rows") or []) == 40)
            check("evaluation keeps all four paths",
                  set(result.get("summary") or {}) == {"keyword", "vector", "hybrid", "rerank"})
            check("evaluation splits by question type",
                  set(result.get("by_type") or {}) == {"single", "cross"})
            check("evaluation has unanswerable stats",
                  set(result.get("unanswerable") or {}) == {"keyword", "vector", "hybrid", "rerank"})
            check("evaluation has answerable top scores",
                  set(result.get("answerable_tops") or {}) == {"keyword", "vector", "hybrid", "rerank"})
            check("evaluation records the question hash", bool(result.get("questions_hash")))
            check("evaluation records the corpus fingerprint",
                  bool(result.get("corpus_fingerprint")))
            check("evaluation records the corpus size",
                  int(result.get("corpus_rows") or 0) >= 10,
                  str(result.get("corpus_rows")))

        frames = [frame.value for frame in at.dataframe]
        summary = next((f for f in frames
                        if {"path", "hit@k", "MRR", "found"} <= set(f.columns)
                        and "set" not in f.columns), None)
        check("summary table rendered", summary is not None and len(summary) == 4,
              "none" if summary is None else str(len(summary)))
        breakdown = next((f for f in frames
                          if {"set", "path", "hit@k", "MRR"} <= set(f.columns)), None)
        check("type breakdown rendered", breakdown is not None and len(breakdown) == 8,
              "none" if breakdown is None else str(len(breakdown)))
        comparison = next((f for f in frames
                           if {"set", "mean top", "min top"} <= set(f.columns)), None)
        check("unanswerable comparison rendered",
              comparison is not None and len(comparison) == 2,
              "none" if comparison is None else str(len(comparison)))
        per_question = next((f for f in frames
                             if {"question", "set", "top cosine"} <= set(f.columns)), None)
        check("per-question table rendered",
              per_question is not None and len(per_question) == 40,
              "none" if per_question is None else str(len(per_question)))
        if per_question is not None:
            check("per-question marks unanswerable rows",
                  int((per_question["set"] == "unanswerable").sum()) == 8)
            check("per-question includes the rerank column",
                  "rerank" in per_question.columns)
            check("per-question shows ranks for answerable rows",
                  any(str(value).startswith("#") for value in per_question["hybrid"]))
            check("per-question shows rerank ranks",
                  any(str(value).startswith("#") for value in per_question["rerank"]))

        stale = dict(result) if isinstance(result, dict) else {}
        stale["questions_hash"] = "stale"
        stale["corpus_fingerprint"] = "stale"
        at.session_state["eval"] = stale
        at.run()
        check("stale evaluation is rejected", "changed since this run" in alert_text(at))
        if isinstance(result, dict):
            at.session_state["eval"] = result
            at.run()

    body = markdown_text(at)
    check("quality offers the answer gate",
          "Answer quality" in body and "labelled answers" in body)
    limits = [n for n in at.number_input if n.key == "answer_limit"]
    check("answer gate has a run limit", bool(limits))
    answer_runs = [b for b in at.button if b.key == "run_answer_check"]
    check("answer gate offers a live check", bool(answer_runs))
    if limits and answer_runs:
        limits[0].set_value(1).run()
        original_answer = rc.Agent.ask

        def fake_answer(self, question, **kwargs):
            return rc.Answer(
                text="The corpus does not contain that information.",
                model="fake", steps=[], sources=[], calls=0, elapsed=0.0,
            )

        rc.Agent.ask = fake_answer  # type: ignore[assignment]
        try:
            at.button(key="run_answer_check").click().run()
            check("answer check runs without exception",
                  not at.exception, str(at.exception))
            summary = at.session_state.get("answer_check")
            check("answer check stores a summary", isinstance(summary, dict),
                  type(summary).__name__)
            if isinstance(summary, dict):
                check("answer check covers the chosen limit",
                      summary.get("total") == 1, str(summary.get("total")))
                check("answer check records its checks",
                      set(summary.get("checks") or {})
                      == {"source", "facts", "citations"},
                      str(set(summary.get("checks") or {})))
            check("answer check renders a pass rate",
                  "cases passed" in markdown_text(at))
            frames = [frame.value for frame in at.dataframe]
            check("answer check renders a per-case table",
                  any("failed checks" in frame.columns for frame in frames))
        finally:
            rc.Agent.ask = original_answer  # type: ignore[assignment]

    at.segmented_control[0].set_value("Health").run()
    health = markdown_text(at)
    check("health shows the answer cache", "Answer cache" in health and "entries" in health)
    check("sidebar shows the cache size", ">cache</span>" in health)
    check("health offers clearing the cache",
          any("Clear answer cache" in str(b.label) for b in at.button))
    check("health shows the corpus backup", "Corpus backup" in health)
    check("health offers preparing a backup",
          any("Prepare corpus backup" in str(b.label) for b in at.button))

    at.button(key="prepare_backup").click().run()
    check("backup prepares without exception", not at.exception, str(at.exception))
    backup = at.session_state.get("backup")
    check("backup stored in session", isinstance(backup, dict), type(backup).__name__)
    if isinstance(backup, dict):
        check("backup includes every chunk", backup.get("chunks", 0) >= 10,
              str(backup.get("chunks")))
        name = str(backup.get("name", ""))
        check("backup filename is timestamped json",
              name.startswith("corpus-") and name.endswith(".json"), name)
        try:
            parsed = json.loads(backup["data"])
            valid = (parsed.get("format") == "oracle-rag-corpus"
                     and len(parsed.get("documents") or []) == backup.get("chunks"))
        except Exception:
            valid = False
        check("backup bytes are a valid export", valid)
    downloads = [d for d in at.download_button if d.key == "download_backup"]
    check("download button offered",
          len(downloads) == 1 and "Download backup" in str(downloads[0].label))

    print("\n== discussion history ==")
    at.segmented_control[0].set_value("Ask").run()
    body = markdown_text(at)
    check("sidebar names the discussions section", "discussions" in body)
    new_buttons = [b for b in at.button if b.key == "new_chat"]
    check("sidebar offers a new discussion", bool(new_buttons))

    history = rc.ChatHistory(rc.RagEngine(rc.Settings.from_env()))
    chat_id = ""
    try:
        if new_buttons:
            new_buttons[0].click().run()
            chat_id = str(at.session_state.get("chat_id") or "")
            check("new discussion gets an id", bool(chat_id), chat_id)
            check("new discussion starts empty", not at.session_state.get("chat"))
            at.session_state["local_mode"] = True
            if at.chat_input:
                at.chat_input[0].set_value(
                    "Which company raised an $85M Series B?").run()
                check("local question renders", not at.exception, str(at.exception))
                saved = history.messages(chat_id)
                check("question is saved in Oracle",
                      len(saved) >= 1 and saved[0]["role"] == "user")
                check("local answer is saved in Oracle",
                      len(saved) >= 2 and saved[1]["role"] == "assistant")
                listed = history.list(limit=50)
                check("discussion appears in the sidebar",
                      any(item["id"] == chat_id for item in listed))

                finder = [t for t in at.text_input if t.key == "history_search"]
                check("sidebar offers discussion search", bool(finder))
                if finder:
                    finder[0].set_value("Series B").run()
                    check("discussion search renders", not at.exception,
                          str(at.exception))
                    check("matching discussion stays selectable",
                          any(b.key == f"chat_{chat_id}" for b in at.button))
                    check("search offers a clear action",
                          any(b.key == "clear_history_search" for b in at.button))
                    search_body = markdown_text(at)
                    check("search reports the match count",
                          "1 discussion match" in search_body,
                          search_body[-160:])
                    check("search highlights the matching words",
                          "<mark>" in search_body)

                rename = [t for t in at.text_input
                          if t.key == f"rename_title_{chat_id}"]
                check("manage offers a title field", bool(rename))
                if rename:
                    rename[0].set_value("Series B follow-up").run()
                    at.button(key="rename_chat").click().run()
                    check("rename is saved in Oracle",
                          history.title(chat_id) == "Series B follow-up",
                          history.title(chat_id))
                check("manage offers a discussion download",
                      any(d.key == "download_discussion" for d in at.download_button))
                check("sidebar offers follow-up context",
                      any(t.key == "follow_context" for t in at.toggle))

                original_ask = rc.Agent.ask
                seen: dict = {"context": [], "tokens": []}

                def fake_ask(self, question, on_step=None, recursion_limit=25,
                             use_cache=True, on_token=None, on_reset=None,
                             context=None):
                    seen["context"] = list(context or [])
                    if on_reset:
                        on_reset()
                    for piece in ("Streamed ", "answer."):
                        seen["tokens"].append(piece)
                        if on_token:
                            on_token(piece)
                    return rc.Answer(
                        text="Streamed answer.", model="fake",
                        steps=[rc.Step("answer", "answer ready")],
                        sources=[{
                            "source": "01_nvidia_ai_infrastructure.md",
                            "relevance": 0.99,
                            "snippet": "NVIDIA today announced a sweeping set of "
                                       "infrastructure partnerships.",
                            "published": "2026-03-10",
                        }],
                        calls=0, elapsed=0.1,
                    )

                rc.Agent.ask = fake_ask  # type: ignore[assignment]
                try:
                    at.session_state["local_mode"] = False
                    at.session_state["follow_context"] = True
                    at.chat_input[0].set_value("And who funded them?").run()
                    check("streaming UI renders without exception",
                          not at.exception, str(at.exception))
                    check("streamed answer is shown",
                          "Streamed answer." in markdown_text(at))
                    body = markdown_text(at)
                    check("answer cites a clickable source",
                          "Click one to read the passage" in body)
                    check("citation shows the passage",
                          "NVIDIA today announced" in body)
                    check("citation shows the published date",
                          "2026-03-10" in body)
                    check("streamed tokens were delivered",
                          "".join(seen["tokens"]) == "Streamed answer.",
                          repr("".join(seen["tokens"])))
                    check("follow-up context reached the agent",
                          len(seen["context"]) >= 2, str(len(seen["context"])))
                    saved = history.messages(chat_id)
                    check("streamed answer is saved in Oracle",
                          any(m["role"] == "assistant"
                              and getattr(m.get("answer"), "text", "") == "Streamed answer."
                              for m in saved))
                finally:
                    rc.Agent.ask = original_ask  # type: ignore[assignment]

                second = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
                second.run()
                check("reopening restores the latest discussion",
                      second.session_state.get("chat_id") == chat_id,
                      str(second.session_state.get("chat_id")))
                restored = second.session_state.get("chat") or []
                check("restored discussion keeps its messages",
                      len(restored) >= 2 and restored[0]["role"] == "user")
                if restored and restored[0]["role"] == "user":
                    check("restored question matches",
                          "Series B" in restored[0]["content"])
            else:
                check("chat input is available", False)
    finally:
        if chat_id:
            history.delete(chat_id)
        history.e.close()

    print("\n" + "-" * 60)
    if failures:
        print(f"  {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  app renders and searches correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
