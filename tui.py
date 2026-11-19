"""tui.py - full-screen terminal interface for the Oracle RAG Console.

    python tui.py
    python tui.py --local
    python tui.py --no-history
    python tui.py --theme light

This is the pane-and-mouse version of `ask.py --chat`. It uses the same
`Agent`, `ChatHistory` and `artifacts` helpers as the Streamlit app, so answers,
follow-up context, saved discussions and artefacts are shared by both surfaces.

The left side is the discussion and the prompt. The right side has three panes:
sources for the last answer, saved discussions, and saved artefacts. Keys:

    ctrl+q  quit            ctrl+t  dark / light
    ctrl+s  save artefact   ctrl+l  focus the prompt

Type `/help` for the command list.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import artifacts as arts
import rag_core as rc

try:
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.text import Text
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.suggester import SuggestFromList
    from textual.widgets import (Button, Footer, Header, Input, OptionList, RichLog,
                                 Static, TabbedContent, TabPane)
except ImportError as exc:  # pragma: no cover - shown as a readable error
    print("error: the TUI needs textual and rich.")
    print("install them with:")
    print(r"  %USERPROFILE%\rag-venv\Scripts\python.exe -m pip install -r requirements.txt")
    print(f"({exc})")
    raise SystemExit(2) from exc


COMMANDS = (
    "/help", "/local", "/remote", "/mode", "/search", "/context",
    "/new", "/history", "/open", "/find", "/clear", "/sources",
    "/trace", "/save", "/artefacts", "/preview", "/title", "/delete",
    "/theme", "/quit",
)

# One source of truth for /help and for the live slash-command popup.
COMMAND_HELP = (
    ("/help", "this list"),
    ("/local", "retrieval only, no model, zero quota"),
    ("/remote", "use the Gemini agent again"),
    ("/mode local|remote", "set the answer mode"),
    ("/search hybrid|vector|keyword|rerank", "choose corpus retrieval"),
    ("/context on|off", "toggle follow-up context"),
    ("/new", "start a fresh discussion"),
    ("/history", "the twelve most recent discussions"),
    ("/open N", "reopen discussion N from that list"),
    ("/find words", "search saved questions and answers"),
    ("/clear", "clear the transcript display"),
    ("/sources", "sources of the last answer"),
    ("/trace", "trace of the last answer"),
    ("/save", "save the last answer as Markdown, JSON and HTML"),
    ("/artefacts", "list saved answer files"),
    ("/preview N", "read artefact N in the transcript"),
    ("/title [name]", "show or rename this discussion"),
    ("/delete", "delete this discussion, twice to confirm"),
    ("/theme", "switch dark and light"),
    ("/quit", "leave the TUI"),
)


def filter_commands(token: str) -> list[tuple[str, str]]:
    """Commands whose base name starts with the typed ``/...`` token."""
    value = (token or "").strip().lower()
    if not value.startswith("/"):
        return []
    head = value.split()[0] if value.split() else value
    return [(name, description) for name, description in COMMAND_HELP
            if name.split()[0].startswith(head)]


class SlashSuggester(SuggestFromList):
    """Inline ghost completion that only kicks in for slash commands."""

    def __init__(self) -> None:
        super().__init__([name for name, _ in COMMAND_HELP], case_sensitive=False)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        return await super().get_suggestion(value)


def when(value: object) -> str:
    try:
        import datetime as dt

        stamp = dt.datetime.fromisoformat(str(value))
        now = dt.datetime.now().astimezone()
        if stamp.date() == now.date():
            return stamp.strftime("%H:%M")
        if stamp.year == now.year:
            return stamp.strftime("%b %d")
        return stamp.strftime("%b %d, %Y")
    except Exception:
        return ""


def source_label(item: dict[str, Any]) -> str:
    source = str(item.get("source", "unknown"))
    published = str(item.get("published") or "")
    numbers = item.get("citations")
    if not isinstance(numbers, (list, tuple)) or not numbers:
        numbers = [item["citation"]] if item.get("citation") else []
    prefix = f"[{', '.join(str(n) for n in numbers)}] " if numbers else ""
    label = f"{prefix}{source}"
    return f"{label}  {published}" if published else label


class OracleTui(App[None]):
    """A small dashboard over `rag_core`; it owns no SQL and no prompts."""

    TITLE = "Oracle RAG Console"
    SUB_TITLE = "26ai · terminal"
    CSS = """
    Screen { background: $surface; }
    #body { height: 1fr; }
    #left { width: 2fr; border-right: solid $border; }
    #right { width: 1fr; min-width: 34; }
    #toolbar { height: 3; padding: 0 1; background: $panel; }
    #status { width: 1fr; content-align: left middle; opacity: 70%; }
    #toolbar Button { min-width: 9; margin-left: 1; }
    #transcript { height: 1fr; padding: 0 1; background: $surface; }
    #prompt { margin: 1 1 0 1; border: tall $accent; }
    #command-suggestions { display: none; height: auto; max-height: 9;
      margin: 0 1 1 1; border: round $accent; background: $panel; }
    #command-suggestions.visible { display: block; }
    TabbedContent { height: 1fr; }
    TabPane { height: 1fr; padding: 0 1; }
    RichLog { height: 1fr; background: $surface; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+t", "toggle_theme", "Theme"),
        Binding("ctrl+s", "save_artefact", "Save"),
        Binding("ctrl+l", "focus_prompt", "Ask"),
    ]

    def __init__(
        self,
        settings: rc.Settings,
        agent: rc.Agent,
        history: rc.ChatHistory | None = None,
        *,
        local: bool = False,
        k: int = 4,
        weight: float = 0.6,
        use_cache: bool = True,
        dark: bool = True,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.agent = agent
        self.history = history
        self.local = bool(local)
        self.k = max(1, int(k))
        self.weight = float(weight)
        self.search_method = "hybrid"
        self.follow_context = True
        self.use_cache = bool(use_cache)
        self.dark = bool(dark)
        self.chat_id: str | None = None
        self.turns: list[dict[str, Any]] = []
        self.last_answer: rc.Answer | None = None
        self.last_question = ""
        self._listed: list[dict[str, Any]] = []
        self._artefacts: list[arts.Artefact] = []
        self._confirm_delete = False
        self._busy = False
        self._suggest_matches: list[tuple[str, str]] = []

    # -- layout --------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                with Horizontal(id="toolbar"):
                    yield Static("ready", id="status")
                    yield Button("Search: Hybrid", id="search")
                    yield Button("Theme", id="theme")
                    yield Button("Save artefact", id="save")
                yield RichLog(id="transcript", wrap=True, markup=False,
                              highlight=True, max_lines=4000)
                yield Input(placeholder="Ask the corpus, or type /help", id="prompt",
                            suggester=SlashSuggester())
                yield OptionList(id="command-suggestions")
            with Vertical(id="right"):
                with TabbedContent(id="panes"):
                    with TabPane("Sources", id="sources-tab"):
                        yield RichLog(id="sources", wrap=True, markup=False,
                                      highlight=False, max_lines=2000)
                    with TabPane("History", id="history-tab"):
                        yield RichLog(id="history", wrap=True, markup=False,
                                      highlight=False, max_lines=2000)
                    with TabPane("Artefacts", id="artefacts-tab"):
                        yield RichLog(id="artefacts", wrap=True, markup=False,
                                      highlight=False, max_lines=2000)
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "textual-dark" if self.dark else "textual-light"
        self._welcome()
        self._restore()
        self._refresh_history()
        self._refresh_artefacts()
        self.query_one("#prompt", Input).focus()

    # -- small UI helpers ----------------------------------------------------
    def _say(self, renderable: Any) -> None:
        self.query_one("#transcript", RichLog).write(renderable)

    def _set_status(self, message: str, style: str = "") -> None:
        text = Text(message, style=style or "dim")
        self.query_one("#status", Static).update(text)

    def _clear(self, widget_id: str) -> RichLog:
        log = self.query_one(f"#{widget_id}", RichLog)
        log.clear()
        return log

    def _welcome(self) -> None:
        self._say(Text("Oracle RAG Console", style="bold"))
        self._say(Text("Ask a question, or type /help for commands.", style="dim"))
        if self.local:
            self._say(Text("Local mode: retrieval only, no model calls.", style="dim"))
        self._say(Text(""))

    # -- restore and refresh -------------------------------------------------
    def _restore(self) -> None:
        if self.history is None:
            self.chat_id = uuid.uuid4().hex
            self._say(Text("History is off for this session.", style="dim"))
            return
        try:
            recent = self.history.list(limit=1)
            if not recent:
                self.chat_id = uuid.uuid4().hex
                return
            item = recent[0]
            self.turns = self.history.messages(item["id"])
            self.chat_id = item["id"]
        except rc.RagError as exc:
            self.history = None
            self.chat_id = uuid.uuid4().hex
            self._say(Text(f"History unavailable: {exc}", style="yellow"))
            return
        self.last_answer = self._last_answer(self.turns)
        self.last_question = self._last_question(self.turns)
        self._say(Text(
            f"Continuing {item['title']} ({item['messages']} messages). "
            "/new starts fresh.",
            style="dim",
        ))
        self._say(Text(""))
        for turn in self.turns[-6:]:
            role = turn.get("role")
            if role == "user":
                self._log_user(str(turn.get("content") or ""))
            elif role == "error":
                self._say(Text(f"error: {turn.get('content')}", style="red"))
            elif role == "assistant":
                answer = turn.get("answer")
                if answer is not None:
                    self._log_answer(rc.enforce_answer_grounding(answer),
                                     show_sources=False)

    @staticmethod
    def _last_answer(turns: list[dict[str, Any]]) -> rc.Answer | None:
        for item in reversed(turns):
            answer = item.get("answer")
            if answer is not None:
                return answer
        return None

    @staticmethod
    def _last_question(turns: list[dict[str, Any]]) -> str:
        for item in reversed(turns):
            if item.get("role") == "user":
                return str(item.get("content") or "")
        return ""

    def _refresh_history(self) -> None:
        log = self._clear("history")
        if self.history is None:
            log.write(Text("History is off. Start with --no-history removed.", style="dim"))
            return
        try:
            self._listed = self.history.list(limit=12)
        except rc.RagError as exc:
            log.write(Text(f"History unavailable: {exc}", style="yellow"))
            return
        if not self._listed:
            log.write(Text("No saved discussions yet.", style="dim"))
            return
        for index, item in enumerate(self._listed, start=1):
            mark = "*" if item["id"] == self.chat_id else " "
            line = Text.assemble(
                (f" {mark} {index}. ", "dim"),
                (str(item["title"]), "bold" if mark.strip() else ""),
                (f"  ({item['messages']} messages, {when(item['updated'])})", "dim"),
            )
            log.write(line)
        log.write(Text(""))
        log.write(Text("/open N reopens one.", style="dim"))

    def _refresh_artefacts(self) -> None:
        log = self._clear("artefacts")
        try:
            self._artefacts = arts.list_artefacts(limit=30)
        except OSError as exc:
            log.write(Text(f"Could not read artefacts: {exc}", style="yellow"))
            return
        if not self._artefacts:
            log.write(Text("No artefacts yet. Press Save artefact after an answer.", style="dim"))
            return
        for index, item in enumerate(self._artefacts, start=1):
            log.write(Text.assemble(
                (f"{index}. ", "dim"),
                (item.title, ""),
                (f"\n   {item.kind} · {arts.human_size(item.size)} · "
                 f"{when(item.created) or 'unknown'}", "dim"),
            ))
        log.write(Text(""))
        log.write(Text("/preview N reads one here.", style="dim"))

    def _refresh_sources(self, answer: rc.Answer | None = None) -> None:
        log = self._clear("sources")
        answer = answer or self.last_answer
        if answer is None:
            log.write(Text("No answer yet.", style="dim"))
            return
        answer = rc.enforce_answer_grounding(answer)
        if answer.local:
            if not answer.hits:
                log.write(Text("No hits.", style="dim"))
                return
            for index, hit in enumerate(answer.hits[: self.k], start=1):
                log.write(Text.assemble(
                    (f"{index}. ", "dim"),
                    (hit.source, "bold"),
                    (f"  {hit.score:.3f}", "cyan"),
                    (f"\n   {hit.snippet(220)}", "dim"),
                ))
            return
        if not answer.sources:
            log.write(Text("This answer listed no sources.", style="dim"))
            return
        for item in answer.sources:
            if not isinstance(item, dict):
                item = {"source": str(item)}
            score = float(item.get("relevance", 0.0) or 0.0)
            log.write(Text.assemble(
                (source_label(item), "bold"),
                (f"  {score:.3f}", "cyan"),
            ))
            snippet = str(item.get("snippet") or "").strip()
            if snippet:
                log.write(Text("   " + " ".join(snippet.split())[:300], style="dim"))

    # -- input ---------------------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        """Show every slash command after '/', then narrow it as the user types."""
        if event.input.id != "prompt":
            return
        popup = self.query_one("#command-suggestions", OptionList)
        value = str(event.value or "")
        matches = filter_commands(value)
        if not matches:
            popup.remove_class("visible")
            self._suggest_matches = []
            return
        self._suggest_matches = matches
        from textual.widgets.option_list import Option
        popup.set_options([Option(f"{name:<30}  {description}")
                           for name, description in matches])
        popup.add_class("visible")

    def on_option_list_option_selected(self,
                                       event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "command-suggestions":
            return
        index = event.option_index
        if index is None or index >= len(self._suggest_matches):
            return
        name = self._suggest_matches[index][0].split()[0]
        prompt = self.query_one("#prompt", Input)
        popup = self.query_one("#command-suggestions", OptionList)
        prompt.value = name + " "
        popup.remove_class("visible")
        self._suggest_matches = []
        prompt.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = str(event.value or "").strip()
        event.input.value = ""
        popup = self.query_one("#command-suggestions", OptionList)
        popup.remove_class("visible")
        self._suggest_matches = []
        if not line:
            return
        if line.startswith("/"):
            self._command(line)
        else:
            self._ask(line)
        self.query_one("#prompt", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "theme":
            self.action_toggle_theme()
        elif event.button.id == "save":
            self.action_save_artefact()
        elif event.button.id == "search":
            methods = list(rc.SEARCH_METHODS)
            self.search_method = methods[(methods.index(self.search_method) + 1)
                                         % len(methods)]
            labels = {"hybrid": "Hybrid", "vector": "Vector",
                      "keyword": "Keyword", "rerank": "Reranked"}
            self.query_one("#search", Button).label = f"Search: {labels[self.search_method]}"
            self._set_status(f"Ask retrieval set to {labels[self.search_method]}")

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_toggle_theme(self) -> None:
        self.dark = not self.dark
        self.theme = "textual-dark" if self.dark else "textual-light"
        self._set_status(f"{'dark' if self.dark else 'light'} theme")

    def action_save_artefact(self) -> None:
        if self.last_answer is None:
            self._set_status("Ask something before saving an artefact", "yellow")
            return
        try:
            saved = arts.save_answer(
                self.last_question, self.last_answer, chat_id=self.chat_id or "")
        except (OSError, ValueError) as exc:
            self._set_status(f"Could not save: {exc}", "red")
            return
        self._refresh_artefacts()
        self._set_status(f"Saved {len(saved)} files to {saved[0].path.parent}", "green")
        self.query_one(TabbedContent).active = "artefacts-tab"

    # -- commands ------------------------------------------------------------
    def _command(self, line: str) -> None:
        name, _, rest = line.partition(" ")
        name = name.lower()
        arg = rest.strip()
        if name != "/delete":
            self._confirm_delete = False
        if name in ("/quit", "/exit", "/q"):
            self.exit()
        elif name == "/help":
            self._help()
        elif name == "/local":
            self.local = True
            self._set_status("local mode: no model calls")
            self._say(Text("Local mode on: retrieval only, no model calls.", style="dim"))
        elif name == "/remote":
            self.local = False
            self._set_status("remote mode: Gemini answers")
            self._say(Text("Remote mode on: the Gemini agent will answer.", style="dim"))
        elif name == "/mode":
            if arg.lower() in ("local", "remote"):
                self._command("/" + arg.lower())
            else:
                self._say(Text("usage: /mode local|remote", style="yellow"))
        elif name == "/search":
            self._search_method(arg)
        elif name == "/context":
            self._context_mode(arg)
        elif name == "/new":
            self._new()
        elif name == "/history":
            self._refresh_history()
            self.query_one(TabbedContent).active = "history-tab"
        elif name == "/open":
            self._open(arg)
        elif name == "/find":
            self._find(arg)
        elif name == "/clear":
            self._clear("transcript")
            self._welcome()
        elif name == "/sources":
            self._refresh_sources()
            self.query_one(TabbedContent).active = "sources-tab"
        elif name == "/trace":
            self._trace()
        elif name == "/save":
            self.action_save_artefact()
        elif name == "/artefacts":
            self._refresh_artefacts()
            self.query_one(TabbedContent).active = "artefacts-tab"
        elif name == "/preview":
            self._preview(arg)
        elif name == "/title":
            self._title(arg)
        elif name == "/delete":
            self._delete()
        elif name == "/theme":
            self.action_toggle_theme()
        else:
            self._say(Text(f"Unknown command {name}. /help lists them.", style="yellow"))

    def _help(self) -> None:
        self._say(Text("Commands", style="bold"))
        for name, description in COMMAND_HELP:
            self._say(Text.assemble((f"  {name:<14}", "cyan"), (description, "")))

    def _search_method(self, arg: str) -> None:
        if not arg:
            self._say(Text(f"Current retrieval: {self.search_method}"))
            return
        try:
            self.search_method = rc.normalise_search_method(arg)
        except rc.RagError as exc:
            self._say(Text(str(exc), style="yellow"))
            return
        labels = {"hybrid": "Hybrid", "vector": "Vector",
                  "keyword": "Keyword", "rerank": "Reranked"}
        self.query_one("#search", Button).label = f"Search: {labels[self.search_method]}"
        self._set_status(f"Ask retrieval set to {labels[self.search_method]}")

    def _context_mode(self, arg: str) -> None:
        value = arg.lower()
        if value not in ("on", "off"):
            self._say(Text("usage: /context on|off", style="yellow"))
            return
        self.follow_context = value == "on"
        self._set_status(f"follow-up context {value}")

    def _new(self) -> None:
        self.chat_id = uuid.uuid4().hex
        self.turns = []
        self.last_answer = None
        self.last_question = ""
        self._confirm_delete = False
        self._say(Text("New discussion. Ask something.", style="dim"))
        self._set_status("new discussion")

    def _find(self, arg: str) -> None:
        if not arg:
            self._say(Text("usage: /find words from a past chat", style="yellow"))
            return
        if self.history is None:
            self._say(Text("History is off for this session.", style="yellow"))
            return
        try:
            found = self.history.search(arg, limit=10)
        except rc.RagError as exc:
            self._say(Text(str(exc), style="yellow"))
            return
        self._listed = found
        if not found:
            self._say(Text("No discussion matches those words.", style="dim"))
            return
        self._say(Text(f"{len(found)} discussions match.", style="dim"))
        for index, item in enumerate(found, start=1):
            self._say(Text.assemble(
                (f" {index}. ", "dim"),
                (str(item["title"]), "bold"),
                (f"  ({item.get('matches', 0)} matching messages)", "dim"),
            ))
            snippet = str(item.get("snippet") or "").strip()
            if snippet:
                self._say(Text("    " + snippet[:240], style="dim"))
        self._refresh_history()
        self.query_one(TabbedContent).active = "history-tab"

    def _open(self, arg: str) -> None:
        if self.history is None:
            self._say(Text("History is off for this session.", style="yellow"))
            return
        try:
            index = int(arg)
        except ValueError:
            self._say(Text("usage: /open N (see /history)", style="yellow"))
            return
        if not self._listed:
            self._refresh_history()
        if not 1 <= index <= len(self._listed):
            self._say(Text("No discussion at that number. Try /history.", style="yellow"))
            return
        item = self._listed[index - 1]
        try:
            turns = self.history.messages(item["id"])
        except rc.RagError as exc:
            self._say(Text(str(exc), style="yellow"))
            return
        self.chat_id = item["id"]
        self.turns = turns
        self.last_answer = self._last_answer(turns)
        self.last_question = self._last_question(turns)
        self._confirm_delete = False
        self._say(Text(f"Opened {item['title']}", style="bold"))
        for turn in turns[-4:]:
            role = turn.get("role")
            if role == "user":
                self._log_user(str(turn.get("content") or ""))
            elif role == "error":
                self._say(Text(f"error: {turn.get('content')}", style="red"))
            elif role == "assistant" and turn.get("answer") is not None:
                self._log_answer(turn["answer"], show_sources=False)
        self._refresh_sources()
        self._refresh_history()
        self._set_status(f"opened {item['title']}")

    def _trace(self) -> None:
        answer = (rc.enforce_answer_grounding(self.last_answer)
                  if self.last_answer is not None else None)
        if answer is None:
            self._say(Text("No answer yet.", style="dim"))
            return
        if not answer.steps:
            self._say(Text("No trace for this answer.", style="dim"))
            return
        self._say(Text("Trace", style="bold"))
        for step in answer.steps:
            detail = f"  {step.detail}" if step.detail else ""
            self._say(Text.assemble((f"  {step.kind}: ", "cyan"),
                                    (step.label, ""), (detail, "dim")))

    def _preview(self, arg: str) -> None:
        try:
            index = int(arg)
        except ValueError:
            self._say(Text("usage: /preview N (see /artefacts)", style="yellow"))
            return
        if not self._artefacts:
            self._refresh_artefacts()
        if not 1 <= index <= len(self._artefacts):
            self._say(Text("No artefact at that number. Try /artefacts.", style="yellow"))
            return
        item = self._artefacts[index - 1]
        self._say(Text(f"{item.title} · {item.filename}", style="bold"))
        body = item.preview()
        if item.kind == "markdown":
            self._say(Markdown(body))
        else:
            self._say(Text(body))
        self._say(Text(""))

    def _title(self, arg: str) -> None:
        if self.history is None or not self.chat_id:
            self._say(Text("No saved discussion to title.", style="yellow"))
            return
        try:
            current = self.history.title(self.chat_id)
        except rc.RagError as exc:
            self._say(Text(str(exc), style="yellow"))
            return
        if not arg:
            self._say(Text(f"Title: {current}"))
            return
        label = " ".join(arg.split())
        try:
            self.history.rename(self.chat_id, label)
        except rc.RagError as exc:
            self._say(Text(str(exc), style="yellow"))
            return
        self._say(Text(f"Renamed to {label}", style="dim"))
        self._refresh_history()
        self._set_status(f"renamed to {label}")

    def _delete(self) -> None:
        if self.history is None or not self.chat_id:
            self._say(Text("No saved discussion to delete.", style="yellow"))
            return
        try:
            title = self.history.title(self.chat_id)
        except rc.RagError:
            title = "this discussion"
        if not self._confirm_delete:
            self._confirm_delete = True
            self._say(Text(f"Delete {title}? Type /delete again to confirm.", style="yellow"))
            return
        try:
            removed = self.history.delete(self.chat_id)
        except rc.RagError as exc:
            self._say(Text(str(exc), style="yellow"))
            self._confirm_delete = False
            return
        self._say(Text(f"Deleted {removed} messages.", style="dim"))
        self._new()
        self._refresh_history()

    # -- answering -----------------------------------------------------------
    def _chat_context(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self.turns:
            role = item.get("role")
            if role == "user":
                out.append({"role": "user", "content": str(item.get("content", ""))})
            elif role == "assistant" and item.get("answer") is not None:
                out.append({"role": "assistant", "answer": item["answer"]})
        return out

    def _ask(self, question: str) -> None:
        if self._busy:
            self._set_status("an answer is already running", "yellow")
            return
        self._busy = True
        self.last_question = question
        context = self._chat_context() if self.follow_context else []
        self.turns.append({"role": "user", "content": question})
        self._log_user(question)
        self._save("user", question)
        self._set_status("searching the corpus…")
        self._answer_worker(question, context)

    @work(thread=True, exclusive=True, group="answer")
    def _answer_worker(self, question: str, context: list[dict[str, Any]]) -> None:
        token_count = {"n": 0}

        def on_step(kind: str, text: str) -> None:
            self.call_from_thread(self._set_status, f"{kind}: {text}")

        def on_token(text: str) -> None:
            token_count["n"] += len(text)
            if token_count["n"] % 160 < len(text):
                self.call_from_thread(
                    self._set_status, "writing the answer…")

        try:
            if self.local:
                options = dict(k=self.k, weight=self.weight, context=context)
                if self.search_method != "hybrid":
                    options["search_method"] = self.search_method
                answer = self.agent.ask_local(question, **options)
            else:
                options = dict(use_cache=self.use_cache, context=context,
                               on_step=on_step, on_token=on_token)
                if self.search_method != "hybrid":
                    options["search_method"] = self.search_method
                answer = self.agent.ask(question, **options)
        except (rc.NoModel, rc.QuotaExhausted) as exc:
            self.call_from_thread(self._failed, str(exc),
                                  "try /local, or add a key to .env")
            return
        except Exception as exc:
            self.call_from_thread(self._failed, str(exc), "")
            return
        self.call_from_thread(self._finish_answer, answer)

    def _finish_answer(self, answer: rc.Answer) -> None:
        self._busy = False
        self.last_answer = answer
        self.turns.append({"role": "assistant", "answer": answer})
        self._save("assistant", answer.text or "", answer=answer)
        self._log_answer(answer, show_sources=True)
        self._refresh_sources(answer)
        self._refresh_history()
        if answer.cached:
            label = f"cached by {answer.model} · 0 API calls"
        elif answer.local:
            label = "local · 0 API calls"
        else:
            label = f"{answer.calls} API calls · {answer.elapsed:.1f}s · {answer.model}"
        self._set_status(label, "green")

    def _failed(self, message: str, hint: str) -> None:
        self._busy = False
        self._say(Text(f"error: {message}", style="red"))
        if hint:
            self._say(Text(hint, style="dim"))
        self.turns.append({"role": "error", "content": message})
        self._save("error", message)
        self._set_status("failed", "red")
        self._refresh_history()

    # -- output --------------------------------------------------------------
    def _log_user(self, question: str) -> None:
        self._say(Text.assemble(("you  ", "bold cyan"), (question, "")))
        self._say(Text(""))

    def _log_answer(self, answer: rc.Answer, *, show_sources: bool) -> None:
        if answer.local:
            table = Table(show_header=True, header_style="bold", box=None,
                          pad_edge=False)
            table.add_column("#", width=3, style="dim")
            table.add_column("source", overflow="fold")
            table.add_column("score", justify="right", style="cyan")
            for index, hit in enumerate(answer.hits[: self.k], start=1):
                table.add_row(str(index), hit.source, f"{hit.score:.3f}")
            self._say(Text("Local mode: no model was called.", style="dim"))
            self._say(table)
            for index, hit in enumerate(answer.hits[: self.k], start=1):
                self._say(Text(f"  {hit.snippet(240)}", style="dim"))
        elif answer.text.strip():
            self._say(Markdown(answer.text.strip()))
        else:
            self._say(Text("The model returned no text.", style="dim"))
        if show_sources and answer.sources:
            for item in answer.sources:
                if not isinstance(item, dict):
                    item = {"source": str(item)}
                score = float(item.get("relevance", 0.0) or 0.0)
                self._say(Text.assemble(
                    (source_label(item), "bold"),
                    (f"  {score:.3f}", "cyan"),
                ))
        self._say(Text(""))

    # -- persistence ---------------------------------------------------------
    def _save(self, role: str, content: str,
              answer: rc.Answer | None = None) -> None:
        if self.history is None or not self.chat_id:
            return
        try:
            self.history.append(self.chat_id, role, content, answer=answer)
        except rc.RagError as exc:
            self._say(Text(f"not saved: {exc}", style="yellow"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full-screen terminal interface for the Oracle RAG corpus.")
    parser.add_argument("--local", action="store_true",
                        help="retrieval only, no model call, zero quota")
    parser.add_argument("--no-history", action="store_true",
                        help="keep this session in memory only")
    parser.add_argument("--k", type=int, default=4,
                        help="hits to show in local mode (default: 4)")
    parser.add_argument("--weight", type=float, default=0.6,
                        help="hybrid weight, 0 keyword to 1 vector (default: 0.6)")
    parser.add_argument("--no-cache", action="store_true",
                        help="skip the answer cache and force a fresh model run")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark",
                        help="starting theme (default: dark)")
    args = parser.parse_args(argv)

    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    agent = rc.Agent(engine, settings)
    history = None if args.no_history else rc.ChatHistory(engine)
    app = OracleTui(
        settings,
        agent,
        history,
        local=args.local or not settings.has_llm,
        k=args.k,
        weight=args.weight,
        use_cache=not args.no_cache,
        dark=args.theme == "dark",
    )
    try:
        app.run()
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
