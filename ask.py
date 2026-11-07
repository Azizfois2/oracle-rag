"""Ask the corpus from a terminal.

    python ask.py "Which company raised a Series B?"
    python ask.py --local "hybrid search"
    python ask.py --json "Who partnered with NVIDIA?" > answer.json
    python ask.py --chat
    python ask.py --chat --local --no-history

Local mode never calls a model and spends no quota. It prints the top hybrid
hits with snippets instead of a written answer.

The chat mode is an interactive terminal session. It uses the same Agent and
ChatHistory as the Streamlit app, so a discussion started here appears in the
sidebar and can be reopened there. Type /help inside it for the commands.

Exit codes
  0  success
  2  bad input or database problem
  3  no usable model (missing key or quota spent) - retry with --local
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_core as rc

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text
except Exception:  # pragma: no cover - rich ships with the app
    Console = Live = Markdown = Text = None  # type: ignore[assignment]

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass


COMMANDS = (
    "/help", "/local", "/remote", "/new", "/history", "/open", "/find",
    "/sources", "/trace", "/title", "/delete", "/quit",
)


def print_local(answer: rc.Answer, k: int) -> None:
    print("Local mode: no model was called.")
    print()
    for index, hit in enumerate(answer.hits[:k], start=1):
        print(f"{index}. {hit.source}  ({hit.score:.3f})")
        print(f"   {hit.snippet(200)}")
        print()


def print_remote(answer: rc.Answer, show_steps: bool) -> None:
    print(answer.text.strip())
    print()
    if answer.sources:
        print("Sources")
        for item in answer.sources:
            source = item.get("source", "unknown")
            score = float(item.get("relevance", 0.0) or 0.0)
            print(f"  {source}  {score:.3f}")
        print()
    if show_steps:
        for step in answer.steps:
            detail = f" {step.detail}" if step.detail else ""
            print(f"  {step.kind}: {step.label}{detail}")
        print()
    if answer.cached:
        print(f"cached, 0 API calls (answered by {answer.model})")
    elif answer.local:
        print("0 API calls (local mode)")
    else:
        print(f"{answer.calls} API calls, {answer.elapsed:.1f} s, {answer.model}")


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


class ChatSession:
    """Interactive terminal chat over the same rag_core contract.

    It keeps no SQL and no prompt text of its own: history goes through
    ChatHistory and answers through Agent. A terminal discussion therefore
    shows up in the Streamlit sidebar, and both share one answer cache.
    """

    def __init__(
        self,
        agent: rc.Agent,
        history: rc.ChatHistory | None = None,
        *,
        local: bool = False,
        k: int = 4,
        weight: float = 0.6,
        show_steps: bool = False,
        use_cache: bool = True,
        console: Any = None,
        tty: bool | None = None,
        reader: Callable[[str], str | None] | None = None,
    ) -> None:
        if Console is None:
            raise rc.RagError("terminal chat needs the rich package: pip install rich")
        self.agent = agent
        self.history = history
        self.local = local
        self.k = max(1, int(k))
        self.weight = float(weight)
        self.show_steps = show_steps
        self.use_cache = use_cache
        self.console = console or Console()
        if tty is None:
            tty = bool(sys.stdin.isatty() and self.console.is_terminal)
        self.tty = tty
        self._reader = reader
        self._prompt: Any = None
        self.chat_id: str | None = None
        self.turns: list[dict[str, Any]] = []
        self.last_answer: rc.Answer | None = None
        self._listed: list[dict[str, Any]] = []
        self._confirm_delete = False

    # -- input ---------------------------------------------------------------
    def _read(self, prompt_text: str) -> str | None:
        if self._reader is not None:
            return self._reader(prompt_text)
        if self.tty:
            try:
                if self._prompt is None:
                    from prompt_toolkit import PromptSession
                    from prompt_toolkit.completion import WordCompleter
                    from prompt_toolkit.history import InMemoryHistory

                    self._prompt = PromptSession(
                        history=InMemoryHistory(),
                        completer=WordCompleter(COMMANDS, ignore_case=True),
                    )
                return self._prompt.prompt(prompt_text)
            except (EOFError, KeyboardInterrupt):
                return None
            except Exception:
                self.tty = False
        try:
            return input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            return None

    # -- lifecycle -----------------------------------------------------------
    def run(self, first: str = "") -> int:
        self._welcome()
        if first.strip():
            self._answer(first.strip())
        while True:
            line = self._read("you> ")
            if line is None:
                self.console.print()
                break
            if not self._handle(line):
                break
        self.console.print("[dim]Bye.[/dim]")
        return 0

    def _welcome(self) -> None:
        self.console.print("[bold]Oracle RAG Console[/bold] [dim]- terminal[/dim]")
        self.console.print("[dim]Ask a question, or type /help for commands.[/dim]")
        if self.local:
            self.console.print("[dim]Local mode: retrieval only, no model calls.[/dim]")
        self._restore()

    def _restore(self) -> None:
        if self.history is None:
            self.chat_id = uuid.uuid4().hex
            self.console.print("[dim]History is off for this session.[/dim]")
            return
        try:
            recent = self.history.list(limit=1)
        except rc.RagError as exc:
            self.history = None
            self.chat_id = uuid.uuid4().hex
            self.console.print(f"[yellow]History unavailable: {exc}[/yellow]")
            return
        if not recent:
            self.chat_id = uuid.uuid4().hex
            return
        item = recent[0]
        try:
            turns = self.history.messages(item["id"])
        except rc.RagError as exc:
            self.chat_id = uuid.uuid4().hex
            self.console.print(f"[yellow]Could not reopen the latest discussion: {exc}[/yellow]")
            return
        self.chat_id = item["id"]
        self.turns = turns
        self.last_answer = self._last_answer(turns)
        self.console.print(
            f"[dim]Continuing[/dim] {item['title']} "
            f"[dim]({item['messages']} messages). /new starts fresh.[/dim]"
        )

    @staticmethod
    def _last_answer(turns: list[dict[str, Any]]) -> rc.Answer | None:
        for item in reversed(turns):
            answer = item.get("answer")
            if answer is not None:
                return answer
        return None

    # -- commands ------------------------------------------------------------
    def _handle(self, line: str) -> bool:
        text = line.strip()
        if not text:
            return True
        if text.startswith("/"):
            return self._command(text)
        self._answer(text)
        return True

    def _command(self, text: str) -> bool:
        name, _, rest = text.partition(" ")
        name = name.lower()
        arg = rest.strip()
        if name != "/delete":
            self._confirm_delete = False
        if name in ("/quit", "/exit", "/q"):
            return False
        if name == "/help":
            self._help()
        elif name == "/local":
            self.local = True
            self.console.print("[dim]Local mode on: retrieval only, no model calls.[/dim]")
        elif name == "/remote":
            self.local = False
            self.console.print("[dim]Remote mode on: the Gemini agent will answer.[/dim]")
        elif name == "/new":
            self._new()
        elif name == "/history":
            self._list_history()
        elif name == "/open":
            self._open(arg)
        elif name == "/find":
            self._find(arg)
        elif name == "/sources":
            self._show_sources()
        elif name == "/trace":
            self._show_trace()
        elif name == "/title":
            self._title(arg)
        elif name == "/delete":
            self._delete()
        else:
            self.console.print(f"[yellow]Unknown command {name}. /help lists them.[/yellow]")
        return True

    def _help(self) -> None:
        rows = (
            ("/help", "this list"),
            ("/local", "retrieval only, no model, zero quota"),
            ("/remote", "use the Gemini agent again"),
            ("/new", "start a fresh discussion"),
            ("/history", "the ten most recent discussions"),
            ("/open N", "reopen discussion N from that list"),
            ("/find words", "search saved questions and answers"),
            ("/sources", "sources of the last answer"),
            ("/trace", "trace of the last answer"),
            ("/title [name]", "show or rename this discussion"),
            ("/delete", "delete this discussion, twice to confirm"),
            ("/quit", "leave the chat"),
        )
        for name, description in rows:
            self.console.print(f"  [cyan]{name:<14}[/cyan] {description}")

    def _new(self) -> None:
        self.chat_id = uuid.uuid4().hex
        self.turns = []
        self.last_answer = None
        self._confirm_delete = False
        self.console.print("[dim]New discussion. Ask something.[/dim]")

    def _list_history(self) -> None:
        if self.history is None:
            self.console.print("[yellow]History is off for this session.[/yellow]")
            return
        try:
            self._listed = self.history.list(limit=10)
        except rc.RagError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            return
        if not self._listed:
            self.console.print("[dim]No saved discussions yet.[/dim]")
            return
        for index, item in enumerate(self._listed, start=1):
            mark = "*" if item["id"] == self.chat_id else " "
            self.console.print(
                f" {mark} {index}. {item['title']} "
                f"[dim]({item['messages']} messages, {when(item['updated'])})[/dim]"
            )
        self.console.print("[dim]/open N reopens one.[/dim]")

    def _open(self, arg: str) -> None:
        if self.history is None:
            self.console.print("[yellow]History is off for this session.[/yellow]")
            return
        try:
            index = int(arg)
        except ValueError:
            self.console.print("[yellow]usage: /open N (see /history)[/yellow]")
            return
        if not self._listed:
            self._list_history()
        if not 1 <= index <= len(self._listed):
            self.console.print("[yellow]No discussion at that number. Try /history.[/yellow]")
            return
        item = self._listed[index - 1]
        try:
            turns = self.history.messages(item["id"])
        except rc.RagError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            return
        self.chat_id = item["id"]
        self.turns = turns
        self.last_answer = self._last_answer(turns)
        self._confirm_delete = False
        self.console.print(
            f"[bold]{item['title']}[/bold] [dim]({item['messages']} messages)[/dim]"
        )
        for turn in turns[-4:]:
            if turn["role"] == "user":
                self.console.print(f"[cyan]you>[/cyan] {turn['content']}")
            elif turn["role"] == "error":
                self.console.print(f"[red]error:[/red] {turn['content']}")
            else:
                answer = turn.get("answer")
                body = str(getattr(answer, "text", "") or "").strip()
                if body:
                    if len(body) > 600:
                        body = body[:600].rsplit(" ", 1)[0] + "..."
                    self.console.print(body)
        self.console.print("[dim]Follow-up context now comes from this discussion.[/dim]")

    def _find(self, arg: str) -> None:
        if not arg:
            self.console.print("[yellow]usage: /find words from a past chat[/yellow]")
            return
        if self.history is None:
            self.console.print("[yellow]History is off for this session.[/yellow]")
            return
        try:
            found = self.history.search(arg, limit=10)
        except rc.RagError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            return
        if not found:
            self.console.print("[dim]No discussion matches those words.[/dim]")
            return
        self._listed = found
        noun = "discussion" if len(found) == 1 else "discussions"
        self.console.print(f"[dim]{len(found)} {noun} match.[/dim]")
        for index, item in enumerate(found, start=1):
            self.console.print(
                f" {index}. {item['title']} "
                f"[dim]({item['matches']} matching messages)[/dim]"
            )
            snippet = str(item.get("snippet", "") or "")
            if snippet:
                self.console.print(f"    [dim]{snippet}[/dim]")
        self.console.print("[dim]/open N reopens one.[/dim]")

    def _show_sources(self) -> None:
        answer = self.last_answer
        if answer is None:
            self.console.print("[dim]No answer yet.[/dim]")
            return
        if answer.local:
            hits = answer.hits[:self.k]
            if not hits:
                self.console.print("[dim]No hits.[/dim]")
                return
            for index, hit in enumerate(hits, start=1):
                self.console.print(f"  {index}. {hit.source}  [dim]{hit.score:.3f}[/dim]")
            return
        if not answer.sources:
            self.console.print("[dim]This answer listed no sources.[/dim]")
            return
        for item in answer.sources:
            source = item.get("source", "unknown")
            score = float(item.get("relevance", 0.0) or 0.0)
            self.console.print(f"  {source}  [dim]{score:.3f}[/dim]")

    def _show_trace(self) -> None:
        answer = self.last_answer
        if answer is None:
            self.console.print("[dim]No answer yet.[/dim]")
            return
        self._print_steps(answer)

    def _title(self, arg: str) -> None:
        if self.history is None or not self.chat_id:
            self.console.print("[yellow]No saved discussion to title.[/yellow]")
            return
        try:
            current = self.history.title(self.chat_id)
        except rc.RagError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            return
        if not arg:
            self.console.print(f"Title: {current}")
            return
        label = " ".join(arg.split())
        try:
            self.history.rename(self.chat_id, label)
        except rc.RagError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            return
        self.console.print(f"[dim]Renamed to[/dim] {label}")

    def _delete(self) -> None:
        if self.history is None or not self.chat_id:
            self.console.print("[yellow]No saved discussion to delete.[/yellow]")
            return
        try:
            title = self.history.title(self.chat_id)
        except rc.RagError:
            title = "this discussion"
        if not self._confirm_delete:
            self._confirm_delete = True
            self.console.print(
                f"[yellow]Delete[/yellow] {title}? Type /delete again to confirm."
            )
            return
        try:
            removed = self.history.delete(self.chat_id)
        except rc.RagError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            self._confirm_delete = False
            return
        self.console.print(f"[dim]Deleted {removed} messages.[/dim]")
        self._new()

    # -- answering -----------------------------------------------------------
    def _context(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self.turns:
            role = item.get("role")
            if role == "user":
                out.append({"role": "user", "content": str(item.get("content", ""))})
            elif role == "assistant":
                answer = item.get("answer")
                if answer is not None:
                    out.append({"role": "assistant", "answer": answer})
        return out

    def _answer(self, question: str) -> None:
        context = self._context()
        self.turns.append({"role": "user", "content": question})
        self._save("user", question)
        try:
            if self.local:
                answer = self.agent.ask_local(
                    question, k=self.k, weight=self.weight, context=context
                )
            else:
                answer = self._ask_remote(question, context)
        except (rc.NoModel, rc.QuotaExhausted) as exc:
            self._failed(str(exc), "try /local, or add a key to .env")
            return
        except rc.RagError as exc:
            self._failed(str(exc), "")
            return
        except Exception as exc:
            self._failed(str(exc), "")
            return

        self.last_answer = answer
        self.turns.append({"role": "assistant", "answer": answer})
        self._save("assistant", answer.text or "", answer=answer)
        if self.local:
            self._print_local(answer)
        else:
            self._print_answer(answer)

    def _failed(self, message: str, hint: str) -> None:
        self.console.print(f"[red]error:[/red] {message}")
        if hint:
            self.console.print(f"[dim]{hint}[/dim]")
        self.turns.append({"role": "error", "content": message})
        self._save("error", message)

    def _ask_remote(self, question: str, context: list[dict[str, Any]]) -> rc.Answer:
        if not self.tty:
            return self.agent.ask(
                question, use_cache=self.use_cache, context=context
            )

        chunks: list[str] = []
        status = {"text": "searching"}
        live = Live(
            self._live_view(status["text"], ""),
            console=self.console,
            transient=True,
            refresh_per_second=12,
        )

        def view() -> None:
            live.update(self._live_view(status["text"], "".join(chunks)))

        def on_reset() -> None:
            chunks.clear()
            status["text"] = "searching"
            view()

        def on_token(token: str) -> None:
            chunks.append(token)
            status["text"] = "writing"
            view()

        def on_step(kind: str, text: str) -> None:
            status["text"] = f"{kind}: {text}"
            view()

        with live:
            answer = self.agent.ask(
                question,
                use_cache=self.use_cache,
                context=context,
                on_token=on_token,
                on_reset=on_reset,
                on_step=on_step,
            )
        return answer

    def _live_view(self, status: str, text: str) -> Any:
        return Text.assemble((status, "dim"), "\n\n", text or " ")

    def _save(self, role: str, content: str, answer: rc.Answer | None = None) -> None:
        if self.history is None or not self.chat_id:
            return
        try:
            self.history.append(self.chat_id, role, content, answer=answer)
        except rc.RagError as exc:
            self.console.print(f"[yellow]not saved: {exc}[/yellow]")

    # -- output --------------------------------------------------------------
    def _print_local(self, answer: rc.Answer) -> None:
        self.console.print()
        self.console.print("[dim]Local mode: no model was called.[/dim]")
        hits = answer.hits[:self.k]
        if not hits:
            self.console.print("[dim]No hits.[/dim]")
            return
        for index, hit in enumerate(hits, start=1):
            self.console.print(f"{index}. {hit.source}  [dim]{hit.score:.3f}[/dim]")
            self.console.print(f"   {hit.snippet(200)}")
        self.console.print()

    def _print_answer(self, answer: rc.Answer) -> None:
        self.console.print()
        if answer.text.strip():
            self.console.print(Markdown(answer.text.strip()))
        else:
            self.console.print("[dim]The model returned no text.[/dim]")
        if answer.sources:
            self.console.print()
            self.console.print("[bold]Sources[/bold]")
            for item in answer.sources:
                source = item.get("source", "unknown")
                score = float(item.get("relevance", 0.0) or 0.0)
                self.console.print(f"  {source}  [dim]{score:.3f}[/dim]")
        if self.show_steps:
            self.console.print()
            self._print_steps(answer)
        self.console.print()
        if answer.cached:
            self.console.print(
                f"[dim]cached, 0 API calls (answered by {answer.model})[/dim]"
            )
        elif answer.local:
            self.console.print("[dim]0 API calls (local mode)[/dim]")
        else:
            self.console.print(
                f"[dim]{answer.calls} API calls, {answer.elapsed:.1f} s, "
                f"{answer.model}[/dim]"
            )

    def _print_steps(self, answer: rc.Answer) -> None:
        if not answer.steps:
            self.console.print("[dim]No trace for this answer.[/dim]")
            return
        self.console.print("[bold]Trace[/bold]")
        for step in answer.steps:
            detail = f" {step.detail}" if step.detail else ""
            self.console.print(f"  {step.kind}: {step.label}{detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Oracle RAG corpus from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="*",
                        help="the question, quoted or as plain words")
    parser.add_argument("--chat", action="store_true",
                        help="interactive terminal chat with saved discussions")
    parser.add_argument("--no-history", action="store_true",
                        help="in chat mode, keep the session in memory only")
    parser.add_argument("--local", action="store_true",
                        help="search only, no model call, zero quota")
    parser.add_argument("--json", action="store_true", help="print the answer as JSON")
    parser.add_argument("--no-cache", action="store_true",
                        help="skip the answer cache and force a fresh model run")
    parser.add_argument("--k", type=int, default=4,
                        help="hits to show in local mode (default: 4)")
    parser.add_argument("--weight", type=float, default=0.6,
                        help="hybrid weight, 0 keyword to 1 vector (default: 0.6)")
    parser.add_argument("--steps", action="store_true", help="print the agent trace")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if args.chat and args.json:
        print("error: --json is for one-shot answers, not --chat")
        return 2
    if not args.chat and not question:
        print("error: ask something first")
        return 2

    k = max(1, args.k)
    settings = rc.Settings.from_env()
    engine = rc.RagEngine(settings)
    agent = rc.Agent(engine, settings)
    try:
        if args.chat:
            if Console is None:
                print("error: terminal chat needs the rich package: pip install rich")
                return 2
            history = None if args.no_history else rc.ChatHistory(engine)
            session = ChatSession(
                agent,
                history,
                local=args.local,
                k=k,
                weight=args.weight,
                show_steps=args.steps,
                use_cache=not args.no_cache,
            )
            return session.run(question)

        if args.local:
            answer = agent.ask_local(question, k=k, weight=args.weight)
        else:
            answer = agent.ask(question, use_cache=not args.no_cache)
    except (rc.NoModel, rc.QuotaExhausted) as exc:
        print(f"error: {exc}")
        print('try: python ask.py --local "your question"')
        return 3
    except rc.RagError as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 2
    finally:
        engine.close()

    if args.json:
        data = rc.answer_to_dict(answer)
        data["question"] = question
        data["cached"] = answer.cached
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.local:
        print_local(answer, k)
    else:
        print_remote(answer, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
