"""Saved answer artefacts.

An artefact is a file on disk that keeps one answer or one discussion outside
the app: Markdown to read, JSON to process again, HTML to open in a browser and
share as a static page. The writer and the readers live here so the Streamlit
app and the TUI produce exactly the same files.

Nothing in this module talks to Oracle or to a model. It takes the `Answer` and
`ChatHistory` objects from `rag_core` and turns them into text.

    artifacts/
      20261110-142530-which-companies-raise-a-series-b-a1b2c3.md
      20261110-142530-which-companies-raise-a-series-b-a1b2c3.json
      20261110-142530-which-companies-raise-a-series-b-a1b2c3.html

The directory can be moved with `RAG_ARTIFACTS_DIR`. Files are never written
outside that directory, and deletes only accept paths that live inside it.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import rag_core as rc

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "artifacts"
FORMAT = "oracle-rag-artefact"
VERSION = 1
SUFFIX_KIND = {".md": "markdown", ".json": "json", ".html": "html", ".txt": "text"}
KIND_SUFFIX = {"markdown": ".md", "json": ".json", "html": ".html", "text": ".txt"}
DEFAULT_FORMATS = ("markdown", "json", "html")


# --------------------------------------------------------------------------- #
# value type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Artefact:
    """One file in the artefacts directory."""

    path: Path
    kind: str
    title: str
    created: str
    size: int

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def stamp(self) -> dt.datetime | None:
        try:
            return dt.datetime.fromisoformat(self.created)
        except (TypeError, ValueError):
            return None

    def preview(self, limit: int = 12_000) -> str:
        """Text for a UI preview. Large files are clipped, not loaded twice."""
        try:
            body = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Could not read this artefact: {exc}"
        if len(body) <= limit:
            return body
        return body[:limit].rstrip() + "\n\n… preview clipped. Download the file to read all of it."


# --------------------------------------------------------------------------- #
# paths and names
# --------------------------------------------------------------------------- #
def directory(path: str | Path | None = None) -> Path:
    """The artefacts root, created if needed."""
    chosen = path or os.getenv("RAG_ARTIFACTS_DIR") or DEFAULT_DIR
    root = Path(chosen).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _resolve(item: Artefact | str | Path,
             root: str | Path | None = None) -> Path:
    base = directory(root)
    raw = item.path if isinstance(item, Artefact) else Path(item)
    path = raw if raw.is_absolute() else base / raw
    path = path.resolve()
    if not _inside(path, base):
        raise ValueError("that artefact is outside the artefacts directory")
    return path


def slug(text: object, limit: int = 48) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    if not clean:
        return "answer"
    return clean[:limit].rstrip("-") or "answer"


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _unique(root: Path, title: object, suffix: str) -> Path:
    stem = f"{_stamp()}-{slug(title)}-{uuid.uuid4().hex[:6]}"
    candidate = root / f"{stem}{suffix}"
    while candidate.exists():
        candidate = root / f"{_stamp()}-{slug(title)}-{uuid.uuid4().hex[:6]}{suffix}"
    return candidate


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _artefact(path: Path) -> Artefact:
    suffix = path.suffix.lower()
    kind = SUFFIX_KIND.get(suffix, "text")
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return Artefact(
        path=path,
        kind=kind,
        title=_title(path, kind),
        created=_created(path, kind),
        size=size,
    )


def _title(path: Path, kind: str) -> str:
    try:
        if kind == "markdown":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("# "):
                    return line[2:].strip() or path.stem
        elif kind == "json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            title = str(data.get("title") or data.get("question") or "").strip()
            if title:
                return title
        elif kind == "html":
            body = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
            if match:
                return html.unescape(" ".join(match.group(1).split()))
    except (OSError, ValueError, TypeError):
        pass
    return path.stem


def _created(path: Path, kind: str) -> str:
    if kind == "json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            stamp = str(data.get("generated_at") or "").strip()
            if stamp:
                return stamp
        except (OSError, ValueError, TypeError):
            pass
    try:
        when = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        when = dt.datetime.now(dt.timezone.utc)
    return when.isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# answer payloads
# --------------------------------------------------------------------------- #
def _source_rows(answer: rc.Answer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in answer.sources or []:
        if isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append({"source": str(item)})
    return rows


def answer_payload(question: str, answer: rc.Answer, *,
                   title: str = "", chat_id: str = "") -> dict[str, Any]:
    data = rc.answer_to_dict(answer)
    data["cached"] = bool(answer.cached)
    return {
        "format": FORMAT,
        "version": VERSION,
        "kind": "answer",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": _clean_title(title or question),
        "question": str(question or ""),
        "chat_id": str(chat_id or "") or None,
        "answer": data,
    }


def _clean_title(value: object) -> str:
    clean = " ".join(str(value or "").split())
    return clean[:120] or "Untitled answer"


def answer_markdown(question: str, answer: rc.Answer, *,
                    title: str = "", chat_id: str = "") -> str:
    label = _clean_title(title or question)
    created = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    lines = [f"# {label}", "", f"_Saved from Oracle RAG Console on {created}._", ""]

    if question:
        lines += ["## Question", "", str(question).strip(), ""]

    if answer.local:
        lines += ["## Local results", ""]
        for index, hit in enumerate(answer.hits, start=1):
            published = f" · {hit.published}" if hit.published else ""
            lines.append(f"{index}. **{hit.source}**{published} · score {hit.score:.3f}")
            lines.append(f"   {hit.snippet(400)}")
            lines.append("")
    else:
        text = str(answer.text or "").strip()
        lines += ["## Answer", "", text or "_The model returned no text._", ""]

    sources = _source_rows(answer)
    if sources:
        lines += ["## Sources", ""]
        for index, item in enumerate(sources, start=1):
            name = str(item.get("source") or "unknown")
            score = float(item.get("relevance", 0.0) or 0.0)
            published = str(item.get("published") or "").strip()
            numbers = item.get("citations")
            if not isinstance(numbers, (list, tuple)) or not numbers:
                numbers = [item["citation"]] if item.get("citation") else []
            prefix = f"[{', '.join(str(n) for n in numbers)}] " if numbers else ""
            meta = [f"relevance {score:.3f}"]
            if published:
                meta.append(f"published {published}")
            lines.append(f"- {index}. {prefix}**{name}** · " + " · ".join(meta))
            snippet = str(item.get("snippet") or "").strip()
            if snippet:
                lines.append(f"  > {' '.join(snippet.split())}")
        lines.append("")

    if answer.steps:
        lines += ["## Trace", ""]
        for step in answer.steps:
            detail = f" - {step.detail}" if step.detail else ""
            lines.append(f"- `{step.kind}` **{step.label}**{detail}")
        lines.append("")

    if answer.cached:
        run = f"cached answer by {answer.model}; no model call"
    elif answer.local:
        run = "local mode; no model call"
    else:
        run = f"{answer.calls} API calls in {answer.elapsed:.1f}s with {answer.model}"
    lines += ["## Run", "", f"- {run}", "- quota used: 0" if (answer.local or answer.cached)
              else f"- quota used: {answer.calls} API calls"]
    if chat_id:
        lines.append(f"- discussion: `{chat_id}`")
    return "\n".join(lines).rstrip() + "\n"


def _paragraphs(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", str(text or "")) if block.strip()]
    if not blocks:
        return "<p><em>The model returned no text.</em></p>"
    out = []
    for block in blocks:
        safe = html.escape(block).replace("\n", "<br>\n")
        out.append(f"<p>{safe}</p>")
    return "\n".join(out)


def answer_html(question: str, answer: rc.Answer, *,
                title: str = "", chat_id: str = "") -> str:
    label = _clean_title(title or question)
    created = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()
    sources = _source_rows(answer)

    if answer.local:
        body = ['<h2>Local results</h2>', '<ol class="hits">']
        for hit in answer.hits:
            published = f" · {html.escape(hit.published)}" if hit.published else ""
            body.append(
                f'<li><strong>{html.escape(hit.source)}</strong>{published}'
                f'<span class="score">{hit.score:.3f}</span>'
                f'<p>{html.escape(hit.snippet(400))}</p></li>'
            )
        body.append("</ol>")
    else:
        body = ["<h2>Answer</h2>", _paragraphs(answer.text)]

    if sources:
        body.append("<h2>Sources</h2>")
        body.append('<ol class="sources">')
        for item in sources:
            name = html.escape(str(item.get("source") or "unknown"))
            score = float(item.get("relevance", 0.0) or 0.0)
            published = str(item.get("published") or "").strip()
            numbers = item.get("citations")
            if not isinstance(numbers, (list, tuple)) or not numbers:
                numbers = [item["citation"]] if item.get("citation") else []
            prefix = ""
            if numbers:
                prefix = f'<span class="cite">[{html.escape(", ".join(str(n) for n in numbers))}]</span> '
            meta = f'<span class="score">{score:.3f}</span>'
            if published:
                meta += f' <span class="date">{html.escape(published)}</span>'
            snippet = str(item.get("snippet") or "").strip()
            snippet_html = f"<p>{html.escape(' '.join(snippet.split()))}</p>" if snippet else ""
            body.append(f"<li>{prefix}<strong>{name}</strong> {meta}{snippet_html}</li>")
        body.append("</ol>")

    if answer.steps:
        body.append("<h2>Trace</h2>")
        body.append('<ol class="trace">')
        for step in answer.steps:
            detail = f' <span class="detail">{html.escape(step.detail)}</span>' if step.detail else ""
            body.append(
                f'<li><code>{html.escape(step.kind)}</code> '
                f'<strong>{html.escape(step.label)}</strong>{detail}</li>'
            )
        body.append("</ol>")

    if answer.cached:
        run = f"cached answer by {html.escape(answer.model)}; no model call"
    elif answer.local:
        run = "local mode; no model call"
    else:
        run = (f"{answer.calls} API calls in {answer.elapsed:.1f}s "
               f"with {html.escape(answer.model)}")
    question_html = (f"<h2>Question</h2>{_paragraphs(question)}" if question else "")
    chat_html = f'<p class="meta">Discussion <code>{html.escape(chat_id)}</code></p>' if chat_id else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(label)}</title>
<style>
:root{{--ink:#1b1816;--ink2:#6b615a;--ink3:#948a82;--line:#e4dbd4;
--paper:#faf7f4;--surface:#fffdfb;--accent:#b23a2b;--mark:#f6dfc9;}}
@media (prefers-color-scheme:dark){{:root{{--ink:#f2eae3;--ink2:#b8aca3;
--ink3:#8c8179;--line:#3a322d;--paper:#171412;--surface:#1e1a17;
--accent:#e06a55;--mark:#5a3a22;}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.65 Inter,"Segoe UI",system-ui,sans-serif}}
main{{max-width:52rem;margin:0 auto;padding:3rem 1.4rem 5rem}}
h1{{font-size:1.7rem;letter-spacing:-.02em;margin:0 0 .3rem}}
h2{{font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;
color:var(--ink3);margin:2rem 0 .5rem;font-family:ui-monospace,Consolas,monospace}}
.meta,.date{{color:var(--ink3);font-size:.78rem}}
header{{border-bottom:1px solid var(--line);padding-bottom:1rem;margin-bottom:1rem}}
.card,li{{background:var(--surface);border:1px solid var(--line);border-radius:4px}}
article{{background:var(--surface);border:1px solid var(--line);border-radius:5px;
padding:1.4rem 1.5rem}}
p{{margin:.45rem 0}}
ol{{padding-left:1.4rem}}
li{{margin:.5rem 0;padding:.6rem .75rem}}
li p{{margin:.35rem 0 0;color:var(--ink2)}}
.score{{font-family:ui-monospace,Consolas,monospace;font-size:.78rem;
color:var(--accent);margin-left:.4rem}}
.cite{{font-family:ui-monospace,Consolas,monospace;font-size:.78rem;
color:var(--accent)}}
.detail{{color:var(--ink2)}}
code{{font-family:ui-monospace,Consolas,monospace;font-size:.82em;
background:var(--mark);padding:.05rem .28rem;border-radius:2px}}
footer{{margin-top:2.5rem;border-top:1px solid var(--line);padding-top:.8rem;
color:var(--ink3);font-size:.75rem}}
</style>
</head>
<body>
<main>
<header>
<h1>{html.escape(label)}</h1>
<p class="meta">Saved from Oracle RAG Console on {html.escape(created)}</p>
{chat_html}
</header>
{question_html}
<article>
{''.join(body)}
</article>
<footer>{html.escape(run)}</footer>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def save_answer(question: str, answer: rc.Answer, *,
                title: str = "", chat_id: str = "",
                formats: Sequence[str] = DEFAULT_FORMATS,
                directory_path: str | Path | None = None) -> list[Artefact]:
    """Write one answer as Markdown, JSON and/or HTML. Returns the new files."""
    root = directory(directory_path)
    label = _clean_title(title or question)
    payload = answer_payload(question, answer, title=label, chat_id=chat_id)
    writers = {
        "markdown": lambda: answer_markdown(
            question, answer, title=label, chat_id=chat_id),
        "json": lambda: json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        "html": lambda: answer_html(question, answer, title=label, chat_id=chat_id),
    }
    out: list[Artefact] = []
    for kind in formats:
        key = str(kind).lower()
        if key not in writers:
            raise ValueError(f"unknown artefact format: {kind}")
        path = _unique(root, label, KIND_SUFFIX[key])
        _write(path, writers[key]())
        out.append(_artefact(path))
    return out


def save_text(title: object, body: str, *, suffix: str = "md",
              directory_path: str | Path | None = None) -> Artefact:
    """Write arbitrary Markdown or text under the artefacts directory."""
    root = directory(directory_path)
    ext = suffix if suffix.startswith(".") else "." + suffix
    if ext.lower() not in SUFFIX_KIND:
        raise ValueError(f"unsupported artefact suffix: {suffix}")
    path = _unique(root, title, ext.lower())
    _write(path, str(body))
    return _artefact(path)


def discussion_payload(history: rc.ChatHistory, chat_id: str) -> dict[str, Any]:
    title = history.title(chat_id)
    messages: list[dict[str, Any]] = []
    for item in history.messages(chat_id):
        row: dict[str, Any] = {"role": item.get("role")}
        if item.get("at"):
            row["at"] = str(item["at"])
        if item.get("content") is not None:
            row["content"] = str(item.get("content") or "")
        answer = item.get("answer")
        if answer is not None:
            data = rc.answer_to_dict(answer)
            data["cached"] = bool(getattr(answer, "cached", False))
            row["answer_markdown"] = str(getattr(answer, "text", "") or "")
            row["answer"] = data
        messages.append(row)
    return {
        "format": FORMAT,
        "version": VERSION,
        "kind": "discussion",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": title,
        "chat_id": str(chat_id),
        "messages": messages,
    }


def save_discussion(history: rc.ChatHistory, chat_id: str, *,
                    formats: Sequence[str] = ("markdown", "json"),
                    directory_path: str | Path | None = None) -> list[Artefact]:
    """Write a whole saved discussion as Markdown and/or JSON."""
    root = directory(directory_path)
    title = history.title(chat_id)
    payload = discussion_payload(history, chat_id)
    out: list[Artefact] = []
    for kind in formats:
        key = str(kind).lower()
        if key == "markdown":
            body = history.export_markdown(chat_id)
        elif key == "json":
            body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        else:
            raise ValueError(f"unsupported discussion format: {kind}")
        path = _unique(root, title, KIND_SUFFIX[key])
        _write(path, body)
        out.append(_artefact(path))
    return out


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def list_artefacts(directory_path: str | Path | None = None,
                   limit: int | None = None) -> list[Artefact]:
    """Every readable artefact, newest first."""
    root = directory(directory_path)
    found = [
        _artefact(path)
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIX_KIND
    ]
    found.sort(key=lambda item: item.path.stat().st_mtime, reverse=True)
    return found[:limit] if limit else found


def read_artefact(item: Artefact | str | Path,
                  root: str | Path | None = None) -> str:
    path = _resolve(item, root)
    return path.read_text(encoding="utf-8", errors="replace")


def delete_artefact(item: Artefact | str | Path,
                    root: str | Path | None = None) -> bool:
    path = _resolve(item, root)
    if not path.is_file():
        return False
    path.unlink()
    return True


def stats(directory_path: str | Path | None = None) -> dict[str, Any]:
    items = list_artefacts(directory_path)
    return {
        "count": len(items),
        "bytes": sum(item.size for item in items),
        "latest": items[0].created if items else "",
        "directory": str(directory(directory_path)),
        "kinds": {
            kind: sum(1 for item in items if item.kind == kind)
            for kind in ("markdown", "json", "html", "text")
        },
    }


def human_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def summary(item: Artefact) -> str:
    stamp = item.stamp
    when = stamp.astimezone().strftime("%b %d, %H:%M") if stamp else "unknown date"
    return f"{item.title} · {item.kind} · {human_size(item.size)} · {when}"
