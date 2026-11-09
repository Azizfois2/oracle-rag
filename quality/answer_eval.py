"""Score answers against labelled cases, with no model call.

The gate checks four things:

- every expected source is cited
- every expected fact appears in the answer text
- an unanswerable question is refused
- every citation names a document and a number the tool returned

`quality/check_answers.py` runs this against live answers. `tests/test_answers.py`
runs it against fake answers, so the scoring itself costs zero quota.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_CASES = Path(__file__).resolve().parent / "answers.json"

_REFUSAL = re.compile(
    r"(not (?:in|found in) the corpus|no (?:relevant )?information|"
    r"(?:do not|don't) have (?:enough )?information|not enough information|"
    r"cannot (?:find|answer)|can't (?:find|answer)|unable to (?:find|answer)|"
    r"does not (?:contain|mention|include)|doesn't (?:contain|mention|include)|"
    r"outside the corpus|not covered by the corpus|i (?:do not|don't) know)",
    re.IGNORECASE,
)

_CITATION = re.compile(r"\[Source:\s*([^|\]\n]+)")
_NUMBER = re.compile(r"\[(\d{1,2})\]")


def norm(text: object) -> str:
    """Lowercase and flatten punctuation so numbers and names compare fairly."""
    body = str(text or "").lower()
    body = body.replace("%", " percent ").replace("$", " ").replace(",", "")
    body = body.replace("-", " ").replace("_", " ")
    body = re.sub(r"[^\w\s.]", " ", body)
    return " ".join(body.split())


def _same_source(left: str, right: str) -> bool:
    a = " ".join(str(left or "").lower().split())
    b = " ".join(str(right or "").lower().split())
    if not a or not b:
        return False
    return a == b or a in b or b in a


def answer_text(answer: Any) -> str:
    return str(getattr(answer, "text", "") or "")


def answer_sources(answer: Any) -> list[str]:
    """Every source name the answer cites, from the tool and from the text."""
    names: list[str] = []
    for item in getattr(answer, "sources", None) or []:
        if isinstance(item, dict):
            name = str(item.get("source") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    for match in _CITATION.finditer(answer_text(answer)):
        name = match.group(1).strip()
        if name:
            names.append(name)
    out: list[str] = []
    for name in names:
        if not any(_same_source(name, seen) for seen in out):
            out.append(name)
    return out


def is_refusal(text: str, sources: Sequence[str] | None = None) -> bool:
    body = str(text or "").strip()
    if _REFUSAL.search(body):
        return True
    return not body and not sources


def _cited(answer: Any, expected: str) -> bool:
    return any(_same_source(name, expected) for name in answer_sources(answer))


def _invented(answer: Any, corpus_sources: Iterable[str]) -> list[str]:
    known = [str(name) for name in corpus_sources]
    return [name for name in answer_sources(answer)
            if not any(_same_source(name, source) for source in known)]


def known_numbers(answer: Any) -> set[int]:
    """Citation numbers the tool actually returned.

    New answers store them on each source. Older answers have none, so the
    source list position is used, which matches the old tool order.
    """
    known: set[int] = set()
    for position, item in enumerate(getattr(answer, "sources", None) or [], start=1):
        if not isinstance(item, dict):
            known.add(position)
            continue
        stored = item.get("citations")
        found = False
        if isinstance(stored, (list, tuple)):
            for value in stored:
                try:
                    known.add(int(value))
                    found = True
                except (TypeError, ValueError):
                    continue
        if item.get("citation") is not None:
            try:
                known.add(int(item["citation"]))
                found = True
            except (TypeError, ValueError):
                pass
        if not found:
            known.add(position)
    return known


def invented_numbers(answer: Any) -> list[int]:
    """Numbers cited in the text that the tool never returned."""
    known = known_numbers(answer)
    return sorted({int(match.group(1)) for match in _NUMBER.finditer(answer_text(answer))
                   if int(match.group(1)) not in known})


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class CaseScore:
    question: str
    kind: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "type": self.kind,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "answer": self.answer,
            "sources": list(self.sources),
        }


def score_case(case: dict[str, Any], answer: Any,
               corpus_sources: Iterable[str]) -> CaseScore:
    kind = str(case.get("type") or "single")
    expected_sources = [str(s) for s in (case.get("expect_sources") or []) if s]
    facts = [str(f) for f in (case.get("expect_facts") or []) if f]
    text = answer_text(answer)
    flat = norm(text)
    sources = answer_sources(answer)
    checks: list[Check] = []

    if expected_sources:
        missing = [src for src in expected_sources if not _cited(answer, src)]
        checks.append(Check(
            "source", not missing,
            "missing: " + ", ".join(missing) if missing else "all expected sources cited",
        ))

    if facts:
        missing = [fact for fact in facts if norm(fact) not in flat]
        checks.append(Check(
            "facts", not missing,
            "missing: " + ", ".join(missing) if missing else "all expected facts present",
        ))

    if kind == "unanswerable":
        refused = is_refusal(text, sources)
        checks.append(Check(
            "refusal", refused,
            "refused" if refused else "answered as if the corpus contained it",
        ))

    invented = _invented(answer, corpus_sources)
    bad_numbers = invented_numbers(answer)
    details: list[str] = []
    if invented:
        details.append("invented sources: " + ", ".join(invented))
    if bad_numbers:
        details.append("invented numbers: "
                       + ", ".join(f"[{number}]" for number in bad_numbers))
    checks.append(Check(
        "citations", not details,
        "; ".join(details) if details else "every citation is real",
    ))

    return CaseScore(
        question=str(case.get("question") or ""),
        kind=kind,
        passed=all(check.passed for check in checks),
        checks=checks,
        answer=text,
        sources=sources,
    )


def score_all(cases: Sequence[dict[str, Any]], answers: Sequence[Any],
              corpus_sources: Iterable[str]) -> dict[str, Any]:
    if len(cases) != len(answers):
        raise ValueError(f"{len(cases)} cases but {len(answers)} answers")
    known = list(corpus_sources)
    results = [score_case(case, answer, known) for case, answer in zip(cases, answers)]

    totals: dict[str, dict[str, int]] = {}
    for result in results:
        for check in result.checks:
            bucket = totals.setdefault(check.name, {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(check.passed)

    by_type: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_type.setdefault(result.kind, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)

    passed = sum(1 for result in results if result.passed)
    return {
        "total": len(results),
        "passed": passed,
        "score": round(passed / len(results), 3) if results else 0.0,
        "checks": totals,
        "by_type": by_type,
        "cases": [result.as_dict() for result in results],
    }


def load_cases(path: Path | str | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else DEFAULT_CASES
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{target} must contain a JSON list")
    return [item for item in data if isinstance(item, dict)]
