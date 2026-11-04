"""Headless render test for app.py.

    python tests/test_app.py

Uses streamlit.testing.AppTest, which executes the real script against the real
database. It spends zero Gemini quota: the chat is never submitted, only the
retrieval form.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def markdown_text(at: AppTest) -> str:
    return "\n".join(str(getattr(element, "value", "")) for element in at.markdown)


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
            check("three paths returned", bool(bundle.keyword) and bool(bundle.vector) and bool(bundle.hybrid),
                  f"kw {len(bundle.keyword)} vec {len(bundle.vector)} hyb {len(bundle.hybrid)}")
            check("timings present", set(bundle.timings) == {"keyword", "vector", "hybrid"},
                  str({k: round(v, 1) for k, v in bundle.timings.items()}))
        body = markdown_text(at)
        check("result rows rendered", 'class="row"' in body)
        check("latency rendered", " ms</div>" in body or "ms" in body)

    print("\n== other surfaces ==")
    for page, expected in (("Corpus", "Sources"), ("Quality", "labelled questions"), ("Health", "Container")):
        at.segmented_control[0].set_value(page).run()
        check(f"{page} renders without exception", not at.exception, str(at.exception))
        check(f"{page} content present", expected.lower() in markdown_text(at).lower(), expected)

    at.segmented_control[0].set_value("Health").run()
    health = markdown_text(at)
    check("health shows the answer cache", "Answer cache" in health and "entries" in health)
    check("sidebar shows the cache size", ">cache</span>" in health)
    check("health offers clearing the cache",
          any("Clear answer cache" in str(b.label) for b in at.button))

    print("\n" + "-" * 60)
    if failures:
        print(f"  {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  app renders and searches correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
