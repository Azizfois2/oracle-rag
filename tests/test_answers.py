"""Zero-quota tests for the answer-quality gate.

    python tests/test_answers.py

Every answer here is fake, so no Gemini call is made and no quota is spent.
The live check lives in quality/check_answers.py and is opt-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_core as rc  # noqa: E402
from quality.answer_eval import (answer_sources, is_refusal, load_cases,  # noqa: E402
                                 norm, score_all, score_case)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def good_answer(case: dict, corpus: list[str]) -> rc.Answer:
    kind = case.get("type", "single")
    if kind == "unanswerable":
        return rc.Answer(text="The corpus does not contain that information.",
                         model="fake", sources=[])
    facts = case.get("expect_facts") or []
    text = "According to the corpus: " + ", ".join(str(f) for f in facts) + "."
    sources = [{"source": src, "relevance": 1.0}
               for src in (case.get("expect_sources") or [])]
    return rc.Answer(text=text, model="fake", sources=sources)


def main() -> int:
    corpus = [
        "01_nvidia_ai_infrastructure.md",
        "02_databricks_lakehouse_ai.md",
        "03_snowflake_cortex_agents.md",
        "04_google_vertex_ai_update.md",
        "05_microsoft_azure_ai_foundry.md",
        "06_aws_bedrock_agents.md",
        "07_startup_vector_database.md",
        "08_langchain_ecosystem_growth.md",
        "09_enterprise_ai_adoption.md",
        "10_open_source_ai_models.md",
    ]

    print("\n== labelled cases ==")
    cases = load_cases()
    check("cases load", len(cases) >= 20, str(len(cases)))
    kinds = {case.get("type") for case in cases}
    check("all three case types exist", kinds == {"single", "cross", "unanswerable"},
          str(kinds))
    check("questions are unique",
          len({case.get("question") for case in cases}) == len(cases))
    check("every expected source is in the corpus",
          all(src in corpus for case in cases
              for src in (case.get("expect_sources") or [])),
          str([src for case in cases for src in (case.get("expect_sources") or [])
               if src not in corpus]))
    check("answerable cases ask for a fact",
          all(case.get("expect_facts") for case in cases
              if case.get("type") != "unanswerable"))
    check("unanswerable cases ask for no source",
          all(not case.get("expect_sources") for case in cases
              if case.get("type") == "unanswerable"))

    print("\n== perfect run ==")
    answers = [good_answer(case, corpus) for case in cases]
    summary = score_all(cases, answers, corpus)
    check("every good answer passes", summary["passed"] == summary["total"],
          f'{summary["passed"]}/{summary["total"]}')
    check("score is 1.0", summary["score"] == 1.0, str(summary["score"]))
    check("all checks are recorded",
          set(summary["checks"]) == {"source", "facts", "refusal", "citations"},
          str(set(summary["checks"])))
    check("by type covers all three",
          set(summary["by_type"]) == {"single", "cross", "unanswerable"},
          str(set(summary["by_type"])))

    print("\n== failure modes ==")
    first = next(case for case in cases if case.get("type") == "single")
    missing_fact = rc.Answer(
        text="The corpus mentions something else.",
        model="fake",
        sources=[{"source": first["expect_sources"][0], "relevance": 1.0}],
    )
    failed = score_case(first, missing_fact, corpus)
    check("missing fact fails", not failed.passed)
    check("missing fact names the fact",
          any(check.name == "facts" and not check.passed for check in failed.checks))

    missing_source = rc.Answer(
        text="According to the corpus: " + ", ".join(first["expect_facts"]) + ".",
        model="fake",
        sources=[],
    )
    failed = score_case(first, missing_source, corpus)
    check("missing source fails", not failed.passed)
    check("missing source names the check",
          any(check.name == "source" and not check.passed for check in failed.checks))

    invented = rc.Answer(
        text="According to the corpus: " + ", ".join(first["expect_facts"]) + ".",
        model="fake",
        sources=[{"source": first["expect_sources"][0], "relevance": 1.0},
                 {"source": "invented_article.md", "relevance": 0.9}],
    )
    failed = score_case(first, invented, corpus)
    check("invented citation fails", not failed.passed)
    detail = next((check.detail for check in failed.checks
                   if check.name == "citations"), "")
    check("invented citation is named", "invented_article.md" in detail, detail)

    unanswerable = next(case for case in cases
                        if case.get("type") == "unanswerable")
    answered = rc.Answer(text="NVIDIA revenue was 5 billion.", model="fake", sources=[])
    failed = score_case(unanswerable, answered, corpus)
    check("unanswerable question must be refused", not failed.passed)
    check("refusal check fails when answered",
          any(check.name == "refusal" and not check.passed for check in failed.checks))

    refused = rc.Answer(text="I don't know. That is not in the corpus.",
                        model="fake", sources=[])
    passed = score_case(unanswerable, refused, corpus)
    check("a refusal passes", passed.passed, str(passed.checks))

    print("\n== text handling ==")
    check("comma in a number is ignored", norm("1,000 sessions") == norm("1000 sessions"))
    check("percent sign equals the word",
          norm("72% of firms") == norm("72 percent of firms"))
    check("hyphen and space compare equally",
          norm("14-billion-parameter") == norm("14 billion parameter"))
    check("refusal catches a missing corpus phrase",
          is_refusal("The answer is not in the corpus.", []))
    check("refusal does not catch a real answer",
          not is_refusal("VectorForge raised 85 million dollars.", ["07.md"]))

    cited = rc.Answer(
        text="See [Source: 07_startup_vector_database.md | Relevance: 0.9] for detail.",
        model="fake", sources=[])
    check("inline source blocks are read",
          answer_sources(cited) == ["07_startup_vector_database.md"],
          str(answer_sources(cited)))

    print("\n== summary math ==")
    broken = list(answers)
    broken[0] = rc.Answer(text="nothing useful", model="fake", sources=[])
    summary = score_all(cases, broken, corpus)
    check("one failure lowers the count",
          summary["passed"] == summary["total"] - 1,
          f'{summary["passed"]}/{summary["total"]}')

    print("\n" + "-" * 60)
    if failures:
        print(f"  {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  answer gate scores correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
