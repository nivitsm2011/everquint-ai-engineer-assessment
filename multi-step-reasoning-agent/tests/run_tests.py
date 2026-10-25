"""Run test cases through the agent and write a report.

Usage:
  python tests/run_tests.py                 run all 13 cases
  python tests/run_tests.py --ids 3 4 5     run only these cases (results are merged
                                            into the existing report, so a run can be
                                            split over several batches or models)

Mock provider -> logs/example_runs_mock.md + logs/test_results_mock.jsonl
Real provider -> logs/example_runs.md      + logs/test_results_live.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Let this file run as a plain script from any folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import describe_provider  # noqa: E402
from reasoner.llm_client import get_llm_client  # noqa: E402
from reasoner.orchestrator import TRACE_PATH, ReasoningAgent  # noqa: E402
from reasoner.verifier import normalize_answer  # noqa: E402
from tests.test_cases import TEST_CASES  # noqa: E402

LOGS = ROOT / "logs"

MOCK_HEADER = """\
# Example runs - MOCK PROVIDER (canned responses)

> These results come from the mock LLM, whose answers are canned. They verify the pipeline
> (planner, executor, tools, verifier, retry loop), not the accuracy of a real model.
> Real-model results are in [`example_runs.md`](example_runs.md).
"""

LIVE_HEADER = """\
# Example runs - REAL MODEL

> Provider: Google Gemini, free tier. Each free model allows only 20 requests per day and one
> question needs 3 or more calls, so the 13 questions were run in batches on different models.
> The **Model** column shows which model produced each result. "NOT EVALUATED" means the API
> rejected every call (no model answer exists), so it is not a model failure.
>
> The full 13-case run on the mock provider is in [`example_runs_mock.md`](example_runs_mock.md).
> The mock is canned, so it proves the pipeline logic, not the model's accuracy.
"""


def last_trace() -> dict:
    """The newest line of the trace log (the case that just ran)."""
    lines = TRACE_PATH.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def run_case(case: dict, llm, model: str) -> dict:
    """Run one case and compare it with what we expect."""
    result = ReasoningAgent(llm).solve(case["question"])  # resets the mock's counters
    attempts = last_trace()["attempts"]
    canonical = str((attempts[-1]["execution"] or {}).get("canonical_answer", ""))
    # If every attempt died with an error (e.g. HTTP 429) there is no model answer to judge.
    evaluated = not all(a["error"] for a in attempts)
    passed = evaluated and (
        result["status"] == case["expected_status"]
        and normalize_answer(canonical) == normalize_answer(case["expected_canonical"])
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_canonical": case["expected_canonical"],
        "expected_status": case["expected_status"],
        "model": model,
        "canonical_answer": canonical,
        "verifier_passed": all(c["passed"] for c in result["metadata"]["checks"]),
        "retries": result["metadata"]["retries"],
        "evaluated": evaluated,
        "test_passed": passed,
        "result": result,
    }


def state_of(row: dict | None) -> str:
    if row is None:
        return "NOT RUN"
    if not row["evaluated"]:
        return "NOT EVALUATED (API error)"
    return "PASS" if row["test_passed"] else "FAIL"


def markdown_report(rows: dict[int, dict], header: str) -> str:
    """Readable report: summary table, then the details of every case that ran."""
    done = [r for r in rows.values() if r["evaluated"]]
    passed = sum(r["test_passed"] for r in done)
    retries = sum(r["retries"] for r in done)
    not_eval = sum(not r["evaluated"] for r in rows.values())
    not_run = len(TEST_CASES) - len(rows)
    summary = f"{passed}/{len(done)} evaluated cases passed, {retries} retries triggered"
    if not_eval or not_run:
        summary += f" ({not_eval} not evaluated, {not_run} not run)"

    parts = [header, f"**{summary}**", "", "| # | Category | Model | Status | Retries |", "|---|---|---|---|---|"]
    for case in TEST_CASES:
        row = rows.get(case["id"])
        parts.append(
            f"| Q{case['id']} | {case['category']} | {row['model'] if row else '-'} | "
            f"{state_of(row)} | {row['retries'] if row and row['evaluated'] else '-'} |"
        )
    for case in TEST_CASES:
        row = rows.get(case["id"])
        if row is None or not row["evaluated"]:
            continue
        parts += [
            "",
            f"## Q{row['id']} ({row['category']}): {state_of(row)}",
            "",
            f"**Question:** {row['question']}",
            "",
            f"- Model: {row['model']}",
            f"- Expected: `{row['expected_canonical']}` ({row['expected_status']})",
            f"- Got: `{row['canonical_answer']}` ({row['result']['status']})",
            f"- Verifier passed: {row['verifier_passed']}",
            f"- Retries used: {row['retries']}",
            "",
            "```json",
            json.dumps(row["result"], indent=2),
            "```",
        ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on the test cases")
    parser.add_argument("--ids", type=int, nargs="*", help="only run these case ids")
    parser.add_argument(
        "--report-only", action="store_true", help="rebuild the report from saved results, run nothing"
    )
    args = parser.parse_args()

    llm = get_llm_client()
    model = describe_provider(llm)
    is_mock = model == "mock"
    results_path = LOGS / ("test_results_mock.jsonl" if is_mock else "test_results_live.jsonl")
    report_path = LOGS / ("example_runs_mock.md" if is_mock else "example_runs.md")
    print(f"Provider: {model}", flush=True)

    # Rows from earlier batches, so this run adds to them instead of replacing them.
    rows: dict[int, dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "evaluated" in row:
                rows[row["id"]] = row

    for case in TEST_CASES:
        if args.report_only or (args.ids and case["id"] not in args.ids):
            continue
        row = run_case(case, llm, model)
        rows[row["id"]] = row
        print(f"Q{row['id']} [{row['category']}] {row['question']}")
        print(json.dumps(row["result"], indent=2))
        print(
            f"  verifier passed: {row['verifier_passed']} | retries: {row['retries']} | "
            f"expected {row['expected_canonical']!r} got {row['canonical_answer']!r} -> "
            f"{state_of(row)}\n",
            flush=True,
        )
        # Save after every case so a crash or quota stop loses nothing.
        LOGS.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            "".join(json.dumps(rows[i]) + "\n" for i in sorted(rows)), encoding="utf-8"
        )
    # Rewrite the report once more so --report-only (which runs no cases) still rebuilds it.
    header = MOCK_HEADER if is_mock else LIVE_HEADER
    notes = LOGS / "live_notes.md"  # hand-written run history, shown in the live report
    if not is_mock and notes.exists():
        header += "\n" + notes.read_text(encoding="utf-8")
    report_path.write_text(markdown_report(rows, header), encoding="utf-8")

    done = [r for r in rows.values() if r["evaluated"]]
    print(
        f"{sum(r['test_passed'] for r in done)}/{len(done)} evaluated cases passed, "
        f"{sum(r['retries'] for r in done)} retries triggered, {len(rows)}/{len(TEST_CASES)} cases have results"
    )


if __name__ == "__main__":
    main()
