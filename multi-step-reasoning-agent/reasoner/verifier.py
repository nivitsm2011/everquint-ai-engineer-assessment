"""Verifier: runs rule-based checks plus an independent re-solve by the LLM."""

import json
from typing import Any

from reasoner import prompts
from reasoner.llm_client import LLMClient
from reasoner.tools import parse_time
from reasoner.utils import extract_json

REQUIRED_KEYS = [
    "solvable",
    "steps",
    "final_answer",
    "canonical_answer",
    "short_explanation",
]
UNSOLVABLE = "unsolvable"


def normalize_answer(answer: Any) -> str:
    """Make canonical answers comparable.

    Lowercase, no spaces, numbers compared as floats ("9" == "9.0"),
    and comma-separated list items sorted.
    """
    items = str(answer).strip().lower().replace(" ", "").split(",")
    cleaned = []
    for item in items:
        try:
            cleaned.append(f"{round(float(item), 6):g}")
        except ValueError:
            cleaned.append(item)
    return ",".join(sorted(cleaned))


def _check(name: str, passed: bool, details: str) -> dict:
    return {"check_name": name, "passed": passed, "details": details}


def _time_strings(tool: str, args: Any) -> list:
    """Pick out the HH:MM arguments of the time-based tools."""
    if not isinstance(args, list) or not args:
        return []
    if tool == "time_diff":
        return list(args[:2])
    if tool == "add_duration":
        return [args[0]]
    if tool == "slots_that_fit" and isinstance(args[0], list):
        return [t for pair in args[0] if isinstance(pair, list) for t in pair]
    return []


class Verifier:
    """Decides whether an executor output can be trusted."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ---- individual checks -------------------------------------------------

    def _check_format(self, execution: dict) -> dict:
        missing = [k for k in REQUIRED_KEYS if k not in execution]
        if missing:
            return _check("output_format", False, f"Missing keys: {missing}")
        if not isinstance(execution["steps"], list):
            return _check("output_format", False, "'steps' is not a list")
        return _check("output_format", True, "All required keys present")

    def _tool_steps(self, execution: dict) -> list:
        steps = execution.get("steps")
        if not isinstance(steps, list):
            return []
        return [s for s in steps if isinstance(s, dict) and s.get("tool", "none") != "none"]

    def _check_tools(self, execution: dict) -> dict:
        steps = self._tool_steps(execution)
        failed = [s.get("step") for s in steps if not s.get("tool_ok")]
        if failed:
            return _check("tool_consistency", False, f"Tool calls failed in steps {failed}")
        corrected = [s.get("step") for s in steps if s.get("corrected_by_tool")]
        note = f"Tool corrected the claimed result in steps {corrected}" if corrected else "ok"
        return _check("tool_consistency", True, f"{len(steps)} tool call(s) succeeded; {note}")

    def _check_constraints(self, execution: dict) -> dict:
        problems = []
        for step in self._tool_steps(execution):
            result = step.get("tool_result")
            if isinstance(result, (int, float)) and not isinstance(result, bool) and result < 0:
                problems.append(f"step {step.get('step')}: negative result {result}")
            for value in _time_strings(step["tool"], step.get("args")):
                try:
                    parse_time(value)
                except ValueError as exc:
                    problems.append(f"step {step.get('step')}: {exc}")
        if not str(execution.get("canonical_answer", "")).strip():
            problems.append("canonical_answer is empty")
        if problems:
            return _check("constraints", False, "; ".join(problems))
        return _check("constraints", True, "No negative values, valid times, answer present")

    def _independent_resolve(self, question: str, execution: dict) -> tuple[list, dict]:
        """Ask the LLM to solve from scratch; return the two related checks + raw output."""
        user_prompt = prompts.VERIFIER_USER_TEMPLATE.format(
            question=question,
            proposed_solution=json.dumps(execution, default=str),
        )
        reply = self.llm.complete(prompts.VERIFIER_SYSTEM_PROMPT, user_prompt, "verifier")
        try:
            output = extract_json(reply)
        except ValueError as exc:
            detail = f"Verifier reply was not valid JSON: {exc}"
            return [_check("independent_resolve", False, detail),
                    _check("llm_verdict", False, detail)], {}

        independent = output.get("independent_canonical_answer", "")
        proposed = execution.get("canonical_answer", "")
        matches = normalize_answer(independent) == normalize_answer(proposed)
        resolve = _check(
            "independent_resolve",
            matches,
            f"Independent answer {independent!r} vs executor answer {proposed!r}",
        )
        verdict = output.get("verdict")
        issues = output.get("issues") or []
        detail = f"Verdict: {verdict}" + (f"; issues: {issues}" if issues else "")
        return [resolve, _check("llm_verdict", verdict == "approve", detail)], output

    # ---- main entry point --------------------------------------------------

    def verify(self, question: str, execution: dict) -> dict:
        """Run all checks and return passed / checks / feedback / unsolvable."""
        checks = [
            self._check_format(execution),
            self._check_tools(execution),
            self._check_constraints(execution),
        ]
        resolve_checks, output = self._independent_resolve(question, execution)
        checks += resolve_checks

        # Special case: both sides agree the question has no valid answer.
        unsolvable = (
            execution.get("solvable") is False
            and normalize_answer(output.get("independent_canonical_answer", "")) == UNSOLVABLE
        )
        if unsolvable:
            for check in checks:
                check["passed"] = True
                check["details"] += " (both sides agree: UNSOLVABLE)"

        passed = all(c["passed"] for c in checks)
        return {
            "passed": passed,
            "unsolvable": unsolvable,
            "checks": checks,
            "feedback": "" if passed else self._build_feedback(checks, output),
        }

    @staticmethod
    def _build_feedback(checks: list, output: dict) -> str:
        """Plain-text notes for the planner on a retry."""
        lines = [f"- {c['check_name']}: {c['details']}" for c in checks if not c["passed"]]
        lines += [f"- verifier issue: {i}" for i in (output.get("issues") or [])]
        return "\n".join(lines)
