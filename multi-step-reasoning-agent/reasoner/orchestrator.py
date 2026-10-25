"""Orchestrator: runs plan -> execute -> verify, retrying with verifier feedback."""

import json
from datetime import datetime, timezone
from pathlib import Path

from reasoner import config
from reasoner.executor import Executor
from reasoner.llm_client import LLMClient, get_llm_client
from reasoner.planner import Planner
from reasoner.verifier import Verifier

# Full traces go here (one JSON line per question), never to the user.
TRACE_PATH = Path(__file__).resolve().parent.parent / "logs" / "trace.jsonl"

UNSOLVABLE_ANSWER = "Cannot be solved"


class ReasoningAgent:
    """Multi-step reasoning agent with a self-check and retry loop."""

    def __init__(self, llm_client: LLMClient | None = None, max_retries: int | None = None) -> None:
        llm = llm_client or get_llm_client()
        llm.reset()  # start from a clean slate (matters for the mock's call counters)
        self.planner = Planner(llm)
        self.executor = Executor(llm)
        self.verifier = Verifier(llm)
        self.max_retries = config.MAX_RETRIES if max_retries is None else max_retries

    # ---- one attempt -------------------------------------------------------

    def _run_attempt(self, question: str, feedback: str) -> dict:
        """Plan, execute and verify once. Errors become a failed check, not a crash."""
        attempt: dict = {"plan": None, "execution": None, "verification": None, "error": None}
        try:
            attempt["plan"] = self.planner.plan(question, feedback)
            attempt["execution"] = self.executor.execute(question, attempt["plan"])
            attempt["verification"] = self.verifier.verify(question, attempt["execution"])
        except (ValueError, RuntimeError) as exc:  # bad JSON or failed LLM call
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            details = attempt["error"]
            attempt["verification"] = {
                "passed": False,
                "unsolvable": False,
                "checks": [{"check_name": "attempt_error", "passed": False, "details": details}],
                "feedback": f"The previous attempt crashed with: {details}",
            }
        return attempt

    # ---- main entry point --------------------------------------------------

    def solve(self, question: str) -> dict:
        """Solve a question; returns the fixed answer/status/reasoning/metadata dict."""
        attempts: list[dict] = []
        feedback = ""
        for _ in range(self.max_retries + 1):
            attempt = self._run_attempt(question, feedback)
            attempts.append(attempt)
            if attempt["verification"]["passed"]:
                break
            feedback = attempt["verification"]["feedback"]

        result = self._build_result(attempts)
        self._write_trace(question, attempts, result)
        return result

    # ---- building the result ----------------------------------------------

    @staticmethod
    def _plan_text(attempt: dict) -> str:
        """Numbered plan steps joined into one string."""
        steps = (attempt["plan"] or {}).get("plan", [])
        return " ".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))

    def _build_result(self, attempts: list[dict]) -> dict:
        final = attempts[-1]
        verification = final["verification"]
        execution = final["execution"] or {}

        if verification["passed"] and verification["unsolvable"]:
            status = "failed"
            answer = UNSOLVABLE_ANSWER
            reasoning = str(execution.get("unsolvable_reason") or "The question has no valid answer.")
        elif verification["passed"]:
            status = "success"
            answer = str(execution.get("final_answer", ""))
            reasoning = str(execution.get("short_explanation", ""))
        else:
            status = "failed"
            answer = self._best_unverified_answer(attempts)
            reasoning = self._failure_reason(verification)

        return {
            "answer": answer,
            "status": status,
            "reasoning_visible_to_user": reasoning,
            "metadata": {
                "plan": self._plan_text(final),
                "checks": verification["checks"],
                "retries": len(attempts) - 1,
            },
        }

    @staticmethod
    def _best_unverified_answer(attempts: list[dict]) -> str:
        """Answer from the attempt that passed the most checks (latest wins ties)."""
        best_answer, best_score = "", -1
        for attempt in attempts:
            answer = (attempt["execution"] or {}).get("final_answer")
            if not answer:
                continue
            score = sum(c["passed"] for c in attempt["verification"]["checks"])
            if score >= best_score:
                best_answer, best_score = str(answer), score
        return f"Unverified: {best_answer or 'no answer could be produced'}"

    @staticmethod
    def _failure_reason(verification: dict) -> str:
        """Short, user-facing summary of which checks failed."""
        failed = [c["check_name"] for c in verification["checks"] if not c["passed"]]
        return "The answer could not be verified. Failed checks: " + ", ".join(failed) + "."

    # ---- trace log ---------------------------------------------------------

    @staticmethod
    def _write_trace(question: str, attempts: list[dict], result: dict) -> None:
        """Append the full trace (all attempts) as one JSON line."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "result": result,
            "attempts": attempts,
        }
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
