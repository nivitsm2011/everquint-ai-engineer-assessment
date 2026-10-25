"""Executor: follows the plan, then re-runs every calculation with real tools."""

import inspect
import json
from typing import Any

from reasoner import prompts
from reasoner.llm_client import LLMClient
from reasoner.tools import TOOLS, run_tool
from reasoner.utils import extract_json

TOLERANCE = 1e-6


def _call_tool(tool: str, args: Any) -> dict:
    """Run a tool whose args came from the LLM as a positional list."""
    func = TOOLS.get(tool)
    if func is None:
        return run_tool(tool, {})  # gives the standard "Unknown tool" error
    if isinstance(args, dict):
        return run_tool(tool, args)
    if not isinstance(args, list):
        return {"ok": False, "result": None, "error": "args must be a list"}

    # Map positional args onto the tool's parameter names.
    names = list(inspect.signature(func).parameters)
    if len(args) != len(names):
        error = f"{tool} expects {len(names)} args, got {len(args)}"
        return {"ok": False, "result": None, "error": error}
    return run_tool(tool, dict(zip(names, args)))


def _normalize(value: Any) -> Any:
    """Make values comparable: numbers -> float, lists -> tuples, text -> lowercase."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(v) for v in value)
    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        return text


def _same(a: Any, b: Any) -> bool:
    """Compare two normalized values (floats within TOLERANCE)."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= TOLERANCE
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


class Executor:
    """Asks the LLM to solve following the plan, then checks its maths with tools."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def execute(self, question: str, plan_dict: dict) -> dict:
        """Return the executor output with tool results added to every step."""
        user_prompt = prompts.EXECUTOR_USER_TEMPLATE.format(
            question=question, plan=json.dumps(plan_dict.get("plan", []))
        )
        reply = self.llm.complete(prompts.EXECUTOR_SYSTEM_PROMPT, user_prompt, "executor")
        result = extract_json(reply)

        steps = result.get("steps")
        if not isinstance(steps, list):
            return result  # the verifier's output_format check will flag this

        for step in steps:
            tool = step.get("tool", "none")
            if tool == "none":
                continue
            outcome = _call_tool(tool, step.get("args", []))
            step["tool_ok"] = outcome["ok"]
            step["tool_result"] = outcome["result"] if outcome["ok"] else outcome["error"]

            # Trust the tool over the LLM: replace a wrong claimed result.
            if outcome["ok"] and not _same(
                _normalize(outcome["result"]), _normalize(step.get("claimed_result"))
            ):
                step["claimed_result"] = outcome["result"]
                step["corrected_by_tool"] = True
        return result
