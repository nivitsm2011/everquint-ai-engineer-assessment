"""Deterministic helper tools, so we never trust the LLM's arithmetic."""

import ast
import operator
import re
from typing import Any, Callable

MINUTES_PER_DAY = 1440

# The only arithmetic operators safe_calc will evaluate.
_BINARY_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# Guard against absurd exponents like 9**9**9 that would hang the program.
_MAX_EXPONENT = 1000


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST node, rejecting everything else."""
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so exclude it explicitly.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("Exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_calc(expression: str) -> float:
    """Evaluate +, -, *, /, //, %, ** and parentheses safely (no eval)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return float(_eval_node(tree.body))
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    except (SyntaxError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid expression {expression!r}: {exc}") from exc


def parse_time(hhmm: str) -> int:
    """Convert "14:30" to minutes since midnight."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(hhmm).strip())
    if not match:
        raise ValueError(f"Time must look like HH:MM, got {hhmm!r}")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if not 0 <= hours <= 23:
        raise ValueError(f"Hours must be 00-23, got {hours}")
    if not 0 <= minutes <= 59:
        raise ValueError(f"Minutes must be 00-59, got {minutes}")
    return hours * 60 + minutes


def time_diff(start: str, end: str) -> int:
    """Minutes from start to end; if end is earlier, assume the next day."""
    diff = parse_time(end) - parse_time(start)
    if diff < 0:
        diff += MINUTES_PER_DAY
    return diff


def add_duration(start: str, minutes: int) -> str:
    """Add minutes to a start time and return HH:MM, wrapping past midnight."""
    total = (parse_time(start) + int(minutes)) % MINUTES_PER_DAY
    return f"{total // 60:02d}:{total % 60:02d}"


def format_minutes(total: int) -> str:
    """Turn 215 into "3 h 35 min"."""
    total = int(total)
    return f"{total // 60} h {total % 60} min"


def slots_that_fit(slots: list[list[str]], needed_minutes: int) -> list[list[str]]:
    """Return the [start, end] slots at least needed_minutes long (exact fit counts)."""
    return [
        [start, end]
        for start, end in slots
        if time_diff(start, end) >= needed_minutes
    ]


# Tools the executor/verifier may call by name.
TOOLS: dict[str, Callable[..., Any]] = {
    "calc": safe_calc,
    "time_diff": time_diff,
    "add_duration": add_duration,
    "slots_that_fit": slots_that_fit,
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool by name with keyword args; never raises, always returns a dict."""
    if name not in TOOLS:
        return {"ok": False, "result": None, "error": f"Unknown tool: {name!r}"}
    try:
        return {"ok": True, "result": TOOLS[name](**args), "error": None}
    except Exception as exc:  # report any tool failure back to the caller
        return {"ok": False, "result": None, "error": str(exc)}
