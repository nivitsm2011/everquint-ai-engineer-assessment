"""Swappable LLM layer: one interface, three backends (Anthropic, Gemini, mock)."""

import json
import time
from abc import ABC, abstractmethod
from typing import Callable

from reasoner import config

# Calls to a real API are retried this many times after the first failure.
API_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0
# Rate limits (HTTP 429) need a much longer pause: the quota resets per minute.
RATE_LIMIT_DELAY_SECONDS = 20.0


class LLMClient(ABC):
    """Common interface so the rest of the project never cares which LLM is used."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, role: str) -> str:
        """Return the model's text reply.

        role is "planner", "executor" or "verifier"; it is only used by the
        mock (to pick a canned reply) and for logging.
        """

    def reset(self) -> None:
        """Forget any per-run state. Only the mock has any."""


def _is_rate_limited(exc: Exception) -> bool:
    """True if the error looks like an HTTP 429 from either SDK."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code == 429 or "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


def _call_with_retries(call: Callable[[], str], provider: str) -> str:
    """Run an API call, retrying up to API_RETRIES times, then raise a clear error."""
    last_error: Exception | None = None
    for attempt in range(API_RETRIES + 1):
        try:
            return call()
        except Exception as exc:  # network/API errors differ per SDK, so catch broadly
            last_error = exc
            if attempt < API_RETRIES:
                if _is_rate_limited(exc):
                    time.sleep(RATE_LIMIT_DELAY_SECONDS * (attempt + 1))  # 20 s, then 40 s
                else:
                    time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(
        f"{provider} API call failed after {API_RETRIES + 1} attempts: {last_error}"
    ) from last_error


class AnthropicClient(LLMClient):
    """Claude via the anthropic SDK."""

    def __init__(self) -> None:
        import anthropic  # imported here so mock mode works without the SDK

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def complete(self, system_prompt: str, user_prompt: str, role: str) -> str:
        def call() -> str:
            response = self._client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=1500,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # The reply is a list of content blocks; join the text ones.
            return "".join(b.text for b in response.content if b.type == "text")

        return _call_with_retries(call, "Anthropic")


class GeminiClient(LLMClient):
    """Gemini via the google-genai SDK."""

    def __init__(self) -> None:
        from google import genai  # imported here so mock mode works without the SDK

        self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def complete(self, system_prompt: str, user_prompt: str, role: str) -> str:
        from google.genai import types

        def call() -> str:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                ),
            )
            return response.text or ""

        return _call_with_retries(call, "Gemini")


def _plan(problem_type: str, quantities: dict, steps: list[str], answer_format: str) -> dict:
    """Canned planner reply in the format from prompts.py."""
    return {
        "problem_type": problem_type,
        "quantities": quantities,
        "plan": steps,
        "expected_answer_format": answer_format,
    }


def _step(number: int, description: str, tool: str, args: list, claimed: object) -> dict:
    """One canned executor step."""
    return {
        "step": number,
        "description": description,
        "tool": tool,
        "args": args,
        "claimed_result": claimed,
    }


def _execution(steps: list[dict], final: str, canonical: str, explanation: str) -> dict:
    """Canned executor reply for a solvable question."""
    return {
        "solvable": True,
        "unsolvable_reason": "",
        "steps": steps,
        "final_answer": final,
        "canonical_answer": canonical,
        "short_explanation": explanation,
    }


_SLOTS_60 = [["09:00", "09:30"], ["09:45", "10:30"], ["11:00", "12:00"]]
_SLOTS_30 = [["09:00", "09:30"], ["10:00", "10:20"], ["13:15", "13:45"]]

# For each known question: the planner reply, a list of executor replies (the
# Nth call gets the Nth reply, the last one repeats) and the correct answer the
# mock verifier will "independently" arrive at.
_MOCK_CASES: dict[str, dict] = {
    "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?": {
        "answer": "215",
        "plan": _plan("time", {"start": "14:30", "end": "18:05"},
                      ["Parse the departure and arrival times", "Compute the minutes between them",
                       "Format the answer as integer minutes"], "integer minutes"),
        "executor": [_execution(
            [_step(1, "Minutes from 14:30 to 18:05", "time_diff", ["14:30", "18:05"], 215)],
            "3 h 35 min (215 minutes)", "215",
            "The train leaves at 14:30 and arrives at 18:05, so the journey takes 3 h 35 min.")],
    },
    "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?": {
        "answer": "9",
        "plan": _plan("arithmetic", {"red": 3, "green": None, "total": None},
                      ["Extract red = 3", "Compute green = red * 2", "Compute total = red + green",
                       "Format as a plain number"], "plain number"),
        "executor": [_execution(
            [_step(1, "Green apples", "calc", ["3*2"], 6), _step(2, "Total apples", "calc", ["3+6"], 9)],
            "9 apples", "9", "Alice has 6 green apples (twice 3), so 3 + 6 = 9 apples in total.")],
    },
    "A meeting needs 60 minutes. There are free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which slots can fit the meeting?": {
        "answer": "11:00-12:00",
        "plan": _plan("scheduling", {"needed_minutes": 60, "slots": 3},
                      ["Parse the slots and the needed duration", "Keep slots of at least 60 minutes",
                       "Format as a sorted list"], "comma-separated slots"),
        "executor": [_execution(
            [_step(1, "Slots at least 60 minutes long", "slots_that_fit", [_SLOTS_60, 60], [["11:00", "12:00"]])],
            "Only 11:00-12:00 fits", "11:00-12:00",
            "Only the 11:00-12:00 slot (60 minutes) is long enough for the meeting.")],
    },
    "What is 15% of 240?": {
        "answer": "36",
        "plan": _plan("arithmetic", {"percent": 15, "base": 240},
                      ["Extract 15% and 240", "Compute 240 * 15 / 100", "Format as a plain number"],
                      "plain number"),
        "executor": [_execution(
            [_step(1, "15 percent of 240", "calc", ["240*15/100"], 36)],
            "36", "36", "15% of 240 is 240 x 0.15 = 36.")],
    },
    "A book has 12 chapters with 25 pages each. How many pages in total?": {
        "answer": "300",
        "plan": _plan("arithmetic", {"chapters": 12, "pages_per_chapter": 25},
                      ["Extract chapters and pages per chapter", "Multiply them", "Format as a plain number"],
                      "plain number"),
        "executor": [_execution(
            [_step(1, "Total pages", "calc", ["12*25"], 300)],
            "300 pages", "300", "12 chapters of 25 pages each make 300 pages.")],
    },
    "A bus leaves at 08:15 and the trip takes 2 hours 50 minutes. What time does it arrive?": {
        "answer": "11:05",
        "plan": _plan("time", {"start": "08:15", "duration": "2 h 50 min"},
                      ["Convert the trip length to minutes", "Add it to the departure time",
                       "Format as HH:MM"], "HH:MM"),
        "executor": [_execution(
            [_step(1, "Trip length in minutes", "calc", ["2*60+50"], 170),
             _step(2, "Arrival time", "add_duration", ["08:15", 170], "11:05")],
            "11:05", "11:05", "Leaving at 08:15 and travelling 2 h 50 min, the bus arrives at 11:05.")],
    },
    "Tom had 50 rupees, spent 18 on lunch and 7 on tea. How much is left?": {
        "answer": "25",
        "plan": _plan("arithmetic", {"start": 50, "lunch": 18, "tea": 7},
                      ["Extract the amounts", "Subtract both expenses", "Format as a plain number"],
                      "plain number"),
        "executor": [_execution(
            [_step(1, "Money left", "calc", ["50-18-7"], 25)],
            "25 rupees", "25", "Tom spent 18 + 7 = 25 rupees, so 25 of his 50 rupees are left.")],
    },
    "What is the average of 12, 18 and 30?": {
        "answer": "20",
        "plan": _plan("arithmetic", {"numbers": [12, 18, 30]},
                      ["List the numbers", "Add them and divide by the count", "Format as a plain number"],
                      "plain number"),
        "executor": [_execution(
            [_step(1, "Average", "calc", ["(12+18+30)/3"], 20)],
            "20", "20", "The numbers add up to 60, and 60 divided by 3 is 20.")],
    },
    "A train leaves at 22:45 and arrives at 01:15 the next day. How long is the journey?": {
        "answer": "150",
        "plan": _plan("time", {"start": "22:45", "end": "01:15"},
                      ["Parse the times", "Compute the minutes, allowing for passing midnight",
                       "Format as integer minutes"], "integer minutes"),
        "executor": [_execution(
            [_step(1, "Minutes from 22:45 to 01:15 (overnight)", "time_diff", ["22:45", "01:15"], 150)],
            "2 h 30 min (150 minutes)", "150",
            "The journey crosses midnight: 75 minutes to midnight plus 75 after, 2 h 30 min.")],
    },
    "A meeting needs 30 minutes. Free slots: 09:00-09:30, 10:00-10:20, 13:15-13:45. Which slots fit?": {
        "answer": "09:00-09:30,13:15-13:45",
        "plan": _plan("scheduling", {"needed_minutes": 30, "slots": 3},
                      ["Parse the slots and the needed duration", "Keep slots of at least 30 minutes (exact fit counts)",
                       "Format as a sorted list"], "comma-separated slots"),
        "executor": [_execution(
            [_step(1, "Slots at least 30 minutes long", "slots_that_fit", [_SLOTS_30, 30],
                   [["09:00", "09:30"], ["13:15", "13:45"]])],
            "09:00-09:30 and 13:15-13:45 fit", "09:00-09:30,13:15-13:45",
            "Two slots are exactly 30 minutes long, which is enough; 10:00-10:20 is too short.")],
    },
    "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost?": {
        "answer": "0.05",
        "plan": _plan("arithmetic", {"total": 1.10, "difference": 1.00, "ball": None},
                      ["Let ball = x and bat = x + 1.00", "Set x + (x + 1.00) = 1.10 and solve for x",
                       "Check bat - ball = 1.00", "Format as a plain number"], "plain number"),
        "executor": [
            # First reply: the classic naive mistake (just subtracting 1.00 from 1.10).
            _execution([_step(1, "Subtract the bat difference from the total", "calc", ["1.10-1.00"], 0.10)],
                       "0.10", "0.10", "The bat costs 1.00, so the ball costs the remaining 0.10."),
            # Second reply (after verifier feedback): solve properly.
            _execution([_step(1, "Solve 2x + 1.00 = 1.10, so x = (1.10 - 1.00) / 2", "calc",
                              ["(1.10-1.00)/2"], 0.05)],
                       "0.05", "0.05", "If the ball costs 0.05 the bat costs 1.05, together 1.10 and 1.00 apart."),
        ],
    },
    "A recipe needs 3 eggs per cake. How many eggs are needed for 2 dozen cakes?": {
        "answer": "72",
        "plan": _plan("arithmetic", {"eggs_per_cake": 3, "dozens": 2},
                      ["Convert 2 dozen to 24 cakes", "Multiply by 3 eggs per cake", "Format as a plain number"],
                      "plain number"),
        "executor": [_execution(
            [_step(1, "Cakes in 2 dozen", "calc", ["2*12"], 24), _step(2, "Eggs needed", "calc", ["3*24"], 72)],
            "72 eggs", "72", "2 dozen is 24 cakes, and at 3 eggs each that needs 72 eggs.")],
    },
    "Alice has 5 apples and gives away 8. How many apples does she have left?": {
        "answer": "UNSOLVABLE",
        "plan": _plan("logic", {"apples": 5, "given_away": 8},
                      ["Extract the quantities", "Check Alice can give away that many", "Report the result"],
                      "plain number, or UNSOLVABLE"),
        "executor": [{
            "solvable": False,
            "unsolvable_reason": "Alice cannot give away 8 apples when she only has 5.",
            "steps": [_step(1, "Check 8 <= 5", "none", [], None)],
            "final_answer": "Cannot be solved",
            "canonical_answer": "UNSOLVABLE",
            "short_explanation": "She only has 5 apples, so she cannot give away 8.",
        }],
    },
}


class MockLLMClient(LLMClient):
    """Returns canned JSON so the project runs with no API key.

    It recognises the 13 test questions by their text and answers in the exact
    formats from prompts.py. Unknown questions get {"error": "no mock response"}.
    """

    CANNED_RESPONSES: dict[str, dict] = _MOCK_CASES

    def __init__(self) -> None:
        self._calls: dict[tuple[str, str], int] = {}

    def reset(self) -> None:
        """Clear the per-question, per-role call counters."""
        self._calls = {}

    def complete(self, system_prompt: str, user_prompt: str, role: str) -> str:
        question = next((q for q in self.CANNED_RESPONSES if q in user_prompt), None)
        if question is None:
            return json.dumps({"error": "no mock response"})
        case = self.CANNED_RESPONSES[question]

        call_number = self._calls.get((question, role), 0)
        self._calls[(question, role)] = call_number + 1

        if role == "planner":
            return json.dumps(case["plan"])
        if role == "executor":
            replies = case["executor"]
            return json.dumps(replies[min(call_number, len(replies) - 1)])
        return json.dumps(self._verify(case["answer"], user_prompt))

    @staticmethod
    def _verify(correct: str, user_prompt: str) -> dict:
        """Verifier reply: always the correct answer; approve only if the proposal matches."""
        proposed = ""
        marker = "Proposed solution: "
        if marker in user_prompt:
            text = user_prompt.split(marker, 1)[1].split("\n\nFirst solve", 1)[0]
            try:
                proposed = str(json.loads(text).get("canonical_answer", ""))
            except (ValueError, AttributeError):
                pass
        if _same_answer(proposed, correct):
            return {"independent_canonical_answer": correct, "verdict": "approve", "issues": [],
                    "explanation": f"My own working also gives {correct}, which matches."}
        return {"independent_canonical_answer": correct, "verdict": "reject",
                "issues": [f"Proposed answer {proposed!r} does not match my independent answer {correct!r}"],
                "explanation": f"Solving from scratch gives {correct}, not {proposed}."}


def _same_answer(a: str, b: str) -> bool:
    """Loose equality for the mock verifier: numbers as floats, lists order-free."""
    def norm(text: str) -> list[str]:
        items = []
        for item in text.strip().lower().replace(" ", "").split(","):
            try:
                items.append(f"{float(item):g}")
            except ValueError:
                items.append(item)
        return sorted(items)

    return norm(a) == norm(b)


def get_llm_client() -> LLMClient:
    """Build the client named by LLM_PROVIDER, falling back to the mock if no key."""
    provider = config.LLM_PROVIDER

    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            print("WARNING: ANTHROPIC_API_KEY is missing; falling back to mock LLM.")
            return MockLLMClient()
        return AnthropicClient()

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY is missing; falling back to mock LLM.")
            return MockLLMClient()
        return GeminiClient()

    if provider != "mock":
        print(f"WARNING: unknown LLM_PROVIDER '{provider}'; using mock LLM.")
    return MockLLMClient()
