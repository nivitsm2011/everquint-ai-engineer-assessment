"""Planner: turns a question into a short step-by-step plan (no solving)."""

from reasoner import prompts
from reasoner.llm_client import LLMClient
from reasoner.utils import extract_json


class Planner:
    """Asks the LLM for a plan and checks that it looks usable."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def plan(self, question: str, feedback: str = "") -> dict:
        """Return the plan dict. `feedback` is the verifier's notes on a retry."""
        feedback_text = prompts.PLANNER_FEEDBACK_PREFIX + feedback if feedback else ""
        user_prompt = prompts.PLANNER_USER_TEMPLATE.format(
            question=question, feedback=feedback_text
        )
        reply = self.llm.complete(prompts.PLANNER_SYSTEM_PROMPT, user_prompt, "planner")
        result = extract_json(reply)

        steps = result.get("plan")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Planner reply has no non-empty 'plan' list: {result}")
        return result
