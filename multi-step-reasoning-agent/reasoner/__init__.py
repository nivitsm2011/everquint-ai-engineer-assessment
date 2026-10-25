"""Multi-step reasoning agent with self-checking."""

from reasoner.llm_client import get_llm_client
from reasoner.orchestrator import ReasoningAgent


def solve(question: str) -> dict:
    """Notebook interface: build an agent from the configured LLM and solve."""
    return ReasoningAgent(get_llm_client()).solve(question)
