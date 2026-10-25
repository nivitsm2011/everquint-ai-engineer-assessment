"""End-to-end tests of the agent using the mock LLM (no API key needed)."""

from reasoner.llm_client import MockLLMClient
from reasoner.orchestrator import ReasoningAgent
from tests.test_cases import TEST_CASES


def solve_case(case_id: int) -> dict:
    question = next(c["question"] for c in TEST_CASES if c["id"] == case_id)
    return ReasoningAgent(MockLLMClient()).solve(question)


def test_output_schema_is_exact():
    result = solve_case(1)
    assert set(result) == {"answer", "status", "reasoning_visible_to_user", "metadata"}
    assert set(result["metadata"]) == {"plan", "checks", "retries"}
    assert result["status"] in ("success", "failed")


def test_q11_bat_and_ball_is_caught_and_retried():
    result = solve_case(11)
    assert result["metadata"]["retries"] >= 1
    assert result["status"] == "success"
    assert "0.05" in result["answer"]


def test_q13_impossible_question_fails():
    result = solve_case(13)
    assert result["status"] == "failed"
    assert result["answer"] == "Cannot be solved"


def test_q1_answer_mentions_hours_and_minutes():
    result = solve_case(1)
    assert "3" in result["answer"] and "35" in result["answer"]


def test_unknown_question_fails_without_crashing():
    result = ReasoningAgent(MockLLMClient()).solve("something the mock does not know")
    assert result["status"] == "failed"
    assert result["answer"].startswith("Unverified: ")
