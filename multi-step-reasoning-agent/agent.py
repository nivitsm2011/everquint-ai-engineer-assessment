"""CLI for the Multi-Step Reasoning Agent.

  python agent.py                      interactive loop
  python agent.py --question "text"    answer once and exit
  python agent.py --debug ...          also print where the full trace is saved

The answer JSON goes to stdout; status messages go to stderr, so stdout can be
piped straight into other tools.
"""

import argparse
import json
import sys

from reasoner import config
from reasoner.llm_client import AnthropicClient, GeminiClient, LLMClient, get_llm_client
from reasoner.orchestrator import TRACE_PATH, ReasoningAgent


def describe_provider(llm: LLMClient) -> str:
    """Name of the LLM actually in use (the mock if a key was missing)."""
    if isinstance(llm, AnthropicClient):
        return f"anthropic ({config.ANTHROPIC_MODEL})"
    if isinstance(llm, GeminiClient):
        return f"gemini ({config.GEMINI_MODEL})"
    return "mock"


def answer(agent: ReasoningAgent, llm: LLMClient, question: str, debug: bool) -> None:
    """Solve one question and print the result JSON."""
    llm.reset()  # so the mock's per-question call counters start fresh each time
    print(json.dumps(agent.solve(question), indent=2))
    if debug:
        print(f"Full trace: {TRACE_PATH}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-step reasoning agent with self-checking")
    parser.add_argument("--question", help="answer this question once and exit")
    parser.add_argument("--debug", action="store_true", help="also print the trace file path")
    args = parser.parse_args()

    llm = get_llm_client()
    agent = ReasoningAgent(llm)
    print(f"LLM provider: {describe_provider(llm)}", file=sys.stderr)

    if args.question:
        answer(agent, llm, args.question, args.debug)
        return

    while True:
        try:
            question = input("Ask a question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit"):
            break
        if question:
            answer(agent, llm, question, args.debug)


if __name__ == "__main__":
    main()
