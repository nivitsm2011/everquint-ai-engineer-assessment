"""Small helpers shared across the project."""

import json
import re


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LLM reply.

    Handles replies wrapped in markdown code fences (```json ... ```) or
    surrounded by extra prose. Raises ValueError if nothing parses.
    """
    # Remove markdown code fence markers, keep the content inside.
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)

    # Take everything from the first "{" to the last "}".
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in text: {text[:200]!r}")

    try:
        result = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON ({exc}): {text[:200]!r}") from exc

    if not isinstance(result, dict):
        raise ValueError("Parsed JSON is not an object")
    return result
