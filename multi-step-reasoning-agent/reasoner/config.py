"""Settings loaded from environment variables (or a local .env file)."""

import os

from dotenv import load_dotenv

# Read .env into the process environment (does nothing if the file is missing).
load_dotenv()

# Which LLM backend to use: "mock", "anthropic" or "gemini".
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock").lower()

# Anthropic settings.
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

# Gemini settings.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# How many times to retry when the self-check fails.
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
