"""
LLM configuration and factory for CodePilot AI.

Supports Groq as the default provider. Modular design allows swapping
to OpenAI, Anthropic, or any other LangChain-compatible provider.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

MAX_REVISION_CYCLES: int = 3
MAX_TOKENS_PER_FILE: int = 4000   # Truncate large files before sending to LLM
MAX_FILES_PER_AGENT: int = 20     # Cap files an agent reads in one pass
REPO_TEMP_DIR: str = os.getenv("REPO_TEMP_DIR", "/tmp/codepilot_repos")


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.1):
    """
    Return a configured ChatGroq LLM instance.

    Raises:
        ValueError: If GROQ_API_KEY is not configured.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )

    # Import here so the module can be imported without langchain installed
    from langchain_groq import ChatGroq  # type: ignore

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
        max_retries=3,
    )


def check_llm_config() -> tuple[bool, str]:
    """
    Check whether the LLM is properly configured.

    Returns:
        (ok: bool, message: str)
    """
    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is not set in environment variables."
    if GROQ_API_KEY in ("your_groq_api_key_here", "REPLACE_ME", ""):
        return False, "GROQ_API_KEY appears to be a placeholder value."
    return True, f"Groq configured with model: {GROQ_MODEL}"
