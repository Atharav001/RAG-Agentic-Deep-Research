"""
LLM Client — wrapper around OpenAI API.
Uses gpt-4o-mini for cost efficiency. Includes retry logic for rate limits.
"""

import time
import os
from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import LLM_TEMPERATURE, LLM_MAX_TOKENS

_client = None


def get_client():
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(prompt: str, system_prompt: str = "", temperature: float | None = None, max_retries: int = 5) -> str:
    """Call the LLM and return the response text. Retries on rate limits."""
    client = get_client()
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "quota" in error_str:
                wait = min(15 * (attempt + 1), 60)
                print(f"  [Rate limited] Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")
