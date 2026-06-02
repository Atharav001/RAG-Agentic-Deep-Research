import os
import requests
import json

import time


def call_llm(prompt: str, provider: str = "ollama", temperature: float = 0.3, max_retries: int = 2) -> str:
    if provider == "ollama":
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "gemma3:4b",
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except requests.exceptions.ConnectionError:
                print("❌ Ollama not running! Execute 'ollama serve' in another terminal.")
                return ""
            except Exception as e:
                print(f"⏳ Ollama retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
        return ""

    print(f"Warning: Provider {provider} not supported in current local setup.")
    return ""
