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
                print("WARNING: Ollama not running! Execute 'ollama serve' in another terminal.")
                return ""
            except Exception as e:
                print(f"Retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
        return ""

    elif provider == "nvidia":
        api_key = os.environ.get("NVIDIA_NIM_API_KEY")
        base_url = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
        model = os.environ.get("NVIDIA_NIM_MODEL", "meta/llama-3.1-8b-instruct")

        if not api_key:
            print("WARNING: NVIDIA_NIM_API_KEY not set in .env")
            return ""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
        return ""

    print(f"Warning: Provider {provider} not supported.")
    return ""
