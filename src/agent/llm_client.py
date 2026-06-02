import time
import os
from dotenv import load_dotenv

load_dotenv()


def call_llm(prompt: str, provider: str = "groq", temperature: float = 0.3) -> str:
    if provider == "groq":
        from groq import Groq

        api_key = os.environ["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
        model = "llama-3.3-70b-versatile"
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "rate" in error_str or "quota" in error_str or "model_not_found" in error_str:
                    if attempt < 2:
                        time.sleep(15 * (attempt + 1))
                elif "tokens per day" in error_str:
                    print(f"  [Groq daily token limit reached: {e}]")
                    raise
                else:
                    raise
        raise RuntimeError("[Groq] API quota exhausted after 3 retries. Wait for daily reset or upgrade.")

    elif provider == "gemini":
        from google import genai

        api_key = os.environ["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        model = "gemini-2.0-flash"
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("[Gemini] API quota exhausted after 3 retries. Wait for daily reset or upgrade.")

    else:
        raise ValueError(f"Unknown provider: {provider}")
