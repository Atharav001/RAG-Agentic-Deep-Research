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
                if "429" in str(e) or "rate" in error_str or "quota" in error_str:
                    if attempt < 2:
                        time.sleep(10 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("Failed after 3 retries due to rate limiting")

    elif provider == "gemini":
        import google.generativeai as genai

        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    if attempt < 2:
                        time.sleep(10 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("Failed after 3 retries due to rate limiting")

    else:
        raise ValueError(f"Unknown provider: {provider}")
