"""
Agent Components — each can be toggled on/off for ablation.

1. Planner: decomposes question into sub-questions
2. Reflector: decides if evidence is sufficient or needs more retrieval
3. Synthesizer: writes cited answer from evidence
4. Citation Verifier: checks each citation is grounded
"""

import json
import re
from src.agent.llm_client import call_llm


def plan(question: str, provider: str = "ollama") -> list[str]:
    """Decompose a research question into sub-questions."""
    prompt = f"""You are a research assistant. Decompose the following research question into 2-5 focused sub-questions that, when answered together, will fully address the original question.

Each sub-question should be specific enough to be answered by searching a corpus of academic papers about LLM agents.

Question: {question}

Return ONLY a JSON array of sub-question strings. No explanation.
Example: ["What is X?", "How does Y compare to Z?"]"""

    response = call_llm(prompt, provider=provider)
    # Parse JSON array from response
    try:
        # Find JSON array in response
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            sub_questions = json.loads(match.group())
            return sub_questions[:5]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: return original question
    return [question]


def reflect(question: str, evidence: list[dict], round_num: int, provider: str = "ollama", q_type: str = "factoid") -> dict:
    """
    Evaluate whether collected evidence is sufficient to answer the question.

    Returns:
        {"sufficient": bool, "reasoning": str, "refined_queries": list[str]}
    """
    unique_ids = sorted(set(e.get("arxiv_id", "") for e in evidence if e.get("arxiv_id")))

    if q_type == "survey":
        min_needed = 4
    else:
        min_needed = 2

    prompt = f"""You are a retrieval manager. Question type: {q_type}. Unique papers found so far: {unique_ids}. Minimum needed: {min_needed} if survey, 2 if comparative/factoid. Output JSON: {{"sufficient": true/false, "reasoning": "...", "refined_queries": ["..."]}}."""

    response = call_llm(prompt, provider=provider)
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"sufficient": True, "reasoning": "Could not parse reflection", "refined_queries": []}


def compress_context(query: str, evidence_list: list[dict], provider: str = "ollama") -> list[dict]:
    result = []
    for ev in evidence_list:
        text = ev.get("compressed_text") or ev["text"]
        try:
            response = call_llm(
                f"Query: {query}\nPassage: {text}\n\nExtract ONLY the 1-2 sentences that answer the query. If none, output 'NONE'.",
                provider=provider,
                temperature=0.0
            )
            compressed = response.strip()
            if compressed.upper() == "NONE" or not compressed:
                continue
            result.append({"arxiv_id": ev["arxiv_id"], "compressed_text": compressed})
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            fallback = " ".join(sentences[:2])
            result.append({"arxiv_id": ev["arxiv_id"], "compressed_text": fallback})
    return result


def synthesize(question: str, evidence: list[dict], provider: str = "ollama", q_type: str = "factoid") -> str:
    """
    Write a research answer using ONLY the provided evidence, with inline citations.
    Citations use arXiv IDs: [XXXX.XXXXX]
    """
    # Deduplicate evidence by arxiv_id to provide cleaner context
    seen_ids = set()
    unique_evidence = []
    for e in evidence:
        key = (e["arxiv_id"], e.get("chunk_index", 0))
        if key not in seen_ids:
            seen_ids.add(key)
            unique_evidence.append(e)

    evidence_text = "\n\n".join(
        f"[Source: {e['arxiv_id']}] Title: {e.get('title', 'N/A')}\n"
        f"Section: {e.get('section', 'unknown')}\n"
        f"Content: {e.get('compressed_text') or e['text'][:500]}"
        for e in unique_evidence
    )

    if q_type == "factoid":
        length_constraint = "LENGTH CONSTRAINT: You must answer in exactly 1 to 3 sentences maximum. Do not write a paragraph."
    elif q_type == "comparative":
        length_constraint = "LENGTH CONSTRAINT: Write between 100 and 300 words."
    elif q_type == "survey":
        length_constraint = "LENGTH CONSTRAINT: Write a comprehensive synthesis between 250 and 600 words. You must cite at least 4 papers."
    else:
        length_constraint = ""

    prompt = f"""You are a research assistant writing an answer to a research question based ONLY on the provided evidence from academic papers.

Rules:
{length_constraint}
- Use ONLY information from the provided evidence
- Cite every claim with the arXiv ID in brackets, e.g., [2210.03629]
- Be specific and detailed
- If evidence is insufficient for some aspect, say so explicitly
- Do not make up or hallucinate information

Question: {question}

Evidence:
{evidence_text}

Write a comprehensive answer with inline citations:"""

    return call_llm(prompt, provider=provider)


def verify_citations(answer: str, evidence: list[dict], provider: str = "ollama") -> str:
    valid_ids = sorted({e["arxiv_id"] for e in evidence})
    valid_ids_string = ", ".join(valid_ids)

    prompt = (
        f"You are a strict citation auditor. Valid paper IDs: {valid_ids_string}. "
        f"Read the following text. If a sentence cites an ID NOT in the valid list, "
        f"completely delete that sentence. If a sentence cites a valid ID, keep it "
        f"exactly as is. Do not change wording. Output only the final cleaned text.\n\n"
        f"Text: {answer}"
    )

    try:
        response = call_llm(prompt, provider=provider, temperature=0.0)
        cleaned = response.strip()
        return cleaned if cleaned else answer
    except Exception:
        return answer
