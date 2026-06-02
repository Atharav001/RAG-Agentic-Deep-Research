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


def plan(question: str, provider: str = "groq") -> list[str]:
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


def reflect(question: str, evidence: list[dict], round_num: int, provider: str = "groq") -> dict:
    """
    Evaluate whether collected evidence is sufficient to answer the question.

    Returns:
        {"sufficient": bool, "reasoning": str, "refined_queries": list[str]}
    """
    evidence_text = "\n\n".join(
        f"[{e['arxiv_id']}] (Section: {e.get('section', 'unknown')})\n{e['text'][:500]}"
        for e in evidence
    )

    prompt = f"""You are a research evidence evaluator. Analyze whether the following evidence is sufficient to answer the research question.

Question: {question}

Evidence collected so far (round {round_num}):
{evidence_text}

Evaluate:
1. Is the evidence sufficient to write a complete, well-cited answer?
2. If not, what specific information is still missing?
3. What refined search queries would help find the missing information?

Return ONLY a JSON object:
{{"sufficient": true/false, "reasoning": "...", "refined_queries": ["query1", "query2"]}}"""

    response = call_llm(prompt, provider=provider)
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return result
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"sufficient": True, "reasoning": "Could not parse reflection", "refined_queries": []}


def compress_context(query: str, evidence_texts: list, client_function=None) -> list:
    result = []
    for text in evidence_texts:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        compressed = " ".join(sentences[:2])
        result.append(compressed)
    return result


def synthesize(question: str, evidence: list[dict], provider: str = "groq") -> str:
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

    prompt = f"""You are a research assistant writing an answer to a research question based ONLY on the provided evidence from academic papers.

Rules:
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


def verify_citations(answer: str, evidence: list[dict], provider: str = "groq") -> str:
    cited_ids_in_evidence = {e["arxiv_id"] for e in evidence}
    citation_pattern = re.compile(r"\[(\d{4}\.\d{4,5})\]")
    sentence_pattern = re.compile(r'[^.!?]*\[\d{4}\.\d{4,5}\][^.!?]*[.!?]')

    cleaned_sentences = []
    for match in sentence_pattern.finditer(answer):
        sentence = match.group()
        ids_in_sentence = citation_pattern.findall(sentence)
        if all(cid in cited_ids_in_evidence for cid in ids_in_sentence):
            cleaned_sentences.append(sentence)

    return " ".join(cleaned_sentences) if cleaned_sentences else answer
