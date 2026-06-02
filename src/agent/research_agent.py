"""
Research Agent — ReAct-style loop with toggleable components.

The agent follows this loop:
1. PLAN: Decompose question into sub-questions (optional)
2. RETRIEVE: Search the corpus for relevant passages
3. REFLECT: Evaluate if evidence is sufficient (optional)
4. If insufficient → refine queries and go to step 2 (up to max rounds)
5. SYNTHESIZE: Write cited answer from evidence
6. VERIFY: Check citations are grounded (optional)
"""

import time
import re
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.components import plan, reflect, synthesize, verify_citations, compress_context
from src.indexer.retriever import HybridRetriever
from src.config import MAX_REFLECTION_ROUNDS, FINAL_TOP_K


@dataclass
class AgentConfig:
    """Configuration for which agent components are enabled."""
    use_planner: bool = True
    use_reflector: bool = True
    use_citation_verifier: bool = True
    use_compressor: bool = True
    use_semantic: bool = True
    use_bm25: bool = True
    use_reranker: bool = True
    name: str = "full_agent"

    def describe(self) -> str:
        components = []
        if self.use_planner:
            components.append("planner")
        if self.use_reflector:
            components.append("reflector")
        if self.use_citation_verifier:
            components.append("citation_verifier")
        if self.use_semantic:
            components.append("semantic")
        if self.use_bm25:
            components.append("bm25")
        if self.use_reranker:
            components.append("reranker")
        return f"{self.name}: [{', '.join(components)}]"


# Predefined configurations for ablation study
CONFIGS = {
    "full_agent": AgentConfig(use_compressor=True, name="full_agent"),
    "baseline": AgentConfig(
        use_planner=False,
        use_reflector=False,
        use_citation_verifier=False,
        use_compressor=False,
        use_reranker=True,
        use_bm25=True,
        use_semantic=True,
        name="baseline",
    ),
    "no_planner": AgentConfig(use_planner=False, name="no_planner"),
    "no_reranker": AgentConfig(use_reranker=False, name="no_reranker"),
    "no_reflector": AgentConfig(use_reflector=False, name="no_reflector"),
    "no_hybrid": AgentConfig(use_bm25=False, name="no_hybrid"),
    "no_citation_verifier": AgentConfig(use_citation_verifier=False, name="no_citation_verifier"),
}


@dataclass
class AgentTrace:
    """Records the agent's actions for debugging and the trace view."""
    question: str = ""
    sub_questions: list[str] = field(default_factory=list)
    retrieval_rounds: list[dict] = field(default_factory=list)
    reflections: list[dict] = field(default_factory=list)
    all_evidence: list[dict] = field(default_factory=list)
    raw_answer: str = ""
    final_answer: str = ""
    cited_arxiv_ids: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    latency_seconds: float = 0.0
    config_name: str = ""


class ResearchAgent:
    """Agentic deep research system with toggleable components."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.retriever = HybridRetriever(
            use_semantic=config.use_semantic,
            use_bm25=config.use_bm25,
            use_reranker=config.use_reranker,
        )

    def run(self, question: str) -> tuple[str, AgentTrace]:
        """Run the full agent loop on a question. Returns (answer, trace)."""
        start_time = time.time()
        trace = AgentTrace(question=question, config_name=self.config.name)
        tool_calls = 0

        print(f"\n{'='*60}")
        print(f"Agent [{self.config.name}] — Question: {question[:80]}...")
        print(f"{'='*60}")

        # Step 1: PLAN — decompose into sub-questions
        if self.config.use_planner:
            print("\n[PLAN] Decomposing question...")
            sub_questions = plan(question, provider="groq")
            tool_calls += 1
            trace.sub_questions = sub_questions
            print(f"  Sub-questions: {sub_questions}")
        else:
            sub_questions = [question]
            trace.sub_questions = [question]

        # Step 2-4: RETRIEVE + REFLECT loop
        all_evidence = []
        for round_num in range(1, MAX_REFLECTION_ROUNDS + 1):
            print(f"\n[RETRIEVE] Round {round_num}...")
            round_evidence = []

            queries = sub_questions if round_num == 1 else sub_questions
            for query in queries:
                results = self.retriever.retrieve(query, top_k=FINAL_TOP_K)
                round_evidence.extend(results)
                tool_calls += 1
                print(f"  Query: '{query[:60]}...' → {len(results)} passages")

            # Deduplicate by chunk_id
            seen = {e["chunk_id"] for e in all_evidence}
            new_evidence = [e for e in round_evidence if e["chunk_id"] not in seen]
            all_evidence.extend(new_evidence)

            trace.retrieval_rounds.append({
                "round": round_num,
                "queries": queries,
                "new_passages": len(new_evidence),
                "total_passages": len(all_evidence),
            })

            # Step 3: REFLECT — check if evidence is sufficient
            if self.config.use_reflector and round_num < MAX_REFLECTION_ROUNDS:
                print(f"\n[REFLECT] Evaluating evidence sufficiency...")
                reflection = reflect(question, all_evidence, round_num, provider="groq")
                tool_calls += 1
                trace.reflections.append(reflection)
                print(f"  Sufficient: {reflection.get('sufficient', True)}")
                print(f"  Reasoning: {reflection.get('reasoning', '')[:100]}")

                if reflection.get("sufficient", True):
                    print("  → Evidence sufficient, moving to synthesis.")
                    break

                # Update queries for next round
                refined = reflection.get("refined_queries", [])
                if refined:
                    sub_questions = refined
                    print(f"  → Refined queries: {refined}")
                else:
                    break
            else:
                break

        # Store evidence in trace for evaluation
        trace.all_evidence = all_evidence

        # Step 5: Compress context (optional, one batch LLM call)
        if self.config.use_compressor and all_evidence:
            print(f"\n[COMPRESS] Compressing {len(all_evidence)} passages...")
            for e in all_evidence:
                try:
                    compressed = compress_context(question, [e["text"]])
                    e["compressed_text"] = compressed[0]
                except Exception:
                    pass

        # Step 6: SYNTHESIZE — write cited answer
        print(f"\n[SYNTHESIZE] Writing answer from {len(all_evidence)} passages...")
        answer = synthesize(question, all_evidence, provider="groq")
        tool_calls += 1
        trace.raw_answer = answer

        # Step 6: VERIFY — check citations
        if self.config.use_citation_verifier:
            print("\n[VERIFY] Checking citations...")
            answer = verify_citations(answer, all_evidence, provider="gemini")
            tool_calls += 1

        trace.final_answer = answer
        trace.total_tool_calls = tool_calls
        trace.latency_seconds = time.time() - start_time

        # Extract cited arXiv IDs
        trace.cited_arxiv_ids = list(set(re.findall(r"\[(\d{4}\.\d{4,5})\]", answer)))

        print(f"\n[DONE] {tool_calls} tool calls, {trace.latency_seconds:.1f}s")
        print(f"  Cited papers: {trace.cited_arxiv_ids}")

        return answer, trace
