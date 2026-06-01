"""
arXiv Scraper — collects papers on LLM agents from cs.CL/cs.AI/cs.LG (Jan 2024–Apr 2026).

Uses the free arXiv API (Atom XML). Rate-limited to be polite.
Saves metadata as JSON and downloads PDFs.
"""

import json
import time
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    ARXIV_API_URL,
    ARXIV_BATCH_SIZE,
    ARXIV_RATE_LIMIT_SECONDS,
    PAPERS_DIR,
    METADATA_DIR,
)

# Atom XML namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# Keywords to search for in title/abstract (focused set to target 500-700 papers)
SEARCH_KEYWORDS = [
    '"LLM agent"',
    '"language model agent"',
    '"agentic RAG"',
    '"tool use" AND "language model"',
    '"agent benchmark" AND "LLM"',
    '"computer use agent"',
    '"multi-agent" AND "LLM"',
    '"ReAct" AND "agent"',
    '"function calling" AND "agent"',
    '"agent framework" AND "language model"',
]

# Categories to search in
CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]

# Date range
DATE_START = "20240101"
DATE_END = "20260430"


def build_query(keyword: str, category: str) -> str:
    """Build an arXiv API search query string.

    Handles compound keywords like '"tool use" AND "language model"'
    by scoping each term to ti: and abs: separately.
    """
    cat_part = f"cat:{category}"
    date_part = f"submittedDate:[{DATE_START}0000+TO+{DATE_END}2359]"

    if " AND " in keyword:
        # Compound keyword: scope each part to ti: and abs: individually
        parts = [p.strip() for p in keyword.split(" AND ")]
        ti_parts = "+AND+".join(f"ti:{p}" for p in parts)
        abs_parts = "+AND+".join(f"abs:{p}" for p in parts)
        keyword_part = f"(%28{ti_parts}%29+OR+%28{abs_parts}%29)"
    else:
        keyword_part = f"(ti:{keyword}+OR+abs:{keyword})"

    query = f"{keyword_part}+AND+{cat_part}+AND+{date_part}"
    return query


def parse_entry(entry: ET.Element) -> dict | None:
    """Parse a single Atom entry into a metadata dict."""
    try:
        arxiv_id_raw = entry.find(f"{ATOM_NS}id").text
        # Extract just the ID (e.g., "2210.03629" from "http://arxiv.org/abs/2210.03629v1")
        arxiv_id = arxiv_id_raw.split("/abs/")[-1]
        # Remove version suffix
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title = entry.find(f"{ATOM_NS}title").text.strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)

        abstract = entry.find(f"{ATOM_NS}summary").text.strip().replace("\n", " ")
        abstract = re.sub(r"\s+", " ", abstract)

        published = entry.find(f"{ATOM_NS}published").text
        updated = entry.find(f"{ATOM_NS}updated").text

        authors = []
        for author_elem in entry.findall(f"{ATOM_NS}author"):
            name = author_elem.find(f"{ATOM_NS}name").text
            authors.append(name)

        categories = []
        for cat_elem in entry.findall(f"{ATOM_NS}category"):
            categories.append(cat_elem.get("term"))

        # Get PDF link
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url:
            pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "published": published,
            "updated": updated,
            "pdf_url": pdf_url,
        }
    except Exception as e:
        print(f"  Error parsing entry: {e}")
        return None


def fetch_papers_for_query(query: str, max_results: int = 200) -> list[dict]:
    """Fetch papers from arXiv API for a single query, with pagination."""
    papers = []
    start = 0

    while start < max_results:
        batch_size = min(ARXIV_BATCH_SIZE, max_results - start)
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params, safe=':+[]')}"

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            break

        root = ET.fromstring(resp.text)

        # Check total results
        total_elem = root.find(f"{OPENSEARCH_NS}totalResults")
        if total_elem is not None:
            total = int(total_elem.text)
            max_results = min(max_results, total)

        entries = root.findall(f"{ATOM_NS}entry")
        if not entries:
            break

        for entry in entries:
            paper = parse_entry(entry)
            if paper:
                papers.append(paper)

        start += batch_size
        time.sleep(ARXIV_RATE_LIMIT_SECONDS)

    return papers


def scrape_all_papers() -> dict[str, dict]:
    """Scrape papers across all keyword-category combinations. Deduplicate by arXiv ID."""
    all_papers: dict[str, dict] = {}

    total_queries = len(SEARCH_KEYWORDS) * len(CATEGORIES)
    print(f"Running {total_queries} queries ({len(SEARCH_KEYWORDS)} keywords x {len(CATEGORIES)} categories)...")

    pbar = tqdm(total=total_queries, desc="Scraping arXiv")
    for keyword in SEARCH_KEYWORDS:
        for category in CATEGORIES:
            query = build_query(keyword, category)
            papers = fetch_papers_for_query(query, max_results=50)

            new_count = 0
            for p in papers:
                if p["arxiv_id"] not in all_papers:
                    all_papers[p["arxiv_id"]] = p
                    new_count += 1

            pbar.set_postfix(total=len(all_papers), new=new_count)
            pbar.update(1)
            time.sleep(ARXIV_RATE_LIMIT_SECONDS)

    pbar.close()
    print(f"\nTotal unique papers collected: {len(all_papers)}")
    return all_papers


def save_metadata(papers: dict[str, dict]):
    """Save all paper metadata to a single JSON file."""
    output_path = METADATA_DIR / "papers_metadata.json"
    with open(output_path, "w") as f:
        json.dump(list(papers.values()), f, indent=2)
    print(f"Metadata saved to {output_path} ({len(papers)} papers)")


def download_pdfs(papers: dict[str, dict], max_papers: int | None = None):
    """Download PDFs for all papers."""
    paper_list = list(papers.values())
    if max_papers:
        paper_list = paper_list[:max_papers]

    downloaded = 0
    skipped = 0

    for paper in tqdm(paper_list, desc="Downloading PDFs"):
        pdf_path = PAPERS_DIR / f"{paper['arxiv_id'].replace('/', '_')}.pdf"
        if pdf_path.exists():
            skipped += 1
            continue

        try:
            resp = requests.get(paper["pdf_url"], timeout=60)
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
            downloaded += 1
            time.sleep(1)  # rate limit PDF downloads
        except requests.RequestException as e:
            print(f"  Failed to download {paper['arxiv_id']}: {e}")

    print(f"Downloaded: {downloaded}, Skipped (already exist): {skipped}")


def filter_relevant_papers(papers: dict[str, dict]) -> dict[str, dict]:
    """Post-filter papers to ensure they're actually about LLM agents.

    Stricter filter: must mention 'agent' or 'agentic' AND at least one
    LLM-related term to avoid generic AI/ML papers.
    """
    # Score-based relevance filter to get ~500-700 high-quality papers
    RELEVANCE_THRESHOLD = 10

    agent_phrases = ["agent", "agentic"]
    topic_terms = [
        "tool use", "tool-use", "retrieval augmented", "planning",
        "benchmark", "react", "agentic rag", "multi-agent",
        "function call", "reasoning", "memory", "reflexion", "self-rag",
    ]

    scored = []
    for arxiv_id, paper in papers.items():
        title = paper["title"].lower()
        abstract = paper["abstract"].lower()

        # Must have "agent" or "agentic" in the title
        if not any(t in title for t in agent_phrases):
            continue

        score = 0
        # LLM mentioned in title (strong signal)
        if any(t in title for t in ["llm", "language model"]):
            score += 5
        # Specific "LLM agent" phrase in title
        if any(t in title for t in ["llm agent", "language model agent", "llm-based agent"]):
            score += 3
        # LLM mentioned in abstract
        if any(t in abstract for t in ["llm", "large language model", "language model"]):
            score += 2
        # Topical keywords in abstract
        for term in topic_terms:
            if term in abstract:
                score += 1

        scored.append((score, arxiv_id, paper))

    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = {}
    for score, arxiv_id, paper in scored:
        if score >= RELEVANCE_THRESHOLD:
            filtered[arxiv_id] = paper

    print(f"After relevance filtering: {len(filtered)}/{len(papers)} papers kept")
    return filtered


def main():
    """Main scraping pipeline.

    If metadata already exists (from a prior run), skips the API query
    and just re-filters + downloads missing PDFs.
    """
    print("=" * 60)
    print("arXiv Scraper — LLM Agent Papers (Jan 2024 – Apr 2026)")
    print("=" * 60)

    raw_metadata_path = METADATA_DIR / "papers_raw_metadata.json"
    filtered_metadata_path = METADATA_DIR / "papers_metadata.json"

    # Step 1: Scrape metadata (or load existing)
    if raw_metadata_path.exists():
        print(f"\n[1/4] Loading existing raw metadata from {raw_metadata_path}...")
        with open(raw_metadata_path) as f:
            raw_list = json.load(f)
        papers = {p["arxiv_id"]: p for p in raw_list}
        print(f"  Loaded {len(papers)} papers (skipping API queries)")
    else:
        print("\n[1/4] Scraping paper metadata from arXiv API...")
        papers = scrape_all_papers()
        # Save raw metadata before filtering
        with open(raw_metadata_path, "w") as f:
            json.dump(list(papers.values()), f, indent=2)
        print(f"  Raw metadata saved to {raw_metadata_path}")

    # Step 2: Filter for relevance
    print("\n[2/4] Filtering for relevance...")
    papers = filter_relevant_papers(papers)

    # Step 3: Save filtered metadata
    print("\n[3/4] Saving filtered metadata...")
    save_metadata(papers)

    # Step 4: Download PDFs
    print("\n[4/4] Downloading PDFs...")
    download_pdfs(papers)

    print(f"\nDone! {len(papers)} papers collected.")
    return papers


if __name__ == "__main__":
    main()
