"""
PDF Parser & Text Chunker — extracts text from arXiv PDFs and chunks them
with overlapping windows, preserving metadata (arXiv ID, chunk position).
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PAPERS_DIR, METADATA_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  Error reading {pdf_path.name}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page headers/footers patterns (common in arXiv papers)
    text = re.sub(r"arXiv:\d+\.\d+v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d+", "", text)
    # Remove isolated page numbers
    text = re.sub(r"\n\d+\n", "\n", text)
    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Remove hyphenation at line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by approximate word count.
    Uses word-level splitting as a proxy for tokens (~1.3 words per token).
    """
    words = text.split()
    if not words:
        return []

    # Approximate: 1 token ≈ 0.75 words, so chunk_size tokens ≈ chunk_size * 0.75 words
    # But for simplicity we use word count directly (slightly larger chunks)
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def detect_sections(text: str) -> list[dict]:
    """Try to detect section boundaries in the paper text."""
    section_pattern = re.compile(
        r"^(?:\d+\.?\s+)?(Abstract|Introduction|Related Work|Background|"
        r"Methodology|Method|Methods|Approach|Experiments?|Results?|"
        r"Discussion|Conclusion|Conclusions|References|Appendix|"
        r"Evaluation|Analysis|Implementation|Architecture|"
        r"Limitations|Future Work|Acknowledgements?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    sections = []
    matches = list(section_pattern.finditer(text))

    if not matches:
        # No sections detected, treat whole text as one section
        return [{"section": "full_text", "text": text}]

    # Add text before first section
    if matches[0].start() > 100:
        sections.append({
            "section": "header",
            "text": text[: matches[0].start()],
        })

    for i, match in enumerate(matches):
        section_name = match.group(0).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append({
                "section": section_name,
                "text": section_text,
            })

    return sections


def process_single_paper(pdf_path: Path, arxiv_id: str, metadata: dict) -> list[dict]:
    """Process a single PDF into chunks with metadata."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []

    text = clean_text(text)
    sections = detect_sections(text)

    all_chunks = []
    chunk_idx = 0

    for section_info in sections:
        section_chunks = chunk_text(section_info["text"])
        for chunk_text_str in section_chunks:
            all_chunks.append({
                "chunk_id": f"{arxiv_id}_chunk_{chunk_idx}",
                "arxiv_id": arxiv_id,
                "title": metadata.get("title", ""),
                "section": section_info["section"],
                "chunk_index": chunk_idx,
                "text": chunk_text_str,
            })
            chunk_idx += 1

    return all_chunks


def process_all_papers():
    """Process all downloaded PDFs into chunks."""
    # Load metadata
    metadata_path = METADATA_DIR / "papers_metadata.json"
    if not metadata_path.exists():
        print("Error: papers_metadata.json not found. Run the scraper first.")
        return []

    with open(metadata_path) as f:
        papers = json.load(f)

    # Build lookup by arxiv_id
    metadata_lookup = {p["arxiv_id"]: p for p in papers}

    # Find all downloaded PDFs
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")

    all_chunks = []
    failed = 0

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        arxiv_id = pdf_path.stem.replace("_", "/")
        metadata = metadata_lookup.get(arxiv_id, {"title": arxiv_id})

        chunks = process_single_paper(pdf_path, arxiv_id, metadata)
        if chunks:
            all_chunks.extend(chunks)
        else:
            failed += 1

    print(f"\nProcessed {len(pdf_files) - failed}/{len(pdf_files)} PDFs")
    print(f"Total chunks: {len(all_chunks)}")

    # Save chunks
    chunks_path = CHUNKS_DIR / "all_chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Chunks saved to {chunks_path}")

    return all_chunks


if __name__ == "__main__":
    process_all_papers()
