"""
src/collect_papers.py
---------------------
Downloads ArXiv research papers (PDFs + metadata) for the RAG knowledge base.

Usage:
    python src/collect_papers.py                        # default: 100+ papers
    python src/collect_papers.py --count 50             # fewer papers
    python src/collect_papers.py --query "ViT CLIP"     # custom query
"""

import arxiv
import os
import json
import argparse
import time
import requests
from tqdm import tqdm


# ── Topic queries ──────────────────────────────────────────────────────────────
# We mix 3 topic clusters so the corpus covers RAG, Transformers, and LLMs.
DEFAULT_QUERIES = [
    {
        "query": "retrieval augmented generation RAG large language model",
        "count": 40,
        "label": "RAG & Retrieval"
    },
    {
        "query": "transformer attention mechanism BERT GPT language model",
        "count": 40,
        "label": "Transformers & LLMs"
    },
    {
        "query": "instruction tuning RLHF alignment LLM fine-tuning",
        "count": 25,
        "label": "LLM Alignment & Fine-tuning"
    },
]


def download_papers(
    query: str,
    max_results: int = 40,
    output_dir: str = "./papers",
    delay: float = 1.5,
) -> list[dict]:
    """
    Fetch papers from ArXiv and download their PDFs.

    Args:
        query:       ArXiv search query string
        max_results: Maximum number of papers to download
        output_dir:  Local directory to save PDFs
        delay:       Seconds to wait between downloads (be polite to ArXiv)

    Returns:
        List of metadata dicts for successfully downloaded papers
    """
    os.makedirs(output_dir, exist_ok=True)
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = list(client.results(search))
    print(f"\n  Found {len(papers)} results for: '{query}'")

    metadata = []
    for paper in tqdm(papers, desc="  Downloading", unit="paper"):
        paper_id = paper.get_short_id()
        pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")

        # Skip if already downloaded
        if os.path.exists(pdf_path):
            metadata.append(_build_meta(paper, pdf_path))
            continue

        try:
            # arxiv v2.x removed Result.download_pdf() — download directly via URL
            _download_pdf(paper.pdf_url, pdf_path)
            metadata.append(_build_meta(paper, pdf_path))
            time.sleep(delay)  # polite delay
        except Exception as e:
            print(f"\n  ✗ Failed [{paper_id}]: {e}")

    return metadata


def _download_pdf(url: str, dest_path: str, timeout: int = 30) -> None:
    """Download a PDF from a URL and save it to dest_path."""
    headers = {"User-Agent": "research-rag-bot/1.0 (academic use)"}
    response = requests.get(url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def _build_meta(paper, pdf_path: str) -> dict:
    return {
        "id": paper.get_short_id(),
        "title": paper.title,
        "authors": [a.name for a in paper.authors[:5]],  # first 5 authors
        "published": str(paper.published.date()),
        "abstract": paper.summary,
        "categories": paper.categories,
        "pdf_path": pdf_path,
        "arxiv_url": paper.entry_id,
    }


def run_collection(
    output_dir: str = "./papers",
    queries: list[dict] | None = None,
    single_query: str | None = None,
    single_count: int = 100,
):
    """
    Download papers from multiple topic clusters and save metadata.
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    all_metadata = []

    if single_query:
        # CLI override: single custom query
        all_metadata = download_papers(single_query, single_count, output_dir)
    else:
        # Default: multi-topic corpus
        print("=" * 60)
        print("📥  ArXiv Paper Downloader")
        print("=" * 60)
        for topic in queries:
            print(f"\n📂  Topic: {topic['label']}")
            meta = download_papers(topic["query"], topic["count"], output_dir)
            all_metadata.extend(meta)

    # Deduplicate by paper ID
    seen = set()
    unique_meta = []
    for m in all_metadata:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique_meta.append(m)

    # Save metadata JSON (useful for debugging + inspection)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(unique_meta, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅  Downloaded {len(unique_meta)} unique papers → {output_dir}/")
    print(f"📄  Metadata saved → {meta_path}")
    print("=" * 60)

    return unique_meta


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ArXiv papers for the RAG knowledge base"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom ArXiv search query (overrides default multi-topic download)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of papers to fetch (used with --query)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./papers",
        help="Directory to save PDFs (default: ./papers)",
    )
    args = parser.parse_args()

    run_collection(
        output_dir=args.output_dir,
        single_query=args.query,
        single_count=args.count,
    )
