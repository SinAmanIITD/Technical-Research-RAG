"""
src/build_index.py
------------------
Loads PDFs from ./papers/, splits them into chunks, embeds with MiniLM,
and persists to a ChromaDB vector store.

Usage:
    python src/build_index.py
    python src/build_index.py --papers-dir ./papers --db-dir ./chroma_db
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "research_papers"


def build_index(
    papers_dir: str = "./papers",
    db_dir: str = "./chroma_db",
) -> Chroma:
    """
    Full pipeline: PDF loading → text splitting → embedding → ChromaDB storage.

    Args:
        papers_dir: Directory containing downloaded PDF files
        db_dir:     Directory to persist the ChromaDB vector store

    Returns:
        Loaded Chroma vectorstore instance
    """
    # ── Step 1: Load PDFs ──────────────────────────────────────────────────────
    print("=" * 60)
    print("📂  Step 1: Loading PDFs")
    print("=" * 60)

    if not os.path.exists(papers_dir) or not any(
        f.endswith(".pdf") for f in os.listdir(papers_dir)
    ):
        raise FileNotFoundError(
            f"No PDFs found in '{papers_dir}'.\n"
            "Run: python src/collect_papers.py  first."
        )

    loader = DirectoryLoader(
        papers_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    print(f"\n✓  Loaded {len(documents)} pages from PDFs\n")

    # ── Step 2: Chunk documents ────────────────────────────────────────────────
    print("=" * 60)
    print("✂️   Step 2: Splitting into chunks")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✓  Created {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    # ── Step 3: Embed & store in ChromaDB ─────────────────────────────────────
    print("=" * 60)
    print("🧠  Step 3: Embedding chunks with MiniLM-L6-v2")
    print("    (This may take several minutes for 100+ papers)")
    print("=" * 60)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build in batches to avoid memory issues with large corpora
    print(f"\n📦  Storing {len(chunks)} chunks in ChromaDB → {db_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=COLLECTION_NAME,
    )

    print(f"\n✅  Vector store saved to '{db_dir}'")
    print(f"    Collection: '{COLLECTION_NAME}'")
    print(f"    Total vectors: {vectorstore._collection.count()}")
    print("=" * 60)

    return vectorstore


def load_vectorstore(db_dir: str = "./chroma_db") -> Chroma:
    """
    Load an already-built ChromaDB vector store from disk.

    Args:
        db_dir: Path to the persisted ChromaDB directory

    Returns:
        Loaded Chroma vectorstore
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector index from downloaded PDFs"
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=os.getenv("PAPERS_DIR", "./papers"),
        help="Directory containing PDF files (default: ./papers)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
        help="Directory to persist ChromaDB (default: ./chroma_db)",
    )
    args = parser.parse_args()

    build_index(papers_dir=args.papers_dir, db_dir=args.db_dir)
