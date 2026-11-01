"""
src/rag_pipeline.py
-------------------
Core RAG chain using LCEL (LangChain Expression Language) — the modern
LangChain 0.3+ approach that replaces the deprecated RetrievalQA chain.

MMR retriever → Gemini LLM → citation-grounded prompt
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from build_index import load_vectorstore

load_dotenv()

# ── Prompt ─────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a research assistant specializing in machine learning.
Your job is to answer questions accurately using ONLY the research paper excerpts provided below.

Rules:
1. Ground every claim in the provided context — do not hallucinate.
2. Always cite the source document (paper title or filename) for key claims.
3. If the context does not contain enough information to answer, say:
   "I don't have sufficient information in my knowledge base to answer this."
4. Be concise but thorough.

Context (excerpts from research papers):
{context}

Question: {question}

Answer (with inline citations where applicable):"""


# ── RAG Pipeline class ─────────────────────────────────────────────────────────
class RAGPipeline:
    """
    Wraps an LCEL chain + retriever into a single object.
    Exposes an .invoke() method that returns answer + source documents.
    """

    def __init__(self, retriever, chain):
        self.retriever = retriever
        self._chain = chain

    def invoke(self, question: str) -> dict:
        """
        Run retrieval + generation.

        Returns:
            {
                "result": str,                  # LLM answer
                "source_documents": [Document]  # retrieved chunks
            }
        """
        docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        answer = self._chain.invoke({"context": context, "question": question})
        return {"result": answer, "source_documents": docs}


def load_rag_chain(
    db_dir: str = "./chroma_db",
    model: str = "gemini-1.5-flash",
    top_k: int = 5,
    fetch_k: int = 20,
    temperature: float = 0.1,
) -> RAGPipeline:
    """
    Build and return a RAGPipeline backed by:
      - ChromaDB vectorstore with MMR retrieval
      - Google Gemini LLM
      - Citation-grounded prompt

    Args:
        db_dir:      Path to persisted ChromaDB
        model:       Gemini model name
        top_k:       Chunks passed to LLM
        fetch_k:     MMR candidate pool size (higher = more diverse)
        temperature: LLM temperature (low = factual)

    Returns:
        RAGPipeline instance with .invoke(question) method
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Copy .env.example → .env and add your key."
        )

    # ── MMR retriever ──────────────────────────────────────────────────────────
    vectorstore = load_vectorstore(db_dir=db_dir)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k},
    )

    # ── Gemini LLM ────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )

    # ── LCEL chain: prompt | llm | parse ──────────────────────────────────────
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = prompt | llm | StrOutputParser()

    return RAGPipeline(retriever=retriever, chain=chain)


def query(pipeline: RAGPipeline, question: str) -> dict:
    """
    Convenience wrapper — returns structured output for app.py.

    Returns:
        {
            "answer": str,
            "sources": [{"title": ..., "page": ..., "snippet": ...}]
        }
    """
    result = pipeline.invoke(question)
    sources = [
        {
            "title": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:300].strip(),
        }
        for doc in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading RAG pipeline...")
    pipeline = load_rag_chain()

    test_q = "What is retrieval-augmented generation and why is it useful?"
    print(f"\nQuestion: {test_q}\n")
    out = query(pipeline, test_q)
    print(f"Answer:\n{out['answer']}\n")
    print(f"Sources ({len(out['sources'])}):")
    for s in out["sources"]:
        print(f"  • {s['title']} (page {s['page']})")
