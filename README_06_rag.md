# 🔍 Technical Research RAG — AI-Powered Paper Assistant

> Build a Retrieval-Augmented Generation (RAG) system to query and synthesize knowledge from ArXiv ML research papers.
> **CV2** — LLM + RAG + Vector DB + Deployment

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-purple)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)](https://ollama.ai)

---

## 📌 Problem Statement

Researchers spend hours reading papers to stay current. A RAG system that ingests a curated corpus of ML papers and answers questions with cited sources can compress days of reading into minutes. Unlike a generic LLM, RAG grounds answers in specific documents, reducing hallucination and enabling attribution.

**Goal:** Build a production-quality RAG pipeline that answers ML research questions from ArXiv papers, with evaluation using RAGAS and a deployable Streamlit UI.

---

## 🏗️ Architecture

```
ArXiv API / PDFs
       ↓
   PDF Loader (LangChain PyPDFLoader)
       ↓
   Text Splitter (RecursiveCharacterTextSplitter, chunk_size=1000)
       ↓
   Embeddings (sentence-transformers/all-MiniLM-L6-v2)
       ↓
   Vector Store (ChromaDB, persisted locally)
       ↓
User Query → Retriever (top-k=5) → LLM (Llama3/Gemini/OpenAI)
                                          ↓
                              Answer + Source Documents
                                          ↓
                            RAGAS Evaluation (faithfulness, relevancy)
```

---

## 📊 Data Sources

### Option A: ArXiv API (Live Papers — Recommended)
| Field | Value |
|---|---|
| **Name** | ArXiv ML Papers (Live) |
| **Source** | Python `arxiv` library — no account needed |
| **Volume** | Fetch 100-500 papers programmatically |
| **Cost** | 100% free |

```python
pip install arxiv
```

### Option B: ArXiv Metadata Dump (2M+ Papers)
| Field | Value |
|---|---|
| **Source** | [Kaggle — Cornell ArXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv) |
| **Size** | 2M+ paper metadata (titles, abstracts) — no PDFs |
| **Use case** | If you want classification/search, not Q&A over full text |

### Option C: Custom PDF Collection
- Download papers from [papers.huggingface.co](https://papers.huggingface.co) manually
- More impressive if you pick a niche (e.g., "Vision Transformers", "LLM Alignment")
- Better demonstrates domain knowledge

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM (free, local) | Ollama + Llama3.2 | No API cost, no rate limits |
| LLM (paid, faster) | OpenAI GPT-4o / Gemini | Better quality, needs API key |
| Embeddings | sentence-transformers | Free, runs locally, fast |
| Vector DB | ChromaDB | Simple, local, persistent |
| RAG Framework | LangChain | Standard, well-documented |
| UI | Streamlit | Fast to build, easy to deploy |
| Evaluation | RAGAS | Industry standard RAG metrics |
| PDF Parsing | PyPDF2 / pypdf | Free, handles most research PDFs |

---

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install Ollama (free local LLM — macOS)
brew install ollama
ollama serve &                    # Start Ollama server
ollama pull llama3.2              # Download Llama 3.2 (2GB)

# 2. Install Python dependencies
git clone https://github.com/YOUR_USERNAME/research-rag
cd research-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Step 1: Collect papers (downloads PDFs to ./papers/)
python src/collect_papers.py --query "attention mechanism transformer" --count 50

# Step 2: Build vector index
python src/build_index.py

# Step 3: Launch the app
streamlit run src/app.py
```

---

## 📁 Project Structure

```
research-rag/
├── papers/                          # Downloaded PDFs (git ignored, ~500MB)
├── chroma_db/                       # Persisted vector store (git ignored)
├── src/
│   ├── __init__.py
│   ├── collect_papers.py            # ArXiv API fetcher + PDF downloader
│   ├── build_index.py               # Document loading → chunking → embedding
│   ├── rag_pipeline.py              # Core RAG chain (retriever + LLM)
│   ├── evaluate.py                  # RAGAS evaluation script
│   └── app.py                       # Streamlit chat UI
├── eval/
│   ├── test_questions.json          # 20 test Q&A pairs for RAGAS
│   └── eval_results.json            # RAGAS scores (commit this!)
├── notebooks/
│   └── 01_rag_experiments.ipynb    # Prototyping + experiments
├── Dockerfile                       # Container for deployment
├── requirements.txt
└── README.md
```

---

## 🔬 Implementation

### Step 1: Collect Papers
```python
# src/collect_papers.py
import arxiv
import os

def download_papers(query: str, max_results: int = 100, output_dir: str = "./papers"):
    os.makedirs(output_dir, exist_ok=True)
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = list(client.results(search))
    print(f"Found {len(papers)} papers. Downloading PDFs...")

    metadata = []
    for paper in papers:
        try:
            paper.download_pdf(dirpath=output_dir, filename=f"{paper.get_short_id()}.pdf")
            metadata.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published.date()),
                "abstract": paper.summary,
                "categories": paper.categories
            })
            print(f"  ✓ {paper.title[:60]}...")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return metadata

if __name__ == "__main__":
    metadata = download_papers(
        query="large language model RAG retrieval augmented generation",
        max_results=100
    )
    # Also useful queries:
    # "vision transformer image classification"
    # "reinforcement learning from human feedback RLHF"
    # "neural network training optimization"
```

### Step 2: Build Vector Index
```python
# src/build_index.py
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def build_index(papers_dir: str = "./papers", db_dir: str = "./chroma_db"):
    # 1. Load PDFs
    print("Loading PDFs...")
    loader = DirectoryLoader(
        papers_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")

    # 2. Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed + store
    print("Creating embeddings (this takes a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"✓ Vector store saved to {db_dir}")
    return vectorstore
```

### Step 3: RAG Chain
```python
# src/rag_pipeline.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama  # or from langchain.chat_models import ChatOpenAI

PROMPT_TEMPLATE = """You are a research assistant specialized in machine learning.
Use the following research paper excerpts to answer the question.
Always cite the source document title in your answer.
If you cannot find the answer in the provided context, say "I don't have enough information in my knowledge base."

Context:
{context}

Question: {question}

Answer (with citations):"""

def load_rag_chain(db_dir: str = "./chroma_db", model: str = "llama3.2"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",              # Maximal Marginal Relevance — reduces redundancy
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    llm = Ollama(model=model, temperature=0.1)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                             input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain
```

### Step 4: Streamlit App
```python
# src/app.py
import streamlit as st
from rag_pipeline import load_rag_chain

st.set_page_config(page_title="Research RAG", page_icon="🔍", layout="wide")
st.title("🔍 ML Research Assistant")
st.caption("Powered by LangChain + ChromaDB + Llama3 — answers grounded in ArXiv papers")

@st.cache_resource
def get_chain():
    return load_rag_chain()

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about ML research..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..."):
            result = chain({"query": query})
            answer = result["result"]
            sources = result["source_documents"]

        st.markdown(answer)

        with st.expander(f"📄 {len(sources)} source documents"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                st.text(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
```

### Step 5: RAGAS Evaluation
```python
# src/evaluate.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

test_questions = [
    {"question": "What is the attention mechanism in transformers?",
     "ground_truth": "Attention allows the model to focus on relevant parts..."},
    # Add 20 Q&A pairs
]

# Run evaluation
eval_dataset = Dataset.from_list([{
    "question": q["question"],
    "answer": chain({"query": q["question"]})["result"],
    "contexts": [d.page_content for d in chain({"query": q["question"]})["source_documents"]],
    "ground_truth": q["ground_truth"]
} for q in test_questions])

result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_recall])
print(result)
# faithfulness: 0.84, answer_relevancy: 0.79, context_recall: 0.71
```

---

## 📈 Results

> *Fill this in after running the project*

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 0.84 | Answer supported by retrieved context |
| Answer Relevancy | 0.79 | Answer addresses the question |
| Context Recall | 0.71 | Retrieved docs contain ground truth |
| Avg Latency | ~4.2s | Time to generate response (local Llama3) |

**Example Query:**
> "What are the key differences between GPT and BERT architectures?"

**System Response:**
> "Based on the retrieved papers: GPT uses a unidirectional (left-to-right) decoder-only architecture trained with causal language modeling... [Source: attention_is_all_you_need.pdf, page 3]"

---

## 🐳 Deployment (Docker)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t research-rag .
docker run -p 8501:8501 research-rag
```

**Free Deployment Options:**
- [HuggingFace Spaces](https://huggingface.co/spaces) — Streamlit apps, free tier
- [Streamlit Community Cloud](https://streamlit.io/cloud) — Free for public repos
- [Render.com](https://render.com) — Free tier with sleep on inactivity

---

## 📋 Requirements

```
langchain>=0.2
langchain-community>=0.2
chromadb>=0.4
sentence-transformers>=2.2
arxiv>=2.0
pypdf>=3.0
streamlit>=1.28
ragas>=0.1
datasets>=2.14
ollama>=0.1             # Python client for Ollama
openai>=1.0             # Optional: if using OpenAI
```

---

## 🧠 Key Learnings

- [ ] MMR retrieval reduces redundant context vs. similarity search
- [ ] `chunk_overlap=200` preserves context across chunk boundaries
- [ ] Local LLMs (Ollama) = zero API cost; trade-off is speed vs. GPT-4o quality
- [ ] RAGAS evaluates 3 dimensions: faithfulness (no hallucination), relevancy, recall
- [ ] Caching the chain with `@st.cache_resource` avoids reloading on every interaction

---

## 📚 References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [ArXiv Python Library](https://github.com/lukasschwab/arxiv.py)
- [sentence-transformers](https://www.sbert.net/)
- [RAG: Retrieval-Augmented Generation (Original Paper)](https://arxiv.org/abs/2005.11401)
