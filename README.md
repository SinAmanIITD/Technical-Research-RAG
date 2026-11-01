# 🔍 Technical Research RAG — AI-Powered Paper Assistant

> Retrieval-Augmented Generation system that answers ML research questions from 100+ ArXiv papers with citation-grounded responses.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-blue)](https://ai.google.dev)

---

## 📌 Problem Statement

Researchers spend hours reading papers to stay current. This RAG system ingests a curated corpus of ML papers and answers questions with cited sources — compressing days of reading into seconds. Unlike a generic LLM, RAG grounds answers in specific documents, reducing hallucination and enabling source attribution.

---

## 🏗️ Architecture

```
ArXiv API (100+ papers)
       ↓
   PDF Loader (LangChain PyPDFLoader)
       ↓
   Text Splitter (RecursiveCharacterTextSplitter, chunk_size=1000, overlap=200)
       ↓
   Embeddings (sentence-transformers/all-MiniLM-L6-v2)
       ↓
   Vector Store (ChromaDB, persisted locally)
       ↓
User Query → MMR Retriever (k=5, fetch_k=20) → Gemini 1.5 Flash
                                                      ↓
                                          Answer + Source Documents
                                                      ↓
                                        RAGAS Evaluation (20 benchmark Q&A)
```

---

## 📊 Results

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 0.84 | Answer supported by retrieved context |
| Answer Relevancy | 0.79 | Answer addresses the question |
| Context Recall | 0.71 | Retrieved docs contain ground truth |

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Gemini 1.5 Flash | Fast, free tier, no local GPU needed |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, local, high quality |
| Vector DB | ChromaDB | Simple, local, persistent |
| Retrieval | MMR (Maximal Marginal Relevance) | Reduces redundant context |
| RAG Framework | LangChain | Standard, well-documented |
| UI | Streamlit | Fast to build, chat interface |
| Evaluation | RAGAS | Industry standard RAG metrics |
| Deployment | Docker + Docker Compose | Reproducible, portable |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/research-rag
cd research-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API Key

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
# Get a free key at: https://aistudio.google.com/app/apikey
```

### 3. Download Papers (~100 ArXiv PDFs)

```bash
python src/collect_papers.py
# Downloads ~105 papers across 3 topics:
#   • RAG & Retrieval (40 papers)
#   • Transformers & LLMs (40 papers)
#   • LLM Alignment & Fine-tuning (25 papers)
```

### 4. Build Vector Index

```bash
python src/build_index.py
# Loads PDFs → chunks → embeds with MiniLM → saves to ./chroma_db/
# Takes ~5-10 minutes for 100 papers
```

### 5. Launch the App

```bash
streamlit run src/app.py
# Open http://localhost:8501
```

### 6. Run RAGAS Evaluation (Optional)

```bash
python src/evaluate.py
# Runs 20 benchmark questions and saves scores to eval/eval_results.json
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
│   ├── build_index.py               # PDF loading → chunking → embedding
│   ├── rag_pipeline.py              # MMR retriever + Gemini RAG chain
│   ├── evaluate.py                  # RAGAS evaluation (20 questions)
│   └── app.py                       # Streamlit chat UI
├── eval/
│   ├── test_questions.json          # 20 benchmark Q&A pairs
│   └── eval_results.json            # RAGAS scores (committed)
├── notebooks/
│   └── 01_rag_experiments.ipynb    # Prototyping notebook
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker compose up --build

# Or manually
docker build -t research-rag .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key research-rag
```

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|---|---|
| **MMR retrieval** | Balances relevance + diversity; reduces redundant chunks in context |
| `chunk_overlap=200` | Preserves context across chunk boundaries |
| **Gemini Flash** | Zero GPU cost, free tier, 1M token context window |
| **MiniLM-L6-v2** | Runs fully locally; fast inference; strong semantic retrieval quality |
| **RAGAS evaluation** | 3-dimensional quality signal: faithfulness, relevancy, recall |
| **Stuff chain** | Simple and effective for k=5 chunks well within Gemini's context limit |

---

## 📚 References

- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [sentence-transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Google Gemini API](https://ai.google.dev/)
