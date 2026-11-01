"""
src/app.py
----------
Streamlit chat UI for the ML Research RAG Assistant.

Features:
  • Chat interface with conversation history
  • Source document attribution with expandable snippets
  • Sidebar controls for model and retrieval settings
  • Cached chain loading (only loads once per session)

Usage:
    streamlit run src/app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Allow importing sibling modules when running from project root
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from rag_pipeline import load_rag_chain, query as rag_query  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Research RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .metric-badge {
        display: inline-block;
        background: #e9ecef;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
        help="gemini-1.5-flash is fastest; pro is highest quality",
    )

    top_k = st.slider(
        "Retrieved Chunks (k)",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of document chunks passed to the LLM",
    )

    fetch_k = st.slider(
        "MMR Candidate Pool (fetch_k)",
        min_value=10,
        max_value=50,
        value=20,
        help="Larger pool = more diverse retrieval via MMR",
    )

    db_dir = st.text_input(
        "ChromaDB Path",
        value=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
    )

    st.divider()
    st.markdown("""
**About this app**

Queries 100+ ArXiv ML papers using:
- 🧠 **MiniLM-L6-v2** embeddings
- 🗄️ **ChromaDB** vector store
- 🔄 **MMR** retrieval
- 🤖 **Gemini** generation

*Run `python src/collect_papers.py` then `python src/build_index.py` before starting.*
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 ML Research Assistant")
st.caption(
    "Citation-grounded answers from 100+ ArXiv papers · "
    "Powered by LangChain + ChromaDB + Gemini"
)
st.divider()


# ── Load chain (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline... (first load only)")
def get_pipeline(_model: str, _top_k: int, _fetch_k: int, _db_dir: str):
    """Cache the pipeline so it's only loaded once per session."""
    return load_rag_chain(
        db_dir=_db_dir,
        model=_model,
        top_k=_top_k,
        fetch_k=_fetch_k,
    )


try:
    pipeline = get_pipeline(model, top_k, fetch_k, db_dir)
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.info(
        "Run these commands first:\n"
        "```bash\n"
        "python src/collect_papers.py\n"
        "python src/build_index.py\n"
        "```"
    )
    st.stop()
except EnvironmentError as e:
    st.error(f"⚠️ {e}")
    st.info("Add your `GOOGLE_API_KEY` to a `.env` file in the project root.")
    st.stop()


# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources(message["sources"])  # defined below


def _render_sources(sources: list[dict]):
    """Render source documents in an expandable section."""
    with st.expander(f"📄 {len(sources)} source document(s) — click to expand"):
        for i, src in enumerate(sources, 1):
            filename = os.path.basename(src["title"])
            page = src.get("page", "?")
            snippet = src.get("snippet", "")
            st.markdown(
                f'<div class="source-card">'
                f'<strong>Source {i}:</strong> {filename} · Page {page}<br>'
                f'<em>{snippet}...</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Chat input ─────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What is RAG and why does it reduce hallucination?",
    "How does MMR improve retrieval diversity?",
    "What is the difference between BERT and GPT architectures?",
    "How does RLHF align language models with human values?",
    "What are the RAGAS evaluation metrics for RAG?",
]

if not st.session_state.messages:
    st.markdown("**💡 Example questions:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

user_input = st.chat_input("Ask about ML research papers...")

# Handle button-triggered example questions
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching papers and generating answer..."):
            output = rag_query(pipeline, user_input)
            answer = output["answer"]
            sources = output["sources"]

        st.markdown(answer)
        _render_sources(sources)

    # Save assistant message with sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
