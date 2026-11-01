# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ───────────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ─────────────────────────────────────────────────
# libgomp1 is required by sentence-transformers
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ─────────────────────────────────────────────────
# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY eval/ ./eval/

# ── Copy pre-built vector store (must be built before docker build) ─────────────
# If chroma_db/ doesn't exist locally, build it first:
#   python src/collect_papers.py && python src/build_index.py
COPY chroma_db/ ./chroma_db/

# ── Expose Streamlit port ────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entry point ──────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
