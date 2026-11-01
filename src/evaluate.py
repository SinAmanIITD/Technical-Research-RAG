"""
src/evaluate.py
---------------
RAGAS evaluation of the RAG pipeline over 20 benchmark questions.
Measures:
  • Faithfulness    — is the answer grounded in the retrieved context?
  • Answer Relevancy — does the answer address the question?
  • Context Recall  — does the retrieved context cover the ground truth?

Usage:
    python src/evaluate.py
    python src/evaluate.py --output eval/eval_results.json
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag_pipeline import load_rag_chain

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
QUESTIONS_PATH = "./eval/test_questions.json"
DEFAULT_OUTPUT = "./eval/eval_results.json"


def run_evaluation(
    questions_path: str = QUESTIONS_PATH,
    output_path: str = DEFAULT_OUTPUT,
    db_dir: str = "./chroma_db",
    model: str = "gemini-1.5-flash",
) -> dict:
    """
    Run RAGAS evaluation over the 20-question benchmark.

    Steps:
        1. Load test questions from JSON
        2. Run each question through the RAG chain to get answers + contexts
        3. Build a HuggingFace Dataset for RAGAS
        4. Evaluate with faithfulness, answer_relevancy, context_recall
        5. Save results to JSON

    Returns:
        Dict with per-metric scores and raw per-question results
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")

    # ── Load test questions ────────────────────────────────────────────────────
    with open(questions_path) as f:
        test_questions = json.load(f)
    print(f"Loaded {len(test_questions)} benchmark questions from {questions_path}")

    # ── Load RAG pipeline ──────────────────────────────────────────────────────
    print("Loading RAG pipeline...")
    pipeline = load_rag_chain(db_dir=db_dir, model=model)

    # ── Generate answers + retrieve contexts ───────────────────────────────────
    print("\nRunning questions through RAG chain...")
    rows = []
    for i, item in enumerate(test_questions, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i:02d}/{len(test_questions)}] {question[:70]}...")
        result = pipeline.invoke(question)

        answer = result["result"]
        contexts = [doc.page_content for doc in result.get("source_documents", [])]

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    # ── Build RAGAS dataset ────────────────────────────────────────────────────
    eval_dataset = Dataset.from_list(rows)

    # ── Configure RAGAS evaluator LLM + embeddings ────────────────────────────
    # RAGAS uses its own LLM for metric computation (not for answering)
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # ── Run RAGAS evaluation ───────────────────────────────────────────────────
    print("\nRunning RAGAS evaluation (this may take a few minutes)...")
    metrics = [faithfulness, answer_relevancy, context_recall]
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # ── Format results ─────────────────────────────────────────────────────────
    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    output = {
        "run_timestamp": datetime.now().isoformat(),
        "model": model,
        "num_questions": len(test_questions),
        "aggregate_scores": {
            "faithfulness": round(scores.get("faithfulness", 0), 4),
            "answer_relevancy": round(scores.get("answer_relevancy", 0), 4),
            "context_recall": round(scores.get("context_recall", 0), 4),
        },
        "per_question_results": result.to_pandas().to_dict(orient="records"),
    }

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📊  RAGAS Evaluation Results")
    print("=" * 50)
    print(f"  Faithfulness:      {output['aggregate_scores']['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {output['aggregate_scores']['answer_relevancy']:.4f}")
    print(f"  Context Recall:    {output['aggregate_scores']['context_recall']:.4f}")
    print(f"\n  Results saved → {output_path}")
    print("=" * 50)

    return output


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the RAG pipeline"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=QUESTIONS_PATH,
        help=f"Path to test questions JSON (default: {QUESTIONS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to save eval results JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
        help="ChromaDB directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini model name (default: gemini-1.5-flash)",
    )
    args = parser.parse_args()

    run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        db_dir=args.db_dir,
        model=args.model,
    )
