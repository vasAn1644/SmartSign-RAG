# evaluation/pipeline.py

import json
from pathlib import Path
from typing import Dict, List

from src.rag_pipeline import build_rag_chain, run_rag_query
from src.embeddings import load_vectorstore

from evaluation.metrics import (
    compute_retrieval_metrics,
    compute_answer_metrics,
    compute_rag_quality_score
)
from evaluation.retrieval_eval import evaluate_retrieval

# Paths
EVAL_DATASET_PATH = Path("evaluation/eval_dataset.json")
REPORTS_DIR = Path("evaluation/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation() -> List[Dict]:
    # 1. Завантаження eval dataset
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    eval_dataset = dataset_json.get("samples", [])
    print(f"Loaded {len(eval_dataset)} evaluation examples")

    # 2. Ініціалізація vectorstore та RAG chain
    vectorstore = load_vectorstore()
    rag_chain = build_rag_chain(vectorstore)

    results = []

    # 3. Проходимо по кожному прикладу
    for idx, item in enumerate(eval_dataset, start=1):
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        expected_sources = item.get("expected_sources", [])

        # Retrieval
        retrieved_docs = evaluate_retrieval(question, vectorstore=vectorstore)

        # RAG answer
        rag_answer = run_rag_query(rag_chain, question)

        # Метрики
        retrieval_score = compute_retrieval_metrics(retrieved_docs, expected_sources)
        answer_score = compute_answer_metrics(rag_answer, expected_answer)
        rag_quality = compute_rag_quality_score(retrieval_score, answer_score)

        results.append({
            "id": item.get("id", idx),
            "question": question,
            "expected_answer": expected_answer,
            "rag_answer": rag_answer,
            "retrieved_docs": retrieved_docs,
            "retrieval_score": retrieval_score,
            "answer_score": answer_score,
            "rag_quality_score": rag_quality
        })

        print(f"[{idx}/{len(eval_dataset)}] Question processed")

    # 4. Збереження результатів
    report_path = REPORTS_DIR / "eval_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation finished. {len(results)} samples processed.")
    print(f"Results saved to {report_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
