# evaluation/metrics.py

from typing import List, Dict
import re


def normalize(text: str) -> str:
    """Simple text normalization"""
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower()).strip()


def compute_retrieval_metrics(retrieved_docs: List[Dict], expected_sources: List[str]) -> float:
    """
    Measures how well retriever fetched correct sources.
    Score: 0..1
    """
    if not expected_sources:
        return 1.0  # no ground truth → neutral score

    retrieved_sources = [
        doc.get("source") or doc.get("image_url") or ""
        for doc in retrieved_docs
    ]

    hits = 0
    for src in expected_sources:
        for r in retrieved_sources:
            if src in r:
                hits += 1
                break

    return hits / len(expected_sources)


def compute_answer_metrics(answer: str, expected_answer: str) -> float:
    """
    Simple semantic overlap metric (baseline version)
    """
    if not expected_answer:
        return 1.0

    a = normalize(answer)
    e = normalize(expected_answer)

    a_tokens = set(a.split())
    e_tokens = set(e.split())

    if not e_tokens:
        return 0.0

    overlap = a_tokens.intersection(e_tokens)
    return len(overlap) / len(e_tokens)


def compute_rag_quality_score(retrieval_score: float, answer_score: float) -> float:
    """
    Global RAG quality score
    """
    return round((0.5 * retrieval_score + 0.5 * answer_score), 3)
