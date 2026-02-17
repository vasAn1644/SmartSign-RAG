from typing import List, Dict
from src.embeddings import load_vectorstore


def evaluate_retrieval(question: str, vectorstore=None, k: int = 5) -> List[Dict]:
    """
    Runs semantic search and returns a list of dicts with text, metadata, and source.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=k)

    results = []
    for d in docs:
        # гарантуємо, що завжди словник
        results.append({
            "page_content": d.page_content,
            "metadata": d.metadata,
            "source": d.metadata.get("source") or d.metadata.get("image_url") or ""
        })

    return results
