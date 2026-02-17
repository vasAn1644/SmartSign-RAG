"""
Responsible for:
1) Creating embeddings
2) Building vector store
3) Persisting embeddings
4) Providing retriever interface for RAG
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
PERSIST_DIR = "./chroma_db"


def build_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME
) -> HuggingFaceEmbeddings:
    """
    Factory for embedding model
    """
    # REVIEW FIX: extracted model creation for reusability & testability
    # REVIEW FIX: replaced deprecated HuggingFaceEmbeddings usage with recommended alternative
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(
    documents: List[Document],
    persist_dir: str = PERSIST_DIR,
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """
    Create and persist vector store
    """
    if embedding_model is None:
        embedding_model = build_embedding_model()

    os.makedirs(persist_dir, exist_ok=True)

    # REVIEW FIX: added explicit persist_directory & embedding_model
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    return vectorstore


def load_vectorstore(
    persist_dir: str = PERSIST_DIR,
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> Chroma:
    """
    Load existing vector store
    """
    if embedding_model is None:
        embedding_model = build_embedding_model()

    if not os.path.exists(persist_dir):
        # REVIEW FIX: raise error if DB doesn't exist to avoid silent failure
        raise FileNotFoundError(f"Vector DB not found: {persist_dir}")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    return vectorstore


def build_retriever(
    vectorstore: Chroma,
    k: int = 5,
    search_type: str = "similarity"
):
    """
    Build retriever interface
    """
    # REVIEW FIX: made k & search_type configurable for flexibility
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

    return retriever


def embed_and_store(
    chunks: List[Document],
    persist_dir: str = PERSIST_DIR
) -> Chroma:
    """
    Full pipeline: embedding + persistence
    """
    embedding_model = build_embedding_model()
    vectorstore = build_vectorstore(
        documents=chunks,
        persist_dir=persist_dir,
        embedding_model=embedding_model
    )

    # REVIEW FIX: debug prints for QA / verification
    print("Vector store created")
    print("Stored embeddings:", vectorstore._collection.count())

    return vectorstore


# REVIEW FIX: Added convenience wrapper for backward compatibility with main.py
def create_vectorstore(documents):
    """
    Wrapper to match previous main.py API
    """
    vectorstore = build_vectorstore(documents)
    return vectorstore


if __name__ == "__main__":
    from src.data_loader import load_all_documents
    from src.chunker import chunk_documents

    print("Loading documents...")
    docs = load_all_documents()

    print("Chunking documents...")
    chunks = chunk_documents(docs)

    print(f"Total chunks: {len(chunks)}")

    print("Embedding & storing...")
    vs = embed_and_store(chunks)

    print("\nTest retrieval:")
    results = vs.similarity_search("slippery road", k=5)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}")
        print("TEXT:", doc.page_content)
        print("METADATA:", doc.metadata)
