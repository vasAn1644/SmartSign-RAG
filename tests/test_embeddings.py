import os
import shutil
import time
import gc
import pytest
from langchain_core.documents import Document
from src.embeddings import embed_and_store, load_vectorstore

TEST_DB_DIR = "./test_chroma_db"

def test_embedding_pipeline():
    """
    Test the full embedding pipeline:
    - Embed minimal documents
    - Persist to Chroma vectorstore
    - Perform a simple retrieval
    - Cleanup test DB safely (Windows-friendly)
    """

    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR, ignore_errors=True)

    # Fake minimal documents
    docs = [
        Document(
            page_content="Traffic sign: Slippery road. Category: Warning signs.",
            metadata={"type": "image", "category": "Warning signs"}
        ),
        Document(
            page_content="In Germany, warning signs indicate potential danger on the road.",
            metadata={"type": "text", "category": "Rules"}
        )
    ]

    vectorstore = embed_and_store(docs, persist_dir=TEST_DB_DIR)

    assert vectorstore is not None
    assert vectorstore._collection.count() == len(docs)

    results = vectorstore.similarity_search("slippery road", k=1)
    assert len(results) > 0
    assert "Slippery" in results[0].page_content

    del vectorstore              
    gc.collect()                 
    time.sleep(0.5)              
    shutil.rmtree(TEST_DB_DIR, ignore_errors=True)
