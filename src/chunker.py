"""
Responsible for:
1) Splitting documents into semantic chunks
2) Preserving metadata
3) Preparing data for embedding layer
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
]


def build_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """
    Factory for text splitter
    """
    # REVIEW FIX: extracted splitter creation into separate function
    # for testability and flexibility (different chunk sizes)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len  # REVIEW FIX: explicit length function for clarity
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into chunks
    """
    if not documents:
        # REVIEW FIX: handle empty input gracefully
        return []

    splitter = build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # REVIEW FIX: directly use langchain splitter, removed unnecessary class wrapper
    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    from data_loader import load_all_documents

    docs = load_all_documents()
    chunks = chunk_documents(docs)

    # REVIEW FIX: added debug prints for QA and verification
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")

    print("\nSample chunk:")
    print(chunks[0].page_content)
    print(chunks[0].metadata)
