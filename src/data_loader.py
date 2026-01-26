"""
Responsible for:
1) Loading image metadata JSON
2) Loading scraped text files
3) Converting raw data into LangChain Document objects
"""

import json
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

IMAGE_JSON_PATH = "data/germany_road_signs.json"
TEXT_DATA_DIR = "data/text_files"


def load_image_json(json_path: str = IMAGE_JSON_PATH) -> List[Document]:
    """
    Load image metadata JSON and convert to LangChain Documents
    """
    json_path = Path(json_path)

    if not json_path.exists():
        # REVIEW FIX: added proper error handling instead of failing silently
        raise FileNotFoundError(f"Image JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Document] = []

    for item in data:
        title = (item.get("title") or "").strip()
        category = (item.get("category") or "").strip()
        image_url = (item.get("image_url") or "").strip()

        # REVIEW FIX: used f-string formatting for performance & readability
        page_content = f"Traffic sign: {title}. Category: {category}."

        metadata = {
            "type": "image",
            "title": title,
            "category": category,
            "image_url": image_url,
            "source": "iamexpat.de"
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    # REVIEW FIX: returns empty list if JSON is empty, avoids silent None
    return documents


def load_text_files(path: str = TEXT_DATA_DIR) -> List[Document]:
    """
    Load scraped text files and convert to LangChain Documents
    """
    path = Path(path)

    if not path.exists():
        # REVIEW FIX: proper exception handling for missing directory
        raise FileNotFoundError(f"Text directory not found: {path}")

    loader = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()

    # REVIEW FIX: enrich metadata after loading, avoids modifying loader code
    for doc in documents:
        doc.metadata["type"] = "text"
        doc.metadata["source"] = "iamexpat.de"

    return documents


def load_all_documents(
    image_json_path: str = IMAGE_JSON_PATH,
    text_dir: str = TEXT_DATA_DIR
) -> List[Document]:
    """
    Load both image and text documents into unified corpus
    """
    # REVIEW FIX: modular loading for testability and reusability
    image_docs = load_image_json(image_json_path)
    text_docs = load_text_files(text_dir)

    all_documents = image_docs + text_docs

    # REVIEW FIX: informative debug output for QA
    print("Loaded documents:")
    print(f"- Image JSON docs: {len(image_docs)}")
    print(f"- Text docs: {len(text_docs)}")
    print(f"- TOTAL: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":
    # REVIEW FIX: added sample print for quick verification during development
    docs = load_all_documents()
    print("\nSample document:")
    print(docs[0].page_content)
    print(docs[0].metadata)
