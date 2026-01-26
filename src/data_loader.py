"""
Responsible for:
1) Loading image metadata JSON
2) Loading scraped text files
3) Loading PDF files (e.g., English StVO)
4) Converting raw data into LangChain Document objects
"""

import json
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

IMAGE_JSON_PATH = "data/germany_road_signs.json"
TEXT_DATA_DIR = "data/text_files"
PDF_DATA_DIR = "data/pdf_files"  


def load_image_json(json_path: str = IMAGE_JSON_PATH) -> List[Document]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Image JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Document] = []
    for item in data:
        title = (item.get("title") or "").strip()
        category = (item.get("category") or "").strip()
        image_url = (item.get("image_url") or "").strip()
        page_content = f"Traffic sign: {title}. Category: {category}."
        metadata = {
            "type": "image",
            "title": title,
            "category": category,
            "image_url": image_url,
            "source": "iamexpat.de"
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def load_text_files(path: str = TEXT_DATA_DIR) -> List[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text directory not found: {path}")

    loader = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    documents = loader.load()
    for doc in documents:
        doc.metadata["type"] = "text"
        doc.metadata["source"] = "iamexpat.de"
    return documents


def load_pdf_files(path: str = PDF_DATA_DIR) -> List[Document]:
    """
    Load PDF files (e.g., English StVO) and convert to LangChain Documents
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF directory not found: {path}")

    documents: List[Document] = []
    for pdf_file in path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        pdf_docs = loader.load()
        # додатково додаємо metadata
        for doc in pdf_docs:
            doc.metadata["type"] = "pdf"
            doc.metadata["source"] = pdf_file.name
        documents.extend(pdf_docs)
    return documents


def load_all_documents(
    image_json_path: str = IMAGE_JSON_PATH,
    text_dir: str = TEXT_DATA_DIR,
    pdf_dir: str = PDF_DATA_DIR
) -> List[Document]:
    """
    Load images, text files, and PDFs into unified corpus
    """
    image_docs = load_image_json(image_json_path)
    text_docs = load_text_files(text_dir)
    pdf_docs = load_pdf_files(pdf_dir)

    all_documents = image_docs + text_docs + pdf_docs

    print("Loaded documents:")
    print(f"- Image JSON docs: {len(image_docs)}")
    print(f"- Text docs: {len(text_docs)}")
    print(f"- PDF docs: {len(pdf_docs)}")
    print(f"- TOTAL: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":
    docs = load_all_documents()
    print("\nSample document:")
    print(docs[0].page_content)
    print(docs[0].metadata)
