#  SmartSign RAG: Multimodal Traffic Assistant

**SmartSign RAG** is a Multimodal Retrieval-Augmented Generation (RAG) application designed to help users identify and understand German traffic signs and road regulations (StVO). It combines visual and textual information with an AI assistant to provide clear, context-aware explanations.

---

##  Overview

Navigating German traffic rules can be tedious due to dense legal documents and numerous signs. SmartSign RAG simplifies this by combining:

* **Visual Retrieval:** Utilization of high-quality traffic sign images.
* **Textual Context:** Integration of scraped and structured German road regulations (StVO).
* **AI Intelligence:** A Large Language Model (LLM) that explains rules using both image metadata and textual context.

---

##  Tech Stack

| Component | Technology |
| :--- | :--- |
| **Framework** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | HuggingFace / MiniLM |
| **LLM** | GPT-4o-mini (via API) |
| **Data Gathering** | BeautifulSoup (Web Scraping) |
| **UI** | Streamlit *(Planned)* |

---

##  Project Structure

```text
data/
├── germany_road_signs.json      # Structured image metadata for traffic signs
└── text_files/                  # Scraped text pages about German driving rules
chroma_db/                       # Persisted vector store
notebooks/                       # Development notebooks (ingestion, indexing, RAG)
src/
├── scraper.py                   # Web scraping images & text
├── data_loader.py               # Loading JSON/text and converting to Documents
├── chunker.py                   # Splitting documents into semantic chunks
├── embeddings.py                # Embedding, vector store, and retriever logic
└── rag_pipeline.py              # RAG chain creation and query interface
main.py                          # CLI for scraping, indexing, and querying

## Current Status
[x] Data Ingestion: Automated web scraping for high-quality assets

[x] Knowledge Base: Structured JSON descriptions & scraped text files

[x] Document Chunking: Splitting data for embedding layer

[x] Embedding & Vector Store: Build and persist embeddings

[x] RAG Pipeline: Querying via LLM with context

[ ] Interactive UI: Streamlit interface (Planned)