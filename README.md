#  SmartSign RAG: Multimodal Traffic Assistant

A Multimodal Retrieval-Augmented Generation (RAG) application designed to help users identify and understand German traffic signs and road regulations (StVO).

##  Overview
Finding specific traffic rules can be tedious when browsing through dense legal documents. **SmartSign RAG** solves this by combining:
- **Visual Retrieval:** High-quality traffic sign graphics.
- **Textual Context:** Official German Road Traffic Regulations (StVO).
- **AI Intelligence:** An LLM that explains rules based on both the retrieved image and legal text.

##  Tech Stack
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Models:** - **Vision/Text Embedding:** CLIP (OpenAI)
  - **LLM:** Llama 3 (via Ollama or API)
- **UI:** Streamlit
- **Data Gathering:** BeautifulSoup (Web Scraping from *Getting Around Germany*)

##  Project Structure
- `data/samples_hq/`: High-quality traffic sign graphics.
- `data/signs_description.json`: Semantic descriptions for each traffic sign class.
- `data/raw_pdf/`: Official StVO (Road Traffic Regulations) English translation.
- `notebooks/`: Step-by-step development (Ingestion, Indexing, RAG Chain).

##  Current Status
- [x] **Data Ingestion:** Automated web scraping for high-quality assets.
- [x] **Knowledge Base:** Structured JSON descriptions & PDF legal documents.
- [ ] **Multimodal Indexing:** Creating the vector store (Coming soon).
- [ ] **Interactive UI:** Streamlit interface for querying.

##  Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/vasAn1644/SmartSign-RAG.git](https://github.com/vasAn1644/SmartSign-RAG.git)
2. Install dependencies:
    ```bash
    pip install -r requirements.txt