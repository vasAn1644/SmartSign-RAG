import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.rag_pipeline import build_rag_chain, run_rag_query
from src.embeddings import load_vectorstore

st.set_page_config(
    page_title="SmartSign RAG",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 SmartSign RAG")
st.subheader("Multimodal AI assistant for German road signs")

@st.cache_resource
def load_backend():
    vectorstore = load_vectorstore()
    rag_chain = build_rag_chain(vectorstore=vectorstore)
    return rag_chain

rag_chain = load_backend()

st.markdown("Ask a question about German traffic signs or road rules:")

query = st.text_input(
    "Your question",
    placeholder="e.g. What does a slippery road sign mean?"
)

if st.button("🔍 Search"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            answer = run_rag_query(rag_chain, query)

        st.markdown("## 🤖 AI Answer")
        st.markdown(answer)


st.markdown("---")
st.caption("SmartSign RAG · Multimodal Retrieval-Augmented Generation System")
