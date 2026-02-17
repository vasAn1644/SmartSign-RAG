import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.rag_pipeline import build_rag_chain, run_rag_query

# --- Page Configuration ---
st.set_page_config(
    page_title="SmartSign AI",
    page_icon="🚦",
    layout="wide"
)

# --- Image Styling ---
st.markdown("""
    <style>
    /* Center images and prevent them from being oversized */
    img {
        max-width: 250px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 2px solid #f0f2f6;
        transition: transform 0.2s;
    }
    img:hover {
        transform: scale(1.05);
    }
    .stMarkdown {
        font-size: 1.05rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Backend Initialization ---
@st.cache_resource
def load_backend():
    # Lazy import to speed up initial load
    from src.embeddings import load_vectorstore
    try:
        vectorstore = load_vectorstore()
        rag_chain = build_rag_chain(vectorstore=vectorstore)
        return rag_chain
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

rag_chain = load_backend()

# --- Header ---
st.title("🚦 SmartSign RAG Assistant")
st.markdown("### Your personal guide to German road signs")
st.info("I use Artificial Intelligence and the official StVO database to answer your questions.")

# --- Search Input ---
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What does a slippery road sign look like and mean?",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# --- Search Logic ---
if search_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing database and generating response..."):
            try:
                # Execute RAG query
                result = run_rag_query(rag_chain, query)
                # Store in session state to persist through UI interactions
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Output Display ---
if "result" in st.session_state:
    res = st.session_state["result"]
    
    st.markdown("---")
    st.subheader("🤖 AI Response")
    
    # The answer will render Markdown images automatically
    st.markdown(res["answer"])
    
    # --- Sources Expander ---
    st.markdown(" ") # Spacer
    with st.expander("📚 Show Sources"):
        sources = res.get("sources", [])
        if sources:
            st.write("Information was retrieved from the following documents:")
            for i, src in enumerate(sources, 1):
                st.markdown(f"**{i}.** `{src}`")
        else:
            st.write("No specific sources cited.")

# --- Footer ---
st.markdown("---")
st.caption("SmartSign RAG · Powered by Groq (Llama 3.3) · German Traffic Rules Assistant")