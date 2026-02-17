import os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from operator import itemgetter

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.embeddings import build_retriever, load_vectorstore

DEFAULT_MODEL = "llama-3.3-70b-versatile" 
TEMPERATURE = 0

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str = GROQ_API_KEY,
    temperature: float = TEMPERATURE
) -> ChatGroq:
    """
    Initialize Groq LLM
    """
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature,
    )

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant on German road rules.

### PRIORITY RULE:
If the context contains an "IMAGE" URL, you MUST display it using Markdown syntax:
![Traffic Sign](image_url)

### INSTRUCTIONS:
- For every sign you mention, put its image right ABOVE the title.
- Use this structure:
  ![Sign Title](image_url)
  **Title of the sign**
  Rules: ...

Context:
{context}

Question:
{question}

Answer in a clear and structured way.
""")

def format_docs(docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Sort and format documents for RAG prompt
    """
    sorted_docs = sorted(
        docs, 
        key=lambda d: d.metadata.get("image_url") not in [None, "None", ""], 
        reverse=True
    )

    formatted = []
    sources = set()
    
    for d in sorted_docs:
        img = d.metadata.get("image_url") or d.metadata.get("image_path") or d.metadata.get("url") or "None"
        
        block = f"""
[DOCUMENT START]
IMAGE: {img}
TEXT: {d.page_content}
CATEGORY: {d.metadata.get('category', 'Unknown')}
[DOCUMENT END]
"""
        formatted.append(block.strip())
        if "source" in d.metadata:
            sources.add(d.metadata["source"])
            
    return "\n\n---\n\n".join(formatted), list(sources)

def build_rag_chain(vectorstore=None, k: int = 8, llm=None):
    if vectorstore is None:
        vectorstore = load_vectorstore()

    if llm is None:
        llm = build_llm()

    retriever = build_retriever(vectorstore, k=k)

    def _format_docs_step(inputs):
        docs = inputs["docs"]
        context, sources = format_docs(docs)
        return {
            "context": context,
            "sources": sources,
            "question": inputs["question"]
        }

    return (
        RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        | RunnableLambda(_format_docs_step)
        | RunnableParallel({
            "answer": RAG_PROMPT | llm, 
            "sources": itemgetter("sources")
        })
    )

def run_rag_query(rag_chain, question: str) -> Dict[str, Any]:
    result = rag_chain.invoke(question)
    return {
        "answer": result["answer"].content,
        "sources": result["sources"]
    }

if __name__ == "__main__":
    vs = load_vectorstore()
    chain = build_rag_chain(vectorstore=vs)
    query = "What does a slippery road traffic sign mean?"
    result = run_rag_query(chain, query)
    print(f"ANSWER:\n{result['answer']}\n\nSOURCES: {result['sources']}")