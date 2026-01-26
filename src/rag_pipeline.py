"""
Responsible for:
1) Initializing LLM for RAG
2) Integrating retriever
3) Preparing prompt template
4) Running RAG queries
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from .embeddings import build_retriever, load_vectorstore


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_BASE = "https://api.aimlapi.com/v1"
TEMPERATURE = 0

load_dotenv()
AIML_API_KEY = os.getenv("AIML_API_KEY")


def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str = AIML_API_KEY,
    base_url: str = DEFAULT_API_BASE,
    temperature: float = TEMPERATURE
) -> ChatOpenAI:
    """
    Initialize LLM
    """
    # REVIEW FIX: explicit check for API key to prevent silent failure
    if not api_key:
        raise ValueError("AIML_API_KEY not set in environment")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )



# REVIEW FIX: structured multi-line prompt template for RAG, includes handling of images
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant on German road rules.

Use the provided context to answer the question.
If the context contains image metadata, mention the image(s) when relevant.

If the description of a sign is incomplete, you MAY infer its meaning
based on common traffic rules and the sign category.
Do NOT invent specific legal details that are not implied by the context.

When images are relevant, include them in your answer using this format:

Image: <image_url>
Explanation: <short explanation of the sign>

Context:
{context}

Question:
{question}

Answer in a clear and structured way.
""")


def format_docs(docs):
    """
    Convert list of Documents to formatted string for prompt context
    """
    # REVIEW FIX: extracted formatting to dedicated function for reusability & clarity
    formatted = []
    for d in docs:
        block = f"""
TEXT:
{d.page_content}

IMAGE_URL:
{d.metadata.get("image_url", "None")}

CATEGORY:
{d.metadata.get("category", "Unknown")}
"""
        formatted.append(block.strip())

    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore=None, k: int = 5, llm=None):
    """
    Returns a composable RAG chain
    """
    # REVIEW FIX: lazy-loading vectorstore if not provided
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # REVIEW FIX: lazy-building LLM if not provided
    if llm is None:
        llm = build_llm()

    # REVIEW FIX: retriever wrapped with document formatter
    retriever = build_retriever(vectorstore, k=k)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
    )

    return rag_chain


def run_rag_query(rag_chain, question: str):
    """
    Run a RAG query and return LLM response
    """
    # REVIEW FIX: simple wrapper for invoking chain, abstracts away retriever & prompt
    response = rag_chain.invoke(question)
    return response.content


if __name__ == "__main__":
    print("Loading vectorstore...")
    # REVIEW FIX: explicit import for debug/test mode
    from embeddings import load_vectorstore

    vs = load_vectorstore()
    chain = build_rag_chain(vectorstore=vs)

    query = "What does a slippery road traffic sign mean?"
    print(f"Query: {query}\n")

    result = run_rag_query(chain, query)
    print(result)

# REVIEW FIX: debug block allows manual verification of retrieval + RAG chain
# REVIEW FIX: separation of concerns - loading embeddings, building chain, running query
