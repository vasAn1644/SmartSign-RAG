"""
Central entry point for the RAG system.
Options:
--scrape        : run scraper for images + text
--build-index   : create embeddings and vectorstore
--query <text>  : run RAG query
"""

import argparse

from src.scraper import parse_all_signs, crawl_pages
from src.data_loader import load_all_documents
from src.chunker import chunk_documents
from src.embeddings import create_vectorstore
from src.rag_pipeline import build_rag_chain, run_rag_query

def main():
    parser = argparse.ArgumentParser(description="RAG System for German Road Signs")
    parser.add_argument('--scrape', action='store_true', help="Scrape images and text from iamexpat.de")
    parser.add_argument('--build-index', action='store_true', help="Build vectorstore from scraped data")
    parser.add_argument('--query', type=str, help="Run RAG query")
    args = parser.parse_args()


    if args.scrape:
        print(" Running scraper...")
        parse_all_signs()
        crawl_pages()
        print(" Scraping finished.")


    if args.build_index:
        print(" Loading and chunking documents...")
        all_docs = load_all_documents()  
        chunks = chunk_documents(all_docs)

        print(f" Creating vectorstore for {len(chunks)} chunks...")
        create_vectorstore(chunks)
        print(" Vectorstore created.")

    if args.query:
        print(f" Running RAG query: {args.query}")
        chain = build_rag_chain()  
        answer = run_rag_query(chain, args.query)
        print("\n RAG Response:\n")
        print(answer)

if __name__ == "__main__":
    main()
