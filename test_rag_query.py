#!/usr/bin/env python3
"""
Simple script to test querying the RAG system.
"""

import sys
from pathlib import Path

from src.config import validate_config
from src.vector_store import get_vector_store
from src.llm import generate_response, format_source_documents

def main():
    """
    Main function to test querying the RAG system.
    """
    # Validate configuration
    is_valid, error = validate_config()
    if not is_valid:
        print(f"Error: {error}")
        sys.exit(1)
    
    # Get vector store
    vector_store = get_vector_store()
    if not vector_store:
        print("Vector store not found. Please run test_rag.py first.")
        sys.exit(1)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Test queries
    queries = [
        "What are the action items for Jane?",
        "When is the next meeting scheduled?",
        "What feedback did the client provide on the design?",
        "What are the phases of the project timeline?",
    ]
    
    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")
        
        # Generate response
        print("Generating response...")
        response, source_docs = generate_response(query, retriever)
        
        # Print response
        print("\nResponse:")
        print(response)
        
        # Print sources
        print("\nSources:")
        print(format_source_documents(source_docs))

if __name__ == "__main__":
    main()