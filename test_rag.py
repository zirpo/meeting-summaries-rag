#!/usr/bin/env python3
"""
Simple script to test the RAG system.
"""

import sys
from pathlib import Path

from src.config import validate_config
from src.document_processor import process_documents
from src.vector_store import create_or_update_vector_store, get_vector_store
from src.llm import generate_response, format_source_documents

def main():
    """
    Main function to test the RAG system.
    """
    # Validate configuration
    is_valid, error = validate_config()
    if not is_valid:
        print(f"Error: {error}")
        sys.exit(1)
    
    # Process documents
    print("Processing documents...")
    documents = process_documents()
    print(f"Processed {len(documents)} document chunks")
    
    # Create or update vector store
    print("Creating vector store...")
    vector_store = create_or_update_vector_store(documents)
    print(f"Vector store created with {len(documents)} documents")
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Test query
    query = "What technology stack was chosen for the project?"
    print(f"\nQuery: {query}")
    
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