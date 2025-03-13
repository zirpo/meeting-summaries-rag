#!/usr/bin/env python3
"""
Simple script to test the RAG system with Ollama.
"""

import os
import sys
from pathlib import Path

from src.config import validate_config
from src.vector_store import get_vector_store
from src.llm import generate_response, format_source_documents

def main():
    """
    Main function to test the RAG system with Ollama.
    """
    # Set environment variable to use local models
    os.environ["USE_LOCAL_MODELS"] = "true"
    
    # Validate configuration
    is_valid, error = validate_config()
    if not is_valid:
        print(f"Error: {error}")
        print("Note: Make sure Ollama is installed and running.")
        print("You can install Ollama from https://ollama.ai/")
        print("And pull the required models with:")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull llama3")
        sys.exit(1)
    
    # Get vector store
    vector_store = get_vector_store()
    if not vector_store:
        print("Vector store not found. Please run test_rag.py first.")
        sys.exit(1)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Test query
    query = "What technology stack was chosen for the project?"
    print(f"\nQuery: {query}")
    
    # Generate response
    print("Generating response using Ollama...")
    response, source_docs = generate_response(query, retriever)
    
    # Print response
    print("\nResponse:")
    print(response)
    
    # Print sources
    print("\nSources:")
    print(format_source_documents(source_docs))

if __name__ == "__main__":
    main()