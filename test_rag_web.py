#!/usr/bin/env python3
"""
Simple script to run a web interface for the RAG system.
"""

import os
import sys
from pathlib import Path

import gradio as gr

from src.config import validate_config
from src.document_processor import process_documents
from src.vector_store import create_or_update_vector_store, get_vector_store
from src.llm import generate_response, format_source_documents

def initialize_system():
    """
    Initialize the RAG system.
    """
    # Validate configuration
    is_valid, error = validate_config()
    if not is_valid:
        return False, error
    
    # Get vector store
    vector_store = get_vector_store()
    
    # If vector store doesn't exist, create it
    if not vector_store:
        print("Vector store not found. Creating new vector store...")
        
        # Process documents
        print("Processing documents...")
        documents = process_documents()
        print(f"Processed {len(documents)} document chunks")
        
        if not documents:
            return False, "No documents found to process."
        
        # Create vector store
        print("Creating vector store...")
        vector_store = create_or_update_vector_store(documents)
        print(f"Vector store created with {len(documents)} documents")
    
    return True, vector_store

def query_rag(query, use_local_models=False):
    """
    Query the RAG system.
    
    Args:
        query: Query string
        use_local_models: Whether to use local models
        
    Returns:
        Response and sources
    """
    # Set environment variable for model selection
    os.environ["USE_LOCAL_MODELS"] = str(use_local_models).lower()
    
    # Initialize system
    success, result = initialize_system()
    if not success:
        return f"Error: {result}", ""
    
    vector_store = result
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    try:
        # Generate response
        response, source_docs = generate_response(query, retriever)
        
        # Format sources
        sources = format_source_documents(source_docs)
        
        return response, sources
    except Exception as e:
        return f"Error generating response: {str(e)}", ""

def create_web_interface():
    """
    Create a web interface for the RAG system.
    """
    with gr.Blocks(title="Meeting Summaries RAG System") as demo:
        gr.Markdown("# Meeting Summaries RAG System")
        gr.Markdown("Query your meeting summaries using natural language.")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your query here...",
                    lines=2,
                )
                use_local_models = gr.Checkbox(
                    label="Use Local Models (Ollama)",
                    value=False,
                )
                submit_button = gr.Button("Submit")
            
            with gr.Column():
                response_output = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False,
                )
                sources_output = gr.Textbox(
                    label="Sources",
                    lines=5,
                    interactive=False,
                )
        
        submit_button.click(
            fn=query_rag,
            inputs=[query_input, use_local_models],
            outputs=[response_output, sources_output],
        )
        
        gr.Examples(
            [
                ["What technology stack was chosen for the project?", False],
                ["What are the action items for Jane?", False],
                ["When is the next meeting scheduled?", False],
                ["What feedback did the client provide on the design?", False],
                ["What are the phases of the project timeline?", False],
            ],
            [query_input, use_local_models],
        )
    
    return demo

if __name__ == "__main__":
    # Check if gradio is installed
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Please install it with:")
        print("pip install gradio")
        sys.exit(1)
    
    # Create web interface
    demo = create_web_interface()
    
    # Launch web interface
    demo.launch(share=False)