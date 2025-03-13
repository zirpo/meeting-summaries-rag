# LLM module for the RAG system

from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from src.config import (
    COMPLETION_MODEL,
    LOCAL_COMPLETION_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    USE_LOCAL_MODELS,
)


# Define prompt templates
MEETING_QA_TEMPLATE = """
You are an assistant that helps retrieve information from meeting summaries.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""


def get_llm_model() -> LLM:
    """
    Get the appropriate LLM model based on configuration
    
    Returns:
        LLM model
    """
    if USE_LOCAL_MODELS:
        return Ollama(
            model=LOCAL_COMPLETION_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
    else:
        return ChatOpenAI(
            model=COMPLETION_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1,
        )


def create_qa_chain(retriever: BaseRetriever) -> RetrievalQA:
    """
    Create a question-answering chain
    
    Args:
        retriever: Document retriever
        
    Returns:
        RetrievalQA chain
    """
    # Get LLM model
    llm = get_llm_model()
    
    # Create prompt
    prompt = PromptTemplate(
        template=MEETING_QA_TEMPLATE,
        input_variables=["context", "question"],
    )
    
    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    
    return chain


def generate_response(
    query: str,
    retriever: BaseRetriever,
) -> tuple[str, List[Document]]:
    """
    Generate a response to a query
    
    Args:
        query: Query string
        retriever: Document retriever
        
    Returns:
        Tuple of (response, source_documents)
    """
    # Create chain
    chain = create_qa_chain(retriever)
    
    # Generate response
    result = chain.invoke({"query": query})
    
    return result["result"], result["source_documents"]


def format_source_documents(source_docs: List[Document]) -> str:
    """
    Format source documents for display
    
    Args:
        source_docs: List of source documents
        
    Returns:
        Formatted string
    """
    if not source_docs:
        return "No source documents found."
    
    formatted = "Sources:\n"
    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown")
        date = metadata.get("date", "Unknown date")
        topic = metadata.get("topic", "Unknown topic")
        
        formatted += f"{i}. {topic} ({date}) - {source}\n"
    
    return formatted