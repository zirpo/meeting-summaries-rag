# Vector store module for the RAG system

from pathlib import Path
from typing import List, Optional, Union

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    USE_LOCAL_MODELS,
)


def get_embedding_model() -> Embeddings:
    """
    Get the appropriate embedding model based on configuration
    
    Returns:
        Embeddings model
    """
    if USE_LOCAL_MODELS:
        return OllamaEmbeddings(
            model=LOCAL_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    else:
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )


def create_or_update_vector_store(
    documents: List[Document],
    persist_directory: Optional[Union[str, Path]] = None,
) -> Chroma:
    """
    Create or update a vector store with the provided documents
    
    Args:
        documents: List of documents to add to the vector store
        persist_directory: Directory to persist the vector store
        
    Returns:
        Chroma vector store
    """
    if persist_directory is None:
        persist_directory = CHROMA_PERSIST_DIRECTORY
    
    # Get embedding model
    embedding_model = get_embedding_model()
    
    # Create or load vector store
    vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_model,
    )
    
    # Add documents to vector store
    if documents:
        vector_store.add_documents(documents)
        vector_store.persist()
    
    return vector_store


def get_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
) -> Optional[Chroma]:
    """
    Get the vector store
    
    Args:
        persist_directory: Directory where the vector store is persisted
        
    Returns:
        Chroma vector store or None if it doesn't exist
    """
    if persist_directory is None:
        persist_directory = CHROMA_PERSIST_DIRECTORY
    
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        return None
    
    # Get embedding model
    embedding_model = get_embedding_model()
    
    # Load vector store
    try:
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_model,
        )
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


def similarity_search(
    query: str,
    k: int = 4,
    persist_directory: Optional[Union[str, Path]] = None,
) -> List[Document]:
    """
    Perform similarity search on the vector store
    
    Args:
        query: Query string
        k: Number of results to return
        persist_directory: Directory where the vector store is persisted
        
    Returns:
        List of similar documents
    """
    vector_store = get_vector_store(persist_directory)
    if not vector_store:
        return []
    
    return vector_store.similarity_search(query, k=k)


def delete_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Delete the vector store
    
    Args:
        persist_directory: Directory where the vector store is persisted
        
    Returns:
        True if successful, False otherwise
    """
    if persist_directory is None:
        persist_directory = CHROMA_PERSIST_DIRECTORY
    
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        return True
    
    try:
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Load and delete vector store
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_model,
        )
        vector_store.delete_collection()
        return True
    except Exception as e:
        print(f"Error deleting vector store: {e}")
        return False