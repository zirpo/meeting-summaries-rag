# Configuration module for the RAG system

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
BASE_DIR = Path(__file__).parent.parent
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
DOCUMENT_STORE_DIRECTORY = os.getenv("DOCUMENT_STORE_DIRECTORY", "./meetings")

# Ensure paths are absolute
CHROMA_PERSIST_DIRECTORY = BASE_DIR / CHROMA_PERSIST_DIRECTORY if not os.path.isabs(CHROMA_PERSIST_DIRECTORY) else Path(CHROMA_PERSIST_DIRECTORY)
DOCUMENT_STORE_DIRECTORY = BASE_DIR / DOCUMENT_STORE_DIRECTORY if not os.path.isabs(DOCUMENT_STORE_DIRECTORY) else Path(DOCUMENT_STORE_DIRECTORY)

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-3.5-turbo")

# Local Model Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
LOCAL_COMPLETION_MODEL = os.getenv("LOCAL_COMPLETION_MODEL", "llama3")

# Use local models flag
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

def validate_config() -> tuple[bool, Optional[str]]:
    """
    Validate the configuration settings.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not USE_LOCAL_MODELS and not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is required when not using local models"
    
    # Create directories if they don't exist
    CHROMA_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DOCUMENT_STORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    return True, None