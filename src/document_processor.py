# Document processor module for the RAG system

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENT_STORE_DIRECTORY


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from filename using pattern YYYY-MM-DD_MeetingTopic.ext
    
    Args:
        filename: The filename to extract metadata from
        
    Returns:
        Dict containing extracted metadata
    """
    metadata = {}
    
    # Extract date and topic from filename
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)\\.(.*)$", filename)
    if date_match:
        date_str, topic, _ = date_match.groups()
        try:
            # Validate date format
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            metadata["date"] = date_str
            metadata["year"] = date_obj.year
            metadata["month"] = date_obj.month
            metadata["day"] = date_obj.day
        except ValueError:
            pass
        
        # Clean up topic (replace underscores with spaces, etc.)
        topic = topic.replace("_", " ").strip()
        metadata["topic"] = topic
    
    return metadata


def extract_metadata_from_content(content: str) -> Dict[str, str]:
    """
    Extract metadata from document content
    
    Args:
        content: The document content
        
    Returns:
        Dict containing extracted metadata
    """
    metadata = {}
    
    # Extract participants if they exist in the format "Participants: Person1, Person2"
    participants_match = re.search(r"participants:?\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
    if participants_match:
        participants = participants_match.group(1).strip()
        metadata["participants"] = participants
    
    return metadata


def load_document(file_path: Path) -> Tuple[Optional[Document], Dict[str, str]]:
    """
    Load a document from a file and extract metadata
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Tuple of (Document, metadata)
    """
    if not file_path.exists():
        return None, {}
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(file_path.name)
    metadata["source"] = str(file_path)
    metadata["filename"] = file_path.name
    
    # Load document using TextLoader for both .md and .txt files
    try:
        loader = TextLoader(str(file_path))
        docs = loader.load()
        
        if not docs:
            return None, metadata
        
        # Extract additional metadata from content
        content_metadata = extract_metadata_from_content(docs[0].page_content)
        metadata.update(content_metadata)
        
        # Update document with metadata
        doc = docs[0]
        doc.metadata.update(metadata)
        
        return doc, metadata
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return None, metadata


def chunk_document(document: Document) -> List[Document]:
    """
    Split a document into chunks
    
    Args:
        document: The document to split
        
    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_documents([document])
    
    # Ensure all chunks have the same metadata
    for chunk in chunks:
        chunk.metadata.update(document.metadata)
    
    return chunks


def process_documents(directory: Optional[Path] = None) -> List[Document]:
    """
    Process all documents in the specified directory
    
    Args:
        directory: Directory containing documents to process
        
    Returns:
        List of processed document chunks
    """
    if directory is None:
        directory = DOCUMENT_STORE_DIRECTORY
    
    all_chunks = []
    
    # Walk through directory and process all files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".md", ".txt")):
                file_path = Path(root) / file
                doc, _ = load_document(file_path)
                
                if doc:
                    chunks = chunk_document(doc)
                    all_chunks.extend(chunks)
    
    return all_chunks