"""Document processing module for loading and splitting PDF documents."""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import Config
from utils import setup_logging

logger = setup_logging(__name__)

class DocumentProcessor:
    """Handles document loading, processing, and text splitting."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF file and return its documents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error loading the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_path': file_path,
                    'document_type': 'HR Policy'
                })
            
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def load_documents_from_directory(self, directory_path: str = None) -> List[Document]:
        """Load all PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all Document objects from all PDFs
        """
        directory_path = directory_path or Config.DOCUMENTS_PATH
        
        if not os.path.exists(directory_path):
            logger.warning(f"Documents directory doesn't exist: {directory_path}")
            return []
        
        all_documents = []
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        for pdf_file in pdf_files:
            try:
                documents = self.load_pdf(str(pdf_file))
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(all_documents)} total pages from {len(pdf_files)} PDF files")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document chunks
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(doc.page_content)
                })
            
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_documents(self, directory_path: str = None) -> List[Document]:
        """Complete document processing pipeline: load and split documents.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of processed document chunks ready for embedding
        """
        logger.info("Starting document processing pipeline...")
        
        # Load documents
        documents = self.load_documents_from_directory(directory_path)
        
        if not documents:
            logger.warning("No documents loaded, returning empty list")
            return []
        
        # Split documents into chunks
        split_documents = self.split_documents(documents)
        
        logger.info(f"Document processing complete: {len(split_documents)} chunks ready")
        return split_documents
    
    def process_single_file(self, file_path: str) -> List[Document]:
        """Process a single PDF file: load and split.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of processed document chunks
        """
        logger.info(f"Processing single file: {file_path}")
        
        # Load single document
        documents = self.load_pdf(file_path)
        
        # Split into chunks
        split_documents = self.split_documents(documents)
        
        logger.info(f"Single file processing complete: {len(split_documents)} chunks from {file_path}")
        return split_documents
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_chunks": 0, "total_characters": 0, "avg_chunk_size": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents)
        
        # Get unique source files
        source_files = set(doc.metadata.get('source_file', 'unknown') for doc in documents)
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "source_files": list(source_files),
            "num_source_files": len(source_files)
        }