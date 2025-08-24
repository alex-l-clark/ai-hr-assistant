"""Vector store management using ChromaDB for document embeddings and retrieval."""

import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from config import Config
from utils import setup_logging

logger = setup_logging(__name__)

class VectorStoreManager:
    """Manages document embeddings and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory or Config.CHROMA_DB_PATH
        self.collection_name = collection_name or Config.COLLECTION_NAME
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_EMBEDDING_MODEL
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.vectorstore = None
        self._initialize_vectorstore()
        
        logger.info(f"VectorStoreManager initialized with collection: {self.collection_name}")
    
    def _initialize_vectorstore(self):
        """Initialize or load the existing ChromaDB vectorstore."""
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Check if collection exists and has documents
            collection_count = self.get_collection_count()
            logger.info(f"Vectorstore initialized with {collection_count} existing documents")
            
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs added to the store
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        try:
            # Add documents to vectorstore
            doc_ids = self.vectorstore.add_documents(documents)
            
            logger.info(f"Successfully added {len(documents)} documents to vectorstore")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {str(e)}")
            raise
    
    def create_retriever(self, search_type: str = "similarity", k: int = None, search_kwargs: Dict[str, Any] = None):
        """Create a retriever from the vectorstore.
        
        Args:
            search_type: Type of search to perform ("similarity", "mmr")
            k: Number of documents to retrieve
            search_kwargs: Additional search parameters
            
        Returns:
            Langchain retriever object
        """
        k = k or Config.MAX_RETRIEVAL_DOCS
        search_kwargs = search_kwargs or {}
        
        if search_type == "similarity":
            search_kwargs.update({"k": k})
        elif search_type == "mmr":
            search_kwargs.update({
                "k": k,
                "fetch_k": k * 2,  # Fetch more candidates for MMR
                "lambda_mult": 0.5  # Balance between relevance and diversity
            })
        
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        logger.info(f"Created retriever with search_type={search_type}, k={k}")
        return retriever
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        k = k or Config.MAX_RETRIEVAL_DOCS
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Similarity search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with relevance scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        k = k or Config.MAX_RETRIEVAL_DOCS
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Similarity search with scores returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection.
        
        Returns:
            Number of documents in the collection
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self._initialize_vectorstore()
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def update_documents(self, new_documents: List[Document]) -> List[str]:
        """Update the vector store with new documents (replaces existing collection).
        
        Args:
            new_documents: List of new Document objects
            
        Returns:
            List of document IDs
        """
        logger.info("Updating vector store with new documents...")
        
        # Delete existing collection
        self.delete_collection()
        
        # Add new documents
        doc_ids = self.add_documents(new_documents)
        
        logger.info(f"Vector store updated with {len(new_documents)} documents")
        return doc_ids
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.get_collection_count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": Config.OPENAI_EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "embedding_model": Config.OPENAI_EMBEDDING_MODEL,
                "error": str(e)
            }
    
    def check_vectorstore_ready(self) -> bool:
        """Check if the vectorstore is ready for queries.
        
        Returns:
            True if vectorstore has documents, False otherwise
        """
        count = self.get_collection_count()
        is_ready = count > 0
        
        if not is_ready:
            logger.warning("Vectorstore is empty - no documents available for retrieval")
        else:
            logger.info(f"Vectorstore is ready with {count} documents")
            
        return is_ready
    
    def search_with_metadata_filter(self, query: str, metadata_filter: Dict[str, Any], k: int = None) -> List[Document]:
        """Search with metadata filtering.
        
        Args:
            query: Search query string
            metadata_filter: Dictionary of metadata filters
            k: Number of results to return
            
        Returns:
            List of filtered documents
        """
        k = k or Config.MAX_RETRIEVAL_DOCS
        
        try:
            results = self.vectorstore.similarity_search(
                query, 
                k=k, 
                filter=metadata_filter
            )
            logger.info(f"Metadata filtered search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing metadata filtered search: {str(e)}")
            raise