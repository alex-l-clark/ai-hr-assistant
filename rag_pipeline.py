"""RAG (Retrieval-Augmented Generation) pipeline for the HR Assistant."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
from utils import setup_logging, format_documents_for_context

logger = setup_logging(__name__)

class RAGPipeline:
    """Complete RAG pipeline combining document retrieval and response generation."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.llm_handler = LLMHandler()
        self.is_initialized = False
        self.chat_history = []
        
        logger.info("RAG Pipeline components initialized")
    
    def initialize_knowledge_base(self, documents_path: str = None) -> bool:
        """Initialize the knowledge base with HR documents.
        
        Args:
            documents_path: Path to directory containing HR documents
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing knowledge base...")
            
            # Process documents
            documents = self.document_processor.process_documents(documents_path)
            
            if not documents:
                logger.warning("No documents found to initialize knowledge base")
                return False
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(documents)
            
            # Verify vector store is ready
            if not self.vector_store.check_vectorstore_ready():
                logger.error("Vector store initialization failed")
                return False
            
            self.is_initialized = True
            doc_stats = self.document_processor.get_document_stats(documents)
            
            logger.info(f"Knowledge base initialized successfully:")
            logger.info(f"  - {doc_stats['num_source_files']} source files")
            logger.info(f"  - {doc_stats['total_chunks']} document chunks")
            logger.info(f"  - {len(doc_ids)} embeddings created")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            self.is_initialized = False
            return False
    
    def add_document(self, file_path: str) -> bool:
        """Add a single document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if document added successfully
        """
        try:
            logger.info(f"Adding document: {file_path}")
            
            # Process single document
            documents = self.document_processor.process_single_file(file_path)
            
            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully added document {file_path}: {len(documents)} chunks, {len(doc_ids)} embeddings")
            
            # Update initialization status
            self.is_initialized = self.vector_store.check_vectorstore_ready()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {str(e)}")
            return False
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: User's question
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.is_initialized:
            return {
                "answer": "The HR Assistant is not ready yet. Please ensure HR documents have been loaded.",
                "sources": [],
                "error": "System not initialized"
            }
        
        try:
            # Validate question
            is_valid, validation_msg = self.llm_handler.validate_question(question)
            if not is_valid:
                return {
                    "answer": validation_msg,
                    "sources": [],
                    "error": "Invalid question"
                }
            
            # Retrieve relevant documents
            retriever = self.vector_store.create_retriever()
            retrieved_docs = retriever.invoke(question)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information in the HR documents to answer your question. Please try rephrasing your question or contact HR directly for assistance.",
                    "sources": [],
                    "warning": "No relevant documents found"
                }
            
            # Format context from retrieved documents
            context = format_documents_for_context(retrieved_docs)
            
            # Generate response
            if self.chat_history:
                response = self.llm_handler.generate_followup_response(
                    question, context, self.chat_history
                )
            else:
                response = self.llm_handler.generate_response(question, context)
            
            # Prepare sources information
            sources = []
            if include_sources:
                sources = self._extract_sources(retrieved_docs)
            
            # Update chat history
            from datetime import datetime
            self.chat_history.append({
                "question": question,
                "answer": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit chat history size
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            logger.info(f"Query processed successfully: '{question[:50]}...'")
            
            return {
                "answer": response,
                "sources": sources,
                "retrieved_docs_count": len(retrieved_docs),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. Please try again or contact HR directly.",
                "sources": [],
                "error": str(e)
            }
    
    def query_with_history(self, question: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Query with explicit conversation history.
        
        Args:
            question: User's question
            conversation_history: Previous conversation exchanges
            
        Returns:
            Dictionary containing response and metadata
        """
        # Temporarily set chat history
        original_history = self.chat_history
        self.chat_history = conversation_history[-5:]  # Use last 5 exchanges
        
        try:
            result = self.query(question)
            return result
        finally:
            # Restore original history
            self.chat_history = original_history
    
    def _extract_sources(self, documents: List[Any]) -> List[Dict[str, str]]:
        """Extract source information from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source_file = doc.metadata.get('source_file', 'Unknown')
            page_num = doc.metadata.get('page', 'N/A')
            
            # Create unique identifier
            source_id = f"{source_file}_{page_num}"
            
            if source_id not in seen_sources:
                sources.append({
                    "file": source_file,
                    "page": str(page_num),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
                seen_sources.add(source_id)
        
        return sources
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the current knowledge base.
        
        Returns:
            Dictionary with knowledge base information
        """
        vector_info = self.vector_store.get_collection_info()
        
        return {
            "is_initialized": self.is_initialized,
            "document_count": vector_info.get("document_count", 0),
            "collection_name": vector_info.get("collection_name", ""),
            "embedding_model": vector_info.get("embedding_model", ""),
            "llm_model": self.llm_handler.get_model_info(),
            "chat_history_length": len(self.chat_history)
        }
    
    def clear_chat_history(self):
        """Clear the conversation history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def reset_knowledge_base(self) -> bool:
        """Reset the knowledge base (delete all documents).
        
        Returns:
            True if reset successful
        """
        try:
            self.vector_store.delete_collection()
            self.chat_history = []
            self.is_initialized = False
            
            logger.info("Knowledge base reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset knowledge base: {str(e)}")
            return False
    
    def update_knowledge_base(self, documents_path: str = None) -> bool:
        """Update the knowledge base with new documents.
        
        Args:
            documents_path: Path to directory containing new documents
            
        Returns:
            True if update successful
        """
        try:
            logger.info("Updating knowledge base with new documents...")
            
            # Process new documents
            documents = self.document_processor.process_documents(documents_path)
            
            if not documents:
                logger.warning("No new documents found to update knowledge base")
                return False
            
            # Update vector store (replaces existing)
            doc_ids = self.vector_store.update_documents(documents)
            
            # Clear chat history since context has changed
            self.chat_history = []
            
            self.is_initialized = True
            
            logger.info(f"Knowledge base updated: {len(documents)} chunks, {len(doc_ids)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {str(e)}")
            return False
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for documents without generating a response.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        if not self.is_initialized:
            return []
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=max_results)
            
            search_results = []
            for doc, score in results:
                search_results.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source_file": doc.metadata.get('source_file', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "relevance_score": float(score),
                    "metadata": doc.metadata
                })
            
            logger.info(f"Document search completed: {len(search_results)} results for '{query[:50]}...'")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []