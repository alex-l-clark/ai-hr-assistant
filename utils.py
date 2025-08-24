"""Utility functions for the AI HR Assistant."""

import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(
        log_dir / f"hr_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def format_documents_for_context(documents: List[Any], max_length: int = 4000) -> str:
    """Format retrieved documents into context string.
    
    Args:
        documents: List of retrieved documents
        max_length: Maximum length of context string
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(documents):
        # Add document header
        source = doc.metadata.get('source_file', 'Unknown Source')
        page = doc.metadata.get('page', 'N/A')
        header = f"\n--- Document {i+1}: {source} (Page {page}) ---\n"
        
        content = doc.page_content.strip()
        
        # Check if adding this document would exceed max length
        addition_length = len(header) + len(content)
        if current_length + addition_length > max_length and context_parts:
            context_parts.append(f"\n... (Additional documents truncated due to length limit) ...")
            break
        
        context_parts.append(header + content)
        current_length += addition_length
    
    return "\n".join(context_parts)

def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common PDF artifacts
    text = text.replace('\x00', '')
    text = text.replace('\uf0b7', '•')  # Replace bullet point artifacts
    
    return text.strip()

def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> tuple[bool, str]:
    """Validate file path and extension.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, message)
    """
    allowed_extensions = allowed_extensions or ['.pdf']
    
    if not file_path:
        return False, "No file path provided"
    
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    file_extension = Path(file_path).suffix.lower()
    if file_extension not in allowed_extensions:
        return False, f"File type {file_extension} not allowed. Allowed types: {allowed_extensions}"
    
    return True, "File is valid"

def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {directory_path}: {str(e)}")
        return False

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {"error": "File does not exist"}
    
    try:
        stat = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": Path(file_path).suffix.lower()
        }
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}

def format_chat_history(chat_history: List[Dict[str, str]], max_entries: int = 5) -> str:
    """Format chat history for display.
    
    Args:
        chat_history: List of chat exchanges
        max_entries: Maximum number of entries to include
        
    Returns:
        Formatted chat history string
    """
    if not chat_history:
        return "No previous conversation"
    
    formatted_entries = []
    recent_history = chat_history[-max_entries:]
    
    for i, entry in enumerate(recent_history):
        question = entry.get('question', '').strip()
        answer = entry.get('answer', '').strip()
        
        if question and answer:
            formatted_entries.append(f"Q{i+1}: {question}")
            formatted_entries.append(f"A{i+1}: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            formatted_entries.append("")  # Empty line for separation
    
    return "\n".join(formatted_entries)

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Validate OpenAI API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not api_key:
        return False, "API key is empty"
    
    if not api_key.startswith('sk-'):
        return False, "OpenAI API key should start with 'sk-'"
    
    if len(api_key) < 20:
        return False, "API key appears to be too short"
    
    return True, "API key format is valid"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from filename patterns.
    
    Args:
        filename: Name of the file
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'filename': filename,
        'document_type': 'HR Policy'
    }
    
    filename_lower = filename.lower()
    
    # Determine document type based on filename
    if any(word in filename_lower for word in ['leave', 'vacation', 'pto']):
        metadata['category'] = 'Leave Policies'
    elif any(word in filename_lower for word in ['benefit', 'insurance', 'health']):
        metadata['category'] = 'Benefits'
    elif any(word in filename_lower for word in ['conduct', 'ethics', 'compliance']):
        metadata['category'] = 'Code of Conduct'
    elif any(word in filename_lower for word in ['training', 'development']):
        metadata['category'] = 'Training & Development'
    elif any(word in filename_lower for word in ['performance', 'review', 'evaluation']):
        metadata['category'] = 'Performance Management'
    else:
        metadata['category'] = 'General HR'
    
    return metadata