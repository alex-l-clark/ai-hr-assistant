"""Main entry point for the AI HR Assistant application."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from config import Config
from utils import setup_logging, create_directory_if_not_exists

logger = setup_logging(__name__)

def setup_environment():
    """Set up the application environment."""
    logger.info("Setting up application environment...")
    
    # Create necessary directories
    directories = [
        Config.DOCUMENTS_PATH,
        Config.CHROMA_DB_PATH,
        "logs"
    ]
    
    for directory in directories:
        if create_directory_if_not_exists(directory):
            logger.info(f"Directory ready: {directory}")
        else:
            logger.error(f"Failed to create directory: {directory}")
            return False
    
    return True

def check_configuration():
    """Check if the application is properly configured."""
    logger.info("Checking application configuration...")
    
    try:
        Config.validate_config()
        logger.info("Configuration validation passed")
        return True
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"\n❌ Configuration Error: {e}")
        print("\nTo fix this:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add your OpenAI API key")
        print("3. Ensure your OpenAI API key has sufficient credits")
        return False

def check_documents():
    """Check if documents are available."""
    docs_path = Path(Config.DOCUMENTS_PATH)
    
    if not docs_path.exists():
        logger.warning(f"Documents directory doesn't exist: {docs_path}")
        return False
    
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_path}")
        print(f"\n📁 No HR documents found in {docs_path}")
        print("To get started:")
        print("1. Add PDF files containing HR policies to the documents folder")
        print("2. You can also upload documents through the web interface")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files in documents directory")
    return True

def print_startup_info():
    """Print application startup information."""
    print("\n" + "="*60)
    print("🤖 Nestlé AI-Powered HR Assistant")
    print("="*60)
    print(f"📊 Configuration:")
    print(f"   - LLM Model: {Config.OPENAI_MODEL}")
    print(f"   - Embedding Model: {Config.OPENAI_EMBEDDING_MODEL}")
    print(f"   - Documents Path: {Config.DOCUMENTS_PATH}")
    print(f"   - Vector DB Path: {Config.CHROMA_DB_PATH}")
    print(f"   - Port: {Config.GRADIO_PORT}")
    print(f"   - Share: {Config.GRADIO_SHARE}")
    print("="*60)

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="AI HR Assistant for Nestlé")
    parser.add_argument(
        "--port", 
        type=int, 
        default=Config.GRADIO_PORT,
        help=f"Port to run the application on (default: {Config.GRADIO_PORT})"
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        default=Config.GRADIO_SHARE,
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--no-docs-check",
        action="store_true",
        help="Skip document availability check (useful for uploading via web interface)"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    print_startup_info()
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        sys.exit(1)
    
    # Check configuration
    if not check_configuration():
        sys.exit(1)
    
    # Check for documents (unless skipped)
    if not args.no_docs_check:
        has_docs = check_documents()
        if not has_docs:
            print("\n⚠️  Starting without pre-loaded documents.")
            print("You can upload documents through the web interface once the app starts.")
    
    try:
        # Create and launch the application
        logger.info("Creating HR Assistant application...")
        app = create_app()
        
        print(f"\n🚀 Starting HR Assistant...")
        print(f"📱 The application will be available at: http://localhost:{args.port}")
        
        if args.share:
            print("🌐 A public shareable link will be generated")
        
        print("\n💡 Tips:")
        print("- Click 'Initialize System' in the web interface to load documents")
        print("- Upload additional PDF documents using the Document Management tab")
        print("- Ask questions about HR policies, benefits, leave procedures, etc.")
        print("\n⌨️  Press Ctrl+C to stop the application")
        print("-" * 60)
        
        # Launch the application
        app.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        print("\n👋 HR Assistant stopped. Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\n❌ Application Error: {str(e)}")
        print("Check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()