# Nestlé AI HR Assistant - Setup Guide

This guide will help you set up and run the AI-powered HR Assistant for Nestlé.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-hr-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your configuration
# Required: Add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Prepare HR Documents
```bash
# Create documents directory if it doesn't exist
mkdir -p documents

# Add your HR policy PDF files to the documents folder
# Example:
# cp /path/to/your/hr_policies.pdf documents/
```

### 6. Run the Application
```bash
# Start the application
python main.py

# Or with additional options
python main.py --port 8080 --share --debug
```

## Configuration Options

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model to use | gpt-4 |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | text-embedding-ada-002 |
| `CHROMA_DB_PATH` | Vector database path | ./chroma_db |
| `COLLECTION_NAME` | Collection name | nestle_hr_policies |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `MAX_RETRIEVAL_DOCS` | Max docs to retrieve | 4 |
| `TEMPERATURE` | LLM temperature | 0.1 |
| `MAX_TOKENS` | Max response tokens | 500 |
| `GRADIO_PORT` | Web interface port | 7860 |
| `GRADIO_SHARE` | Create public link | false |
| `DOCUMENTS_PATH` | HR documents folder | ./documents |

### Command Line Arguments

```bash
python main.py --help

Options:
  --port PORT           Port to run the application on (default: 7860)
  --share              Create a public shareable link
  --debug              Enable debug mode with verbose logging
  --no-docs-check      Skip document availability check
```

## Usage

### 1. Web Interface
- Access the application at `http://localhost:7860` (or your configured port)
- Click "Initialize System" to load HR documents
- Use the Chat tab to ask questions about HR policies
- Use the Document Management tab to upload new documents
- Use the System Info tab to view system status

### 2. Example Questions
- "What is Nestlé's vacation policy?"
- "How do I request sick leave?"
- "What are the company's performance review procedures?"
- "Tell me about employee benefits"
- "What is the disciplinary process?"

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```
   Error: OPENAI_API_KEY is required
   ```
   - Solution: Add your OpenAI API key to the .env file

2. **No Documents Found**
   ```
   No PDF files found in ./documents
   ```
   - Solution: Add HR policy PDF files to the documents folder

3. **Port Already in Use**
   ```
   Error: Port 7860 is already in use
   ```
   - Solution: Use a different port with `--port 8080`

4. **ChromaDB Permission Issues**
   ```
   Permission denied: ./chroma_db
   ```
   - Solution: Check folder permissions or change CHROMA_DB_PATH

### Logs
- Application logs are saved in the `logs/` directory
- Check logs for detailed error information
- Use `--debug` flag for verbose logging

## File Structure

```
ai-hr-assistant/
├── main.py                 # Application entry point
├── app.py                  # Gradio web interface
├── rag_pipeline.py         # RAG system implementation
├── document_processor.py   # PDF processing
├── vector_store.py         # ChromaDB vector store
├── llm_handler.py          # OpenAI LLM integration
├── utils.py               # Utility functions
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── SETUP.md              # This setup guide
├── documents/            # HR policy documents (PDFs)
├── chroma_db/            # Vector database (auto-created)
└── logs/                 # Application logs (auto-created)
```

## Support

For technical support or questions:
1. Check the logs in the `logs/` directory
2. Review this setup guide
3. Ensure all prerequisites are met
4. Contact your system administrator

## Security Notes

- Keep your OpenAI API key secure and never commit it to version control
- The .env file is excluded from git by default
- HR documents may contain sensitive information - ensure proper access controls
- Use HTTPS in production environments