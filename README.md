# 🤖 Nestlé AI-Powered HR Assistant

An intelligent conversational chatbot that helps employees understand Nestlé's HR policies and procedures using advanced RAG (Retrieval-Augmented Generation) technology.

## 🌟 Features

- **Conversational Interface**: Natural language chat for HR policy questions
- **Document Processing**: Automatic processing of HR policy PDFs
- **Semantic Search**: Intelligent retrieval of relevant policy information
- **Source Citations**: Responses include references to source documents
- **Clean Interface**: Simple, focused chat interface for easy interaction

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Add HR Documents**
   ```bash
   # Place HR policy PDF files in the documents/ folder
   mkdir -p documents
   cp your_hr_policies.pdf documents/
   ```

4. **Run the Application**
   ```bash
   python main.py
   ```

5. **Access the Interface**
   - Open http://localhost:7860 in your browser
   - Click "Initialize System" to load documents
   - Start chatting with your HR assistant!

## 📋 Example Questions

- "What are the main topics in the HR policy?"
- "What does this document say about employee procedures?"
- "Tell me about the policies mentioned"
- "What guidelines are covered in this document?"

## 🏗️ Architecture

The system uses a modular RAG architecture:

- **Document Processing**: PDF loading and text chunking
- **Vector Store**: ChromaDB for semantic document search
- **LLM Integration**: OpenAI GPT-4 for response generation
- **Web Interface**: Gradio for user-friendly interaction

## 📁 Project Structure

```
ai-hr-assistant/
├── main.py                 # Application entry point
├── app.py                  # Gradio web interface
├── rag_pipeline.py         # RAG implementation
├── document_processor.py   # PDF processing
├── vector_store.py         # Vector database management
├── llm_handler.py          # LLM integration
├── utils.py               # Utility functions
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── documents/             # HR policy documents
└── SETUP.md              # Detailed setup guide
```

## 🔧 Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: LLM model (default: gpt-4)
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `MAX_RETRIEVAL_DOCS`: Documents to retrieve (default: 4)

See [SETUP.md](SETUP.md) for complete configuration details.

## 🛠️ Technology Stack

- **Backend**: Python, LangChain
- **Vector Database**: ChromaDB
- **LLM Provider**: OpenAI GPT-4
- **Embeddings**: OpenAI text-embedding-ada-002
- **Web Framework**: Gradio
- **Document Processing**: PyPDF

## 📖 Usage Guide

### Web Interface

- **Simple Chat Interface**: Clean, focused conversational interface
- **System Status**: Shows initialization status and system readiness
- **Example Questions**: Helpful prompts to get started

### Command Line Options

```bash
python main.py --port 8080 --share --debug
```

- `--port`: Custom port (default: 7860)
- `--share`: Create public link
- `--debug`: Enable verbose logging

## 🔒 Security & Privacy

- Secure API key management through environment variables
- Local document processing (no data sent to third parties except OpenAI)
- Configurable access controls
- Audit logging for system operations

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and comments
3. Test new features thoroughly
4. Update documentation as needed

## 📄 License

This project is created for educational and internal use at Nestlé.

## 🆘 Support

For setup assistance, see [SETUP.md](SETUP.md) or check the application logs in the `logs/` directory.

---

**Note**: This AI assistant is designed to provide information based on official HR documents. For sensitive matters or official decisions, always consult directly with the HR department.
