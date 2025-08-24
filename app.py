"""Gradio interface for the AI HR Assistant."""

import gradio as gr
import os
from typing import List, Tuple, Optional, Dict
import logging

from rag_pipeline import RAGPipeline
from config import Config
from utils import setup_logging

logger = setup_logging(__name__)

class HRAssistantApp:
    """Gradio application for the HR Assistant."""
    
    def __init__(self):
        """Initialize the HR Assistant application."""
        self.rag_pipeline = RAGPipeline()
        self.conversation_history = []
        logger.info("HR Assistant App initialized")
    
    def initialize_system(self) -> str:
        """Initialize the RAG system with documents."""
        try:
            # Check if documents directory exists and has files
            if not os.path.exists(Config.DOCUMENTS_PATH):
                return f"❌ Documents directory not found: {Config.DOCUMENTS_PATH}\nPlease create the directory and add HR policy PDF files."
            
            pdf_files = [f for f in os.listdir(Config.DOCUMENTS_PATH) if f.endswith('.pdf')]
            if not pdf_files:
                return f"❌ No PDF files found in {Config.DOCUMENTS_PATH}\nPlease add HR policy PDF files to get started."
            
            # Initialize knowledge base
            success = self.rag_pipeline.initialize_knowledge_base()
            
            if success:
                kb_info = self.rag_pipeline.get_knowledge_base_info()
                return f"""✅ System initialized successfully!
                
📊 Knowledge Base Info:
- Documents loaded: {kb_info['document_count']} chunks
- Source files: {len(pdf_files)} PDF files
- Collection: {kb_info['collection_name']}
- LLM Model: {kb_info['llm_model']['model_name']}

🤖 The HR Assistant is ready to answer your questions!"""
            else:
                return "❌ Failed to initialize the system. Please check the logs and ensure your OpenAI API key is configured."
                
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return f"❌ System initialization error: {str(e)}"
    
    def chat(self, message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
        """Handle chat interaction.
        
        Args:
            message: User's message
            history: Gradio chat history format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            Tuple of (empty string, updated history)
        """
        if not message.strip():
            return "", history
        
        try:
            # Convert Gradio history to our format
            conversation_context = []
            for i in range(0, len(history) - 1, 2):  # Process pairs of messages
                if i + 1 < len(history):
                    user_msg = history[i]
                    assistant_msg = history[i + 1]
                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        conversation_context.append({
                            "question": user_msg.get("content", ""),
                            "answer": assistant_msg.get("content", "")
                        })
            
            # Query the RAG system
            result = self.rag_pipeline.query_with_history(message, conversation_context[-5:])  # Last 5 exchanges
            
            # Format response with sources
            response = result["answer"]
            
            # Add sources information if available
            if result.get("sources") and len(result["sources"]) > 0:
                response += "\n\n📚 **Sources:**\n"
                for i, source in enumerate(result["sources"][:3], 1):  # Show max 3 sources
                    response += f"{i}. {source['file']} (Page {source['page']})\n"
            
            # Add warning if no relevant docs found
            if result.get("warning"):
                response = "⚠️ " + response
            
            # Update history with new message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            # Store in conversation history
            self.conversation_history.append({
                "question": message,
                "answer": response,
                "sources": result.get("sources", [])
            })
            
            # Limit conversation history
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return "", history
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            error_response = "❌ I encountered an error processing your question. Please try again or contact HR directly."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return "", history
    
    
    
    def clear_conversation(self) -> Tuple[List, str]:
        """Clear the conversation history.
        
        Returns:
            Tuple of (empty chat history, status message)
        """
        self.conversation_history = []
        self.rag_pipeline.clear_chat_history()
        return [], "🧹 Conversation history cleared!"
    
    
    def create_interface(self) -> gr.Blocks:
        """Create the simplified Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Nestlé AI HR Assistant",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 900px;
                margin: auto;
            }
            .chat-container {
                height: 600px;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🤖 Nestlé AI-Powered HR Assistant
            
            Welcome to your intelligent HR companion! I can help you understand Nestlé's HR policies and procedures.
            Ask me questions about leave policies, benefits, performance reviews, and more.
            """)
            
            # System Status
            with gr.Row():
                with gr.Column(scale=2):
                    system_status = gr.Textbox(
                        label="System Status",
                        value="Click 'Initialize System' to get started",
                        interactive=False,
                        max_lines=2
                    )
                with gr.Column(scale=1):
                    init_btn = gr.Button("🚀 Initialize System", variant="primary")
            
            # Chat Interface
            chatbot = gr.Chatbot(
                label="HR Assistant",
                elem_classes=["chat-container"],
                show_label=False,
                type="messages"
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me about HR policies, benefits, leave procedures...",
                    label="Your Question",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear Chat")
                gr.HTML("<div style='flex-grow: 1'></div>")  # Spacer
            
            # Example questions
            gr.Markdown("""
            **💡 Example Questions:**
            - "What are the main topics in the HR policy?"
            - "What does this document say about employee procedures?"
            - "Tell me about the policies mentioned"
            - "What guidelines are covered in this document?"
            """)
            
            # Event Handlers
            
            # Initialize system
            init_btn.click(
                self.initialize_system,
                outputs=[system_status]
            )
            
            # Chat functionality
            msg_input.submit(
                self.chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                self.chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            # Clear chat
            clear_btn.click(
                self.clear_conversation,
                outputs=[chatbot, system_status]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments for gr.launch()
        """
        # Validate configuration
        try:
            Config.validate_config()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            print(f"❌ Configuration Error: {e}")
            print("Please check your .env file and ensure OPENAI_API_KEY is set.")
            return
        
        # Set default launch parameters
        launch_params = {
            "server_port": Config.GRADIO_PORT,
            "share": Config.GRADIO_SHARE,
            "show_error": True,
            "quiet": False
        }
        
        # Update with user-provided parameters
        launch_params.update(kwargs)
        
        # Create and launch interface
        interface = self.create_interface()
        
        logger.info(f"Launching HR Assistant on port {launch_params['server_port']}")
        print(f"🚀 Launching Nestlé AI HR Assistant...")
        print(f"📱 Access the app at: http://localhost:{launch_params['server_port']}")
        
        interface.launch(**launch_params)

def create_app() -> HRAssistantApp:
    """Create and return an instance of the HR Assistant app."""
    return HRAssistantApp()