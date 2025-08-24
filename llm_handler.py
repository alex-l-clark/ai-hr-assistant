"""LLM handler for OpenAI GPT integration and response generation."""

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from config import Config
from utils import setup_logging

logger = setup_logging(__name__)

class LLMHandler:
    """Handles LLM interactions for the HR Assistant."""
    
    def __init__(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        """Initialize the LLM handler.
        
        Args:
            model_name: OpenAI model to use
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name or Config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else Config.TEMPERATURE
        self.max_tokens = max_tokens or Config.MAX_TOKENS
        
        # Initialize the chat model
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"LLMHandler initialized with model={self.model_name}, temp={self.temperature}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for HR assistant queries.
        
        Returns:
            ChatPromptTemplate for HR assistant
        """
        system_message = """You are Nestlé's AI-powered HR Assistant. Your role is to help employees understand HR policies and procedures by providing accurate, helpful, and professional responses based on the company's official HR documentation.

Guidelines for responses:
1. Always base your answers on the provided context from official HR documents
2. Be professional, clear, and concise
3. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation
4. For sensitive topics (disciplinary actions, legal issues), advise consulting with HR directly
5. Include relevant policy references when applicable
6. Use a friendly but professional tone appropriate for workplace communication

If you cannot find relevant information in the provided context, politely explain that you need more specific information from HR documentation to provide an accurate answer."""

        human_template = """Based on the following HR policy context, please answer the employee's question:

Context from HR Documents:
{context}

Employee Question: {question}

Please provide a helpful and accurate response based on the context above. If the context doesn't fully address the question, acknowledge this and suggest appropriate next steps."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template)
        ])
    
    def generate_response(self, question: str, context: str) -> str:
        """Generate a response to a user question with given context.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Generated response string
        """
        try:
            # Format the prompt with question and context
            messages = self.prompt_template.format_messages(
                question=question,
                context=context
            )
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.info(f"Generated response for question: '{question[:50]}...'")
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_error_response(str(e))
    
    def generate_followup_response(self, question: str, context: str, chat_history: List[Dict[str, str]]) -> str:
        """Generate a response considering chat history.
        
        Args:
            question: Current user question
            context: Retrieved context from documents
            chat_history: Previous conversation history
            
        Returns:
            Generated response string
        """
        try:
            # Create extended prompt with chat history
            system_message = self.prompt_template.messages[0].prompt.template
            
            # Add chat history to context
            history_context = ""
            if chat_history:
                history_context = "\n\nPrevious conversation context:\n"
                for exchange in chat_history[-3:]:  # Only last 3 exchanges
                    history_context += f"Q: {exchange.get('question', '')}\nA: {exchange.get('answer', '')}\n"
            
            extended_context = context + history_context
            
            messages = self.prompt_template.format_messages(
                question=question,
                context=extended_context
            )
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.info(f"Generated follow-up response with chat history")
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error generating follow-up response: {str(e)}")
            return self._get_error_response(str(e))
    
    def _get_error_response(self, error_msg: str) -> str:
        """Generate a user-friendly error response.
        
        Args:
            error_msg: Technical error message
            
        Returns:
            User-friendly error response
        """
        return """I apologize, but I'm experiencing technical difficulties at the moment. 
Please try rephrasing your question or contact the HR department directly for immediate assistance.

If the issue persists, please check that your OpenAI API key is properly configured."""
    
    def validate_question(self, question: str) -> tuple[bool, str]:
        """Validate if a question is appropriate for the HR assistant.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not question or len(question.strip()) < 3:
            return False, "Please provide a more detailed question about HR policies or procedures."
        
        # Check for inappropriate content (basic filtering)
        inappropriate_keywords = ['hack', 'bypass', 'illegal', 'fraud']
        question_lower = question.lower()
        
        for keyword in inappropriate_keywords:
            if keyword in question_lower:
                return False, "I can only assist with legitimate HR policy questions."
        
        # Check if it's HR-related (basic check)
        hr_keywords = [
            'policy', 'leave', 'vacation', 'sick', 'benefits', 'salary', 'promotion',
            'training', 'performance', 'disciplinary', 'harassment', 'compliance',
            'onboarding', 'termination', 'resignation', 'hr', 'human resources',
            'employee', 'workplace', 'code of conduct', 'ethics'
        ]
        
        has_hr_keyword = any(keyword in question_lower for keyword in hr_keywords)
        
        if not has_hr_keyword and len(question.split()) > 3:
            # For longer questions without HR keywords, suggest HR focus
            return True, "I'll do my best to help with your question. For the most accurate information, please ensure your question relates to HR policies or workplace procedures."
        
        return True, ""
    
    def summarize_documents(self, documents: List[Any]) -> str:
        """Generate a summary of retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Summary string
        """
        if not documents:
            return "No relevant documents found."
        
        try:
            # Combine document content
            combined_content = "\n\n".join([doc.page_content for doc in documents[:3]])  # Limit to first 3
            
            summary_prompt = """Please provide a brief summary of the following HR policy content:

{content}

Summary:"""
            
            messages = [
                SystemMessage(content="You are an HR policy summarization assistant."),
                HumanMessage(content=summary_prompt.format(content=combined_content))
            ]
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing documents: {str(e)}")
            return f"Found {len(documents)} relevant policy documents."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "OpenAI"
        }