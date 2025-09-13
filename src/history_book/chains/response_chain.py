"""Response chain for LLM generation using LCEL."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from history_book.data_models.entities import ChatMessage
from history_book.llm.interfaces.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class ResponseChain:
    """Chain that handles LLM response generation."""
    
    def __init__(self, llm_provider: LLMInterface):
        """
        Initialize the response chain.
        
        Args:
            llm_provider: The LLM provider to use for generation
        """
        self.llm_provider = llm_provider
    
    def build(self, **llm_params: Any) -> Runnable:
        """
        Build the response generation chain.
        
        Args:
            **llm_params: Additional parameters to pass to the LLM
            
        Returns:
            Runnable chain for response generation
        """
        async def generate_wrapper(input_data):
            return await self._generate_response(input_data, **llm_params)
        
        return RunnableLambda(generate_wrapper)
    
    def build_streaming(self, **llm_params: Any) -> Runnable:
        """
        Build the streaming response generation chain.
        
        Args:
            **llm_params: Additional parameters to pass to the LLM
            
        Returns:
            Runnable chain for streaming response generation
        """
        async def generate_streaming_wrapper(input_data):
            # Note: streaming returns AsyncIterator, not a direct value
            return self._generate_streaming_response(input_data, **llm_params)
        
        return RunnableLambda(generate_streaming_wrapper)
    
    async def _generate_response(self, input_data: dict[str, Any], **llm_params: Any) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            input_data: Input containing 'messages' and optional 'context'
            **llm_params: Additional LLM parameters
            
        Returns:
            Generated response text
        """
        try:
            messages = input_data.get("messages", [])
            context = input_data.get("context")
            
            if not messages:
                raise ValueError("No messages provided for response generation")
            
            # Validate that messages are ChatMessage objects
            if not all(isinstance(msg, ChatMessage) for msg in messages):
                raise ValueError("All messages must be ChatMessage objects")
            
            # Generate response using the LLM provider
            response = await self.llm_provider.generate_response(
                messages=messages, 
                context=context, 
                **llm_params
            )
            
            if not response:
                logger.warning("LLM returned empty response")
                return "I'm sorry, I couldn't generate a response. Please try again."
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise
    
    async def _generate_streaming_response(
        self, input_data: dict[str, Any], **llm_params: Any
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using the LLM.
        
        Args:
            input_data: Input containing 'messages' and optional 'context'
            **llm_params: Additional LLM parameters
            
        Yields:
            Response text chunks
        """
        try:
            messages = input_data.get("messages", [])
            context = input_data.get("context")
            
            if not messages:
                raise ValueError("No messages provided for response generation")
            
            # Validate that messages are ChatMessage objects
            if not all(isinstance(msg, ChatMessage) for msg in messages):
                raise ValueError("All messages must be ChatMessage objects")
            
            # Generate streaming response using the LLM provider
            has_yielded_content = False
            async for chunk in self.llm_provider.generate_stream_response(
                messages=messages, 
                context=context, 
                **llm_params
            ):
                if chunk:
                    has_yielded_content = True
                    yield chunk
            
            # If no content was yielded, provide a fallback
            if not has_yielded_content:
                logger.warning("LLM returned no streaming content")
                yield "I'm sorry, I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Streaming response generation failed: {e}")
            raise