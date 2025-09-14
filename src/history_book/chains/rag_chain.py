"""RAG chain for retrieval-augmented generation using LCEL."""

import logging
from typing import Any

from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from history_book.data_models.entities import Paragraph
from history_book.database.repositories import BookRepositoryManager
from history_book.chains.response_chain import ResponseChain

logger = logging.getLogger(__name__)


class RAGChain:
    """Chain that orchestrates RAG pipeline: retrieve → format → generate response."""
    
    def __init__(self, response_chain: ResponseChain, repository_manager: BookRepositoryManager):
        """
        Initialize the RAG chain.
        
        Args:
            response_chain: The response chain to use for LLM generation
            repository_manager: Repository manager for document retrieval
        """
        self.response_chain = response_chain
        self.repository_manager = repository_manager
    
    def build(
        self,
        min_results: int = 5,
        max_results: int = 40,
        similarity_cutoff: float = 0.4,
        retrieval_strategy: str = "similarity_search",
        enable_retrieval: bool = True,
        **llm_params: Any
    ) -> Runnable:
        """
        Build the RAG chain.
        
        Args:
            min_results: Minimum number of documents to retrieve
            max_results: Maximum number of documents to retrieve
            similarity_cutoff: Similarity threshold for document retrieval
            retrieval_strategy: Strategy for document retrieval
            enable_retrieval: Whether to enable document retrieval
            **llm_params: Additional parameters to pass to the LLM
            
        Returns:
            Runnable chain for RAG pipeline
        """
        if not enable_retrieval:
            # Simple non-RAG chain - just pass through to response generation
            return (
                RunnableLambda(self._format_messages_only)
                | RunnableLambda(self._generate_with_metadata_no_retrieval)
            )
        
        # Full RAG pipeline
        async def retrieve_context_wrapper(input_data):
            return await self._retrieve_context(
                input_data, min_results, max_results, similarity_cutoff, retrieval_strategy
            )
        
        return (
            RunnableParallel({
                "context": RunnableLambda(retrieve_context_wrapper),
                "input": RunnablePassthrough()
            })
            | RunnableLambda(self._format_messages_with_context)
            | RunnableLambda(self._generate_with_metadata)
        )
    
    def build_streaming(
        self,
        min_results: int = 5,
        max_results: int = 40,
        similarity_cutoff: float = 0.4,
        retrieval_strategy: str = "similarity_search",
        enable_retrieval: bool = True,
        **llm_params: Any
    ) -> Runnable:
        """
        Build the streaming RAG chain.
        
        Args:
            min_results: Minimum number of documents to retrieve
            max_results: Maximum number of documents to retrieve
            similarity_cutoff: Similarity threshold for document retrieval
            retrieval_strategy: Strategy for document retrieval
            enable_retrieval: Whether to enable document retrieval
            **llm_params: Additional parameters to pass to the LLM
            
        Returns:
            Runnable chain for streaming RAG pipeline
        """
        if not enable_retrieval:
            # Simple non-RAG chain - just pass through to response generation
            return (
                RunnableLambda(self._format_messages_only)
                | self.response_chain.build_streaming(**llm_params)
            )
        
        # Full RAG pipeline with streaming
        async def retrieve_context_wrapper_streaming(input_data):
            return await self._retrieve_context(
                input_data, min_results, max_results, similarity_cutoff, retrieval_strategy
            )
        
        return (
            RunnableParallel({
                "context": RunnableLambda(retrieve_context_wrapper_streaming),
                "input": RunnablePassthrough()
            })
            | RunnableLambda(self._format_messages_with_context)
            | self.response_chain.build_streaming(**llm_params)
        )
    
    async def _retrieve_context(
        self,
        input_data: dict[str, Any],
        min_results: int,
        max_results: int,
        similarity_cutoff: float,
        retrieval_strategy: str
    ) -> list[Paragraph]:
        """
        Retrieve relevant context documents.
        
        Args:
            input_data: Input containing 'query' and other data
            min_results: Minimum number of documents to retrieve
            max_results: Maximum number of documents to retrieve
            similarity_cutoff: Similarity threshold
            retrieval_strategy: Strategy for retrieval
            
        Returns:
            List of relevant paragraphs
        """
        try:
            query = input_data.get("query", "")
            if not query:
                logger.warning("No query provided for context retrieval")
                return []
            
            # Use the same retrieval logic as the original ChatService
            if retrieval_strategy == "similarity_search":
                search_result = (
                    self.repository_manager.paragraphs.similarity_search_by_text(
                        query_text=query,
                        limit=max_results,
                        threshold=similarity_cutoff
                    )
                )
                
                # Fallback if we don't get enough results
                if len(search_result) < min_results:
                    search_result = (
                        self.repository_manager.paragraphs.similarity_search_by_text(
                            query_text=query, limit=min_results
                        )
                    )
            else:
                # For future extension - other retrieval strategies
                logger.warning(f"Unknown retrieval strategy: {retrieval_strategy}, using similarity_search")
                search_result = (
                    self.repository_manager.paragraphs.similarity_search_by_text(
                        query_text=query, limit=max_results
                    )
                )
            
            paragraphs = [para[0] for para in search_result] if search_result else []
            
            if paragraphs:
                logger.info(f"Retrieved {len(paragraphs)} context paragraphs for query")
            else:
                logger.warning("No context paragraphs retrieved")
            
            return paragraphs
            
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
            return []
    
    def _format_messages_only(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Format messages without context for non-RAG scenarios.
        
        Args:
            input_data: Input containing 'messages'
            
        Returns:
            Formatted data for response generation
        """
        messages = input_data.get("messages", [])
        return {
            "messages": messages,
            "context": None
        }
    
    def _format_messages_with_context(self, parallel_output: dict[str, Any]) -> dict[str, Any]:
        """
        Format messages with retrieved context.
        
        Args:
            parallel_output: Output from parallel execution containing 'context' and 'input'
            
        Returns:
            Formatted data for response generation with context
        """
        context_paragraphs = parallel_output.get("context", [])
        input_data = parallel_output.get("input", {})
        messages = input_data.get("messages", [])
        
        # Format context if we have paragraphs
        context_text = None
        if context_paragraphs:
            context_text = self._format_context(context_paragraphs)
        
        return {
            "messages": messages,
            "context": context_text,
            "source_paragraphs": context_paragraphs  # Include for potential citation use
        }
    
    def _format_context(self, paragraphs: list[Paragraph]) -> str:
        """
        Format retrieved paragraphs as context for the LLM.
        
        Args:
            paragraphs: Retrieved paragraphs
            
        Returns:
            Formatted context string
        """
        if not paragraphs:
            return ""
        
        context_parts = []
        for i, para in enumerate(paragraphs, 1):
            # Include page information for citation
            context_parts.append(f"[Source {i}, Page {para.page}]: {para.text}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_with_metadata(self, formatted_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate response and include metadata like source paragraphs.
        
        Args:
            formatted_data: Formatted data from message formatting step
            
        Returns:
            Dictionary containing response and metadata
        """
        # Extract the data we need for the response chain
        response_input = {
            "messages": formatted_data.get("messages", []),
            "context": formatted_data.get("context")
        }
        
        # Generate the response using the response chain
        response_chain = self.response_chain.build()
        response_text = await response_chain.ainvoke(response_input)
        
        # Return structured output with both response and metadata
        return {
            "response": response_text,
            "source_paragraphs": formatted_data.get("source_paragraphs", [])
        }
    
    async def _generate_with_metadata_no_retrieval(self, formatted_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate response without retrieval and include empty metadata.
        
        Args:
            formatted_data: Formatted data from message formatting step
            
        Returns:
            Dictionary containing response and empty metadata
        """
        # Extract the data we need for the response chain
        response_input = {
            "messages": formatted_data.get("messages", []),
            "context": formatted_data.get("context")
        }
        
        # Generate the response using the response chain
        response_chain = self.response_chain.build()
        response_text = await response_chain.ainvoke(response_input)
        
        # Return structured output with response and empty source paragraphs
        return {
            "response": response_text,
            "source_paragraphs": []
        }