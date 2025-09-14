"""RAG service for retrieval-augmented generation."""

import logging
from typing import Any, NamedTuple

from history_book.data_models.entities import Paragraph, ChatMessage
from history_book.database.repositories import BookRepositoryManager
from history_book.chains.response_chain import ResponseChain

logger = logging.getLogger(__name__)


class RAGResult(NamedTuple):
    """Result from RAG service execution."""
    response: str
    source_paragraphs: list[Paragraph]


class RagService:
    """Service that orchestrates RAG operations: retrieve → format → generate response."""

    def __init__(self, response_chain: ResponseChain, repository_manager: BookRepositoryManager):
        """
        Initialize the RAG service.

        Args:
            response_chain: The response chain to use for LLM generation
            repository_manager: Repository manager for document retrieval
        """
        self.response_chain = response_chain
        self.repository_manager = repository_manager

    async def generate_response(
        self,
        query: str,
        messages: list[ChatMessage],
        min_results: int = 5,
        max_results: int = 40,
        similarity_cutoff: float = 0.4,
        retrieval_strategy: str = "similarity_search",
        enable_retrieval: bool = True,
        **llm_params: Any
    ) -> RAGResult:
        """
        Generate a response using RAG pipeline.

        Args:
            query: User query for retrieval
            messages: Chat message history
            min_results: Minimum number of documents to retrieve
            max_results: Maximum number of documents to retrieve
            similarity_cutoff: Similarity threshold for document retrieval
            retrieval_strategy: Strategy for document retrieval
            enable_retrieval: Whether to enable document retrieval
            **llm_params: Additional parameters to pass to the LLM

        Returns:
            RAGResult containing response and source paragraphs
        """
        # Retrieve context if enabled
        source_paragraphs = []
        if enable_retrieval:
            source_paragraphs = await self._retrieve_context(
                {"query": query}, min_results, max_results, similarity_cutoff, retrieval_strategy
            )

        # Format context for LLM
        context = self._format_context(source_paragraphs) if source_paragraphs else None

        # Generate response using the response chain (which uses LCEL for LLM operations)
        response_chain = self.response_chain.build(**llm_params)
        response = await response_chain.ainvoke({
            "messages": messages,
            "context": context
        })

        return RAGResult(response=response, source_paragraphs=source_paragraphs)

    async def stream_response(
        self,
        query: str,
        messages: list[ChatMessage],
        min_results: int = 5,
        max_results: int = 40,
        similarity_cutoff: float = 0.4,
        retrieval_strategy: str = "similarity_search",
        enable_retrieval: bool = True,
        **llm_params: Any
    ) -> tuple:
        """
        Generate a streaming response using RAG pipeline.

        Args:
            query: User query for retrieval
            messages: Chat message history
            min_results: Minimum number of documents to retrieve
            max_results: Maximum number of documents to retrieve
            similarity_cutoff: Similarity threshold for document retrieval
            retrieval_strategy: Strategy for document retrieval
            enable_retrieval: Whether to enable document retrieval
            **llm_params: Additional parameters to pass to the LLM

        Returns:
            Tuple of (async_generator, source_paragraphs)
        """
        # Retrieve context if enabled
        source_paragraphs = []
        if enable_retrieval:
            source_paragraphs = await self._retrieve_context(
                {"query": query}, min_results, max_results, similarity_cutoff, retrieval_strategy
            )

        # Format context for LLM
        context = self._format_context(source_paragraphs) if source_paragraphs else None

        # Generate streaming response using the response chain
        response_chain = self.response_chain.build_streaming(**llm_params)
        stream = response_chain.astream({
            "messages": messages,
            "context": context
        })

        return stream, source_paragraphs

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