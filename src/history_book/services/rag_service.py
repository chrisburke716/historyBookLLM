"""RAG service with direct LangChain integration using LCEL."""

import logging
from typing import Any, NamedTuple

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from history_book.data_models.entities import Paragraph, ChatMessage, MessageRole
from history_book.database.repositories import BookRepositoryManager
from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError, LLMConnectionError, LLMValidationError
from history_book.llm.utils import format_messages_for_llm, format_context_for_llm

logger = logging.getLogger(__name__)


class RAGResult(NamedTuple):
    """Result from RAG service execution."""
    response: str
    source_paragraphs: list[Paragraph]


class RagService:
    """Service that orchestrates RAG operations using LCEL chains."""

    def __init__(self, llm_config: LLMConfig, repository_manager: BookRepositoryManager):
        """
        Initialize the RAG service with direct LangChain integration.

        Args:
            llm_config: Configuration for the LLM
            repository_manager: Repository manager for document retrieval
        """
        self.config = llm_config
        self.repository_manager = repository_manager

        # Create the LangChain model
        self.chat_model = self._create_chat_model()

        # Build LCEL chains
        self.rag_chain = self._build_rag_chain()
        self.simple_chain = self._build_simple_chain()

    def _create_chat_model(self):
        """Create the appropriate LangChain chat model based on configuration."""
        try:
            if self.config.provider == "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=self.config.model_name,
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    **self.config.provider_kwargs,
                )
            elif self.config.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=self.config.model_name,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    **self.config.provider_kwargs,
                )
            else:
                raise LLMValidationError(f"Unsupported provider: {self.config.provider}")

        except ImportError as e:
            raise LLMConnectionError(f"Missing dependency for {self.config.provider}: {e}") from e
        except Exception as e:
            raise LLMConnectionError(f"Failed to create chat model: {e}") from e

    def _build_rag_chain(self) -> Runnable:
        """Build LCEL chain for RAG scenarios with context."""
        # Create prompt template for RAG
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{context}\n\nQuestion: {query}")
        ])

        # Build the chain: prompt -> chat model -> string output parser
        return rag_prompt | self.chat_model | StrOutputParser()

    def _build_simple_chain(self) -> Runnable:
        """Build LCEL chain for non-RAG scenarios without context."""
        # Create prompt template for simple chat
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{query}")
        ])

        # Build the chain: prompt -> chat model -> string output parser
        return simple_prompt | self.chat_model | StrOutputParser()

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
        Generate a response using RAG pipeline with LCEL chains.

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
        try:
            # Retrieve context if enabled
            source_paragraphs = []
            if enable_retrieval:
                source_paragraphs = await self._retrieve_context(
                    query, min_results, max_results, similarity_cutoff, retrieval_strategy
                )

            # Convert chat messages to LangChain format
            formatted_messages = self._convert_to_langchain_messages(messages)

            # Choose the appropriate chain and prepare input
            if enable_retrieval and source_paragraphs:
                # Use RAG chain with context
                context = self._format_context(source_paragraphs)
                formatted_context = format_context_for_llm(context, self.config.max_context_length)

                response = await self.rag_chain.ainvoke({
                    "chat_history": formatted_messages,
                    "context": formatted_context,
                    "query": query
                })
            else:
                # Use simple chain without context
                response = await self.simple_chain.ainvoke({
                    "chat_history": formatted_messages,
                    "query": query
                })

            return RAGResult(response=response, source_paragraphs=source_paragraphs)

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMError(f"Response generation failed: {e}") from e

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
        Generate a streaming response using RAG pipeline with LCEL chains.

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
        try:
            # Retrieve context if enabled
            source_paragraphs = []
            if enable_retrieval:
                source_paragraphs = await self._retrieve_context(
                    query, min_results, max_results, similarity_cutoff, retrieval_strategy
                )

            # Convert chat messages to LangChain format
            formatted_messages = self._convert_to_langchain_messages(messages)

            # Choose the appropriate chain and prepare input
            if enable_retrieval and source_paragraphs:
                # Use RAG chain with context
                context = self._format_context(source_paragraphs)
                formatted_context = format_context_for_llm(context, self.config.max_context_length)

                stream = self.rag_chain.astream({
                    "chat_history": formatted_messages,
                    "context": formatted_context,
                    "query": query
                })
            else:
                # Use simple chain without context
                stream = self.simple_chain.astream({
                    "chat_history": formatted_messages,
                    "query": query
                })

            return stream, source_paragraphs

        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}")
            raise LLMError(f"Streaming response generation failed: {e}") from e

    def _convert_to_langchain_messages(self, messages: list[ChatMessage]) -> list:
        """Convert ChatMessage objects to LangChain message format."""
        # Format messages using existing utility
        formatted_messages = format_messages_for_llm(
            messages,
            system_message=None,  # System message handled by prompt template
            max_messages=self.config.max_conversation_length,
        )

        lc_messages = []
        for msg in formatted_messages:
            # Skip system messages - handled by prompt template
            if msg.role == MessageRole.SYSTEM:
                continue
            elif msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))

        return lc_messages

    async def _retrieve_context(
        self,
        query: str,
        min_results: int,
        max_results: int,
        similarity_cutoff: float,
        retrieval_strategy: str
    ) -> list[Paragraph]:
        """
        Retrieve relevant context documents.

        Args:
            query: User query
            min_results: Minimum number of documents to retrieve
            max_results: Maximum number of documents to retrieve
            similarity_cutoff: Similarity threshold
            retrieval_strategy: Strategy for retrieval

        Returns:
            List of relevant paragraphs
        """
        try:
            if not query:
                logger.warning("No query provided for context retrieval")
                return []

            # Use similarity search strategy
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