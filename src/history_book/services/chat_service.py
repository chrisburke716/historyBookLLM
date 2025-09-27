"""Chat service for handling conversational interactions with historical documents."""

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from langsmith import traceable

from history_book.data_models.entities import (
    ChatMessage,
    ChatSession,
    MessageRole,
)
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager
from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError
from history_book.services.rag_service import RagService

logger = logging.getLogger(__name__)

CONTEXT_MIN_RESULTS = 5
CONTEXT_MAX_RESULTS = 40
CONTEXT_SIMILARITY_CUTOFF = 0.4


class ChatService:
    """Service for chat operations with retrieval-augmented generation."""

    def __init__(
        self,
        config: WeaviateConfig | None = None,
        llm_config: LLMConfig | None = None,
        min_context_results: int = CONTEXT_MIN_RESULTS,
        max_context_results: int = CONTEXT_MAX_RESULTS,
        context_similarity_cutoff: float = CONTEXT_SIMILARITY_CUTOFF,
        retrieval_strategy: str = "similarity_search",
    ):
        """
        Initialize the chat service.

        Args:
            config: Database configuration. If None, loads from environment.
            llm_config: LLM configuration. If None, loads from environment.
            min_context_results: Minimum number of context documents to retrieve.
            max_context_results: Maximum number of context documents to retrieve.
            context_similarity_cutoff: Similarity threshold for context retrieval.
            retrieval_strategy: Strategy for document retrieval.
        """
        if config is None:
            config = WeaviateConfig.from_environment()
        self.config = config
        self.repository_manager = BookRepositoryManager(config)

        # Initialize LLM configuration
        self.llm_config = llm_config or LLMConfig.from_environment()
        self.llm_config.validate()
        logger.info(
            f"Using LLM provider: {self.llm_config.provider}/{self.llm_config.model_name}"
        )

        # Store retrieval configuration
        self.min_context_results = min_context_results
        self.max_context_results = max_context_results
        self.context_similarity_cutoff = context_similarity_cutoff
        self.retrieval_strategy = retrieval_strategy

        # Initialize RAG service
        self.rag_service = RagService(self.llm_config, self.repository_manager)

    async def create_session(self, title: str | None = None) -> ChatSession:
        """
        Create a new chat session.

        Args:
            title: Optional title for the session

        Returns:
            Created chat session

        Raises:
            DatabaseError: If session creation fails
        """
        try:
            session = ChatSession(title=title)
            session_id = self.repository_manager.chat_sessions.create(session)
            session.id = session_id
            logger.info(f"Created chat session {session_id}")
            return session
        except Exception as e:
            logger.error(f"Failed to create chat session: {e}")
            raise

    async def get_session(self, session_id: str) -> ChatSession | None:
        """
        Retrieve a chat session by ID.

        Args:
            session_id: Session ID

        Returns:
            Chat session or None if not found
        """
        try:
            return self.repository_manager.chat_sessions.get_by_id(session_id)
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def list_recent_sessions(self, limit: int = 10) -> list[ChatSession]:
        """
        List recent chat sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of recent chat sessions
        """
        try:
            return self.repository_manager.chat_sessions.find_recent_sessions(limit)
        except Exception as e:
            logger.error(f"Failed to list recent sessions: {e}")
            return []

    async def get_session_messages(self, session_id: str) -> list[ChatMessage]:
        """
        Get all messages for a session.

        Args:
            session_id: Session ID

        Returns:
            List of messages ordered by timestamp
        """
        try:
            return self.repository_manager.chat_messages.find_by_session(session_id)
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []

    @traceable(name="Chat Service: Send Message")
    async def send_message(
        self,
        session_id: str,
        user_message: str,
        enable_retrieval: bool = True,
    ) -> ChatMessage:
        """
        Send a message and get AI response with optional retrieval.

        Args:
            session_id: Session ID
            user_message: User's message content
            max_context_paragraphs: Maximum paragraphs to retrieve for context
            enable_retrieval: Whether to enable retrieval augmentation

        Returns:
            AI response message

        Raises:
            LLMError: If LLM generation fails
            DatabaseError: If database operations fail
        """
        try:
            # Prepare user message and chat history
            user_msg, chat_history = await self._prepare_message_and_history(
                session_id, user_message
            )

            # Generate response using RAG service
            rag_result = await self.rag_service.generate_response(
                query=user_message,
                messages=chat_history,
                min_results=self.min_context_results,
                max_results=self.max_context_results,
                similarity_cutoff=self.context_similarity_cutoff,
                retrieval_strategy=self.retrieval_strategy,
                enable_retrieval=enable_retrieval,
            )

            # Save AI message and update session
            return await self._save_ai_message_and_update_session(
                session_id, rag_result.response, rag_result.source_paragraphs
            )

        except LLMError:
            # Re-raise LLM errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def send_message_stream(
        self,
        session_id: str,
        user_message: str,
        enable_retrieval: bool = True,
    ) -> AsyncIterator[str]:
        """
        Send a message and get streaming AI response with optional retrieval.

        Args:
            session_id: Session ID
            user_message: User's message content
            max_context_paragraphs: Maximum paragraphs to retrieve for context
            enable_retrieval: Whether to enable retrieval augmentation

        Yields:
            AI response chunks

        Note:
            The final complete response is saved to the database after streaming completes.
        """
        try:
            # Prepare user message and chat history
            user_msg, chat_history = await self._prepare_message_and_history(
                session_id, user_message
            )

            # Get streaming response from RAG service
            stream, context_paragraphs = await self.rag_service.stream_response(
                query=user_message,
                messages=chat_history,
                min_results=self.min_context_results,
                max_results=self.max_context_results,
                similarity_cutoff=self.context_similarity_cutoff,
                retrieval_strategy=self.retrieval_strategy,
                enable_retrieval=enable_retrieval,
            )

            # Collect response chunks while yielding them to the client
            response_chunks = []
            async for chunk in stream:
                response_chunks.append(chunk)
                yield chunk

            # Save complete AI response and update session
            complete_response = "".join(response_chunks)
            await self._save_ai_message_and_update_session(
                session_id, complete_response, context_paragraphs
            )

        except Exception as e:
            logger.error(f"Failed to stream message: {e}")
            raise

    async def _prepare_message_and_history(
        self, session_id: str, user_message: str
    ) -> tuple[ChatMessage, list[ChatMessage]]:
        """
        Create user message and get chat history.

        Args:
            session_id: Session ID
            user_message: User's message content

        Returns:
            Tuple of (user_message, chat_history)
        """
        # 1. Create and save user message
        user_msg = ChatMessage(
            content=user_message, role=MessageRole.USER, session_id=session_id
        )
        user_msg_id = self.repository_manager.chat_messages.create(user_msg)
        user_msg.id = user_msg_id
        logger.info(f"Saved user message {user_msg_id} to session {session_id}")

        # 2. Get recent chat history
        chat_history = await self.get_session_messages(session_id)

        # Include the new user message in history for LLM
        if user_msg not in chat_history:
            chat_history.append(user_msg)

        return user_msg, chat_history

    async def _save_ai_message_and_update_session(
        self, session_id: str, ai_response: str, context_paragraphs: list
    ) -> ChatMessage:
        """
        Save AI message and update session timestamp.

        Args:
            session_id: Session ID
            ai_response: AI response text
            context_paragraphs: Retrieved context paragraphs

        Returns:
            Created AI message
        """
        # Create and save AI message
        ai_msg = ChatMessage(
            content=ai_response,
            role=MessageRole.ASSISTANT,
            session_id=session_id,
            retrieved_paragraphs=[p.id for p in context_paragraphs]
            if context_paragraphs
            else None,
        )
        ai_msg_id = self.repository_manager.chat_messages.create(ai_msg)
        ai_msg.id = ai_msg_id

        # Update session timestamp
        await self._update_session_timestamp(session_id)

        logger.info(f"Generated AI response {ai_msg_id} for session {session_id}")
        return ai_msg

    async def _update_session_timestamp(self, session_id: str) -> None:
        """
        Update the session's updated_at timestamp.

        Args:
            session_id: Session ID
        """
        try:
            session = await self.get_session(session_id)
            if session:
                # Update using dictionary format that the repository expects
                updates = {"updated_at": datetime.now(UTC)}
                self.repository_manager.chat_sessions.update(session_id, updates)
        except Exception as e:
            logger.warning(f"Failed to update session timestamp: {e}")

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session and all its messages.

        Args:
            session_id: Session ID

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete all messages first
            messages = await self.get_session_messages(session_id)
            for message in messages:
                if message.id:
                    self.repository_manager.chat_messages.delete(message.id)

            # Delete the session
            self.repository_manager.chat_sessions.delete(session_id)
            logger.info(f"Deleted session {session_id} and {len(messages)} messages")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def search_messages(
        self, query: str, session_id: str | None = None, limit: int = 10
    ) -> list[ChatMessage]:
        """
        Search for messages by content.

        Args:
            query: Search query
            session_id: Optional session to limit search to
            limit: Maximum number of results

        Returns:
            List of matching messages
        """
        try:
            return self.repository_manager.chat_messages.search_message_content(
                query_text=query, session_id=session_id, limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []

    def close(self) -> None:
        """Close all repository connections."""
        try:
            self.repository_manager.close_all()
            logger.info("Chat service closed successfully")
        except Exception as e:
            logger.error(f"Error closing chat service: {e}")
