"""Chat service for handling conversational interactions with historical documents."""

import logging
from typing import List, Optional, AsyncIterator
from datetime import datetime, timezone

from ..data_models.entities import ChatSession, ChatMessage, MessageRole, Paragraph
from ..database.repositories import BookRepositoryManager
from ..database.config import WeaviateConfig
from ..llm import LLMInterface, MockLLMProvider, LLMConfig
from ..llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class ChatService:
    """Service for chat operations with retrieval-augmented generation."""

    def __init__(
        self,
        config: WeaviateConfig | None = None,
        llm_provider: LLMInterface | None = None,
        llm_config: LLMConfig | None = None
    ):
        """
        Initialize the chat service.

        Args:
            config: Database configuration. If None, loads from environment.
            llm_provider: LLM provider instance. If None, creates default provider.
            llm_config: LLM configuration. If None, loads from environment.
        """
        if config is None:
            config = WeaviateConfig.from_environment()
        self.config = config
        self.repository_manager = BookRepositoryManager(config)

        # Initialize LLM provider
        if llm_provider is None:
            llm_config = llm_config or LLMConfig.from_environment()
            # Try to use LangChain provider, fall back to mock
            # TODO: do we want to set mock here? or just fail?
            try:
                from ..llm import LangChainProvider
                self.llm_provider = LangChainProvider(llm_config)
                logger.info("Using LangChain provider for LLM")
            except ImportError:
                self.llm_provider = MockLLMProvider(llm_config)
                logger.info("Using Mock provider for LLM (LangChain not available)")
        else:
            self.llm_provider = llm_provider

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

    async def list_recent_sessions(self, limit: int = 10) -> List[ChatSession]:
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

    async def get_session_messages(self, session_id: str) -> List[ChatMessage]:
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

    async def send_message(
        self,
        session_id: str,
        user_message: str,
        max_context_paragraphs: int = 5,
        enable_retrieval: bool = True
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
            # 1. Create and save user message
            user_msg = ChatMessage(
                content=user_message,
                role=MessageRole.USER,
                session_id=session_id
            )
            user_msg_id = self.repository_manager.chat_messages.create(user_msg)
            user_msg.id = user_msg_id
            logger.info(f"Saved user message {user_msg_id} to session {session_id}")

            # 2. Retrieve relevant context if enabled
            context_paragraphs = []
            if enable_retrieval and max_context_paragraphs > 0:
                context_paragraphs = await self._retrieve_context(
                    user_message, max_context_paragraphs
                )

            # 3. Get recent chat history
            chat_history = await self.get_session_messages(session_id)
            
            # Include the new user message in history for LLM
            if user_msg not in chat_history:
                chat_history.append(user_msg)

            # 4. Generate AI response
            context_text = self._format_context(context_paragraphs)
            ai_response = await self.llm_provider.generate_response(
                messages=chat_history,
                context=context_text
            )

            # 5. Create and save AI message
            ai_msg = ChatMessage(
                content=ai_response,
                role=MessageRole.ASSISTANT,
                session_id=session_id,
                retrieved_paragraphs=[p.id for p in context_paragraphs] if context_paragraphs else None
            )
            ai_msg_id = self.repository_manager.chat_messages.create(ai_msg)
            ai_msg.id = ai_msg_id

            # 6. Update session timestamp
            await self._update_session_timestamp(session_id)

            logger.info(f"Generated AI response {ai_msg_id} for session {session_id}")
            return ai_msg

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
        max_context_paragraphs: int = 5,
        enable_retrieval: bool = True
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
            # 1. Create and save user message
            user_msg = ChatMessage(
                content=user_message,
                role=MessageRole.USER,
                session_id=session_id
            )
            user_msg_id = self.repository_manager.chat_messages.create(user_msg)
            user_msg.id = user_msg_id

            # 2. Retrieve relevant context if enabled
            context_paragraphs = []
            if enable_retrieval and max_context_paragraphs > 0:
                context_paragraphs = await self._retrieve_context(
                    user_message, max_context_paragraphs
                )

            # 3. Get recent chat history
            chat_history = await self.get_session_messages(session_id)
            
            # Include the new user message in history
            if user_msg not in chat_history:
                chat_history.append(user_msg)

            # 4. Generate streaming AI response
            context_text = self._format_context(context_paragraphs)
            
            # Collect response chunks for saving later
            response_chunks = []
            async for chunk in self.llm_provider.generate_stream_response(
                messages=chat_history,
                context=context_text
            ):
                response_chunks.append(chunk)
                yield chunk

            # 5. Save complete AI response
            complete_response = "".join(response_chunks)
            ai_msg = ChatMessage(
                content=complete_response,
                role=MessageRole.ASSISTANT,
                session_id=session_id,
                retrieved_paragraphs=[p.id for p in context_paragraphs] if context_paragraphs else None
            )
            ai_msg_id = self.repository_manager.chat_messages.create(ai_msg)

            # 6. Update session timestamp
            await self._update_session_timestamp(session_id)

            logger.info(f"Completed streaming response {ai_msg_id} for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to stream message: {e}")
            raise

    async def _retrieve_context(
        self,
        query: str,
        max_paragraphs: int
    ) -> List[Paragraph]:
        """
        Retrieve relevant paragraphs for the query.

        Args:
            query: User query
            max_paragraphs: Maximum number of paragraphs to retrieve

        Returns:
            List of relevant paragraphs
        """
        try:
            # Use vector search to find relevant paragraphs
            # return self.repository_manager.paragraphs.vector_search(
            #     query_text=query,
            #     limit=max_paragraphs
            # )            
            search_result = self.repository_manager.paragraphs.similarity_search_by_text(
                query_text=query,
                limit=max_paragraphs
            )
            return [para[0] for para in search_result] if search_result else []
        #
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
            return []

    def _format_context(self, paragraphs: List[Paragraph]) -> str | None:
        """
        Format retrieved paragraphs as context for the LLM.

        Args:
            paragraphs: Retrieved paragraphs

        Returns:
            Formatted context string or None
        """
        if not paragraphs:
            return None

        context_parts = []
        for i, para in enumerate(paragraphs, 1):
            # Include page information for citation
            context_parts.append(f"[Source {i}, Page {para.page}]: {para.text}")

        return "\n\n".join(context_parts)

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
                updates = {"updated_at": datetime.now(timezone.utc)}
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
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10
    ) -> List[ChatMessage]:
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
                query_text=query,
                session_id=session_id,
                limit=limit
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
