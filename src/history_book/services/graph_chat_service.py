"""Chat service for LangGraph-based agentic interactions."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from langsmith import traceable

from history_book.data_models.entities import (
    ChatMessage,
    ChatSession,
    MessageRole,
    Paragraph,
)
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager
from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError
from history_book.services.graph_rag_service import GraphRagService

logger = logging.getLogger(__name__)

CONTEXT_MIN_RESULTS = 5
CONTEXT_MAX_RESULTS = 40
CONTEXT_SIMILARITY_CUTOFF = 0.4


@dataclass
class GraphChatResult:
    """Result from graph chat service containing message and retrieved context."""

    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]


class GraphChatService:
    """
    Chat session orchestration for LangGraph-based RAG.

    Hybrid memory strategy:
    - LangGraph MemorySaver: In-graph state during execution
    - Weaviate: Long-term persistence of sessions and messages
    """

    def __init__(
        self,
        config: WeaviateConfig | None = None,
        llm_config: LLMConfig | None = None,
        min_context_results: int = CONTEXT_MIN_RESULTS,
        max_context_results: int = CONTEXT_MAX_RESULTS,
        context_similarity_cutoff: float = CONTEXT_SIMILARITY_CUTOFF,
    ):
        """
        Initialize the graph chat service.

        Args:
            config: Database configuration. If None, loads from environment.
            llm_config: LLM configuration. If None, loads from environment.
            min_context_results: Minimum number of context documents to retrieve.
            max_context_results: Maximum number of context documents to retrieve.
            context_similarity_cutoff: Similarity threshold for context retrieval.
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

        # Initialize GraphRAG service
        self.graph_rag = GraphRagService(
            llm_config=self.llm_config,
            repository_manager=self.repository_manager,
            min_context_results=min_context_results,
            max_context_results=max_context_results,
            context_similarity_cutoff=context_similarity_cutoff,
        )

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

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages.

        Args:
            session_id: Session ID

        Returns:
            True if deleted successfully
        """
        try:
            # Delete all messages first
            messages = await self.get_session_messages(session_id)
            for msg in messages:
                if msg.id:
                    self.repository_manager.chat_messages.delete(msg.id)

            # Delete session
            self.repository_manager.chat_sessions.delete(session_id)
            logger.info(f"Deleted session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    @traceable(name="Graph Chat Service: Send Message")
    async def send_message(
        self,
        session_id: str,
        user_message: str,
    ) -> GraphChatResult:
        """
        Send a message and get response using LangGraph.

        Flow:
        1. Save user message to Weaviate
        2. Load chat history from Weaviate
        3. Execute graph with history + new message
        4. Save AI response to Weaviate
        5. Return AI message + source paragraphs

        Args:
            session_id: Session ID
            user_message: User's message content

        Returns:
            GraphChatResult containing AI response message and retrieved paragraphs

        Raises:
            LLMError: If LLM generation fails
            DatabaseError: If database operations fail
        """
        try:
            # Prepare user message and chat history
            user_msg, chat_history = await self._prepare_message_and_history(
                session_id, user_message
            )

            # Execute graph
            result_state = await self.graph_rag.invoke(
                question=user_message,
                messages=chat_history,
                session_id=session_id,
            )

            # Save AI response
            ai_message = await self._save_ai_message_and_update_session(
                session_id,
                result_state["generation"],
                result_state["retrieved_paragraphs"],
            )

            # Return GraphChatResult
            return GraphChatResult(
                message=ai_message,
                retrieved_paragraphs=result_state["retrieved_paragraphs"],
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
    ) -> tuple[AsyncIterator[str], list[Paragraph]]:
        """
        Send a message and get streaming response using LangGraph.

        Yields token chunks, then saves complete message at end.

        Args:
            session_id: Session ID
            user_message: User's message content

        Returns:
            Tuple of (response_stream, retrieved_paragraphs)

        Note:
            The final complete response is saved to the database after streaming completes.
        """
        try:
            # Prepare user message and chat history
            user_msg, chat_history = await self._prepare_message_and_history(
                session_id, user_message
            )

            # Track state for saving later
            full_response = ""
            retrieved_paragraphs = []

            # Create wrapper that accumulates response and saves
            async def stream_and_save():
                nonlocal full_response, retrieved_paragraphs

                # Stream graph execution
                async for chunk in self.graph_rag.stream(
                    question=user_message,
                    messages=chat_history,
                    session_id=session_id,
                ):
                    # Extract token chunks from messages
                    if hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        full_response += token
                        yield token

                # After streaming completes, get the final state to extract paragraphs
                # We'll execute the graph again non-streaming to get final state
                # (This is a limitation - ideally we'd capture state during streaming)
                final_state = await self.graph_rag.invoke(
                    question=user_message,
                    messages=chat_history,
                    session_id=session_id,
                )
                retrieved_paragraphs = final_state["retrieved_paragraphs"]

                # Save complete AI response
                await self._save_ai_message_and_update_session(
                    session_id, full_response, retrieved_paragraphs
                )

            # Return the stream generator and empty list (paragraphs filled after streaming)
            return stream_and_save(), retrieved_paragraphs

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
        # Create and save user message
        user_msg = ChatMessage(
            content=user_message, role=MessageRole.USER, session_id=session_id
        )
        user_msg_id = self.repository_manager.chat_messages.create(user_msg)
        user_msg.id = user_msg_id
        logger.info(f"Saved user message {user_msg_id} to session {session_id}")

        # Get recent chat history (excluding the just-saved user message)
        chat_history = await self.get_session_messages(session_id)

        # Filter out the new message to avoid duplication
        # (it will be added by the graph execution)
        chat_history = [msg for msg in chat_history if msg.id != user_msg_id]

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
                updates = {"updated_at": datetime.now(UTC)}
                self.repository_manager.chat_sessions.update(session_id, updates)
        except Exception as e:
            logger.warning(f"Failed to update session timestamp: {e}")
