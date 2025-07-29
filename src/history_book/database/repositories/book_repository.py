"""Book-specific repository implementation."""

import logging
from typing import List, Tuple, TYPE_CHECKING
from ..repositories.weaviate_repository import WeaviateRepository
from ..config.database_config import WeaviateConfig

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ...data_models.entities import Book, Chapter, Paragraph, ChatMessage, ChatSession

logger = logging.getLogger(__name__)


class BookRepository(WeaviateRepository["Book"]):
    """Repository for book entities."""

    def __init__(self, config: WeaviateConfig):
        from ...data_models.entities import Book

        super().__init__(
            config=config,
            collection_name="Books",  # Will be auto-capitalized
            entity_class=Book,
        )


class ChapterRepository(WeaviateRepository["Chapter"]):
    """Repository for chapter entities."""

    def __init__(self, config: WeaviateConfig):
        from ...data_models.entities import Chapter

        super().__init__(
            config=config,
            collection_name="Chapters",  # Will be auto-capitalized
            entity_class=Chapter,
        )

    def find_by_book_index(self, book_index: int) -> List["Chapter"]:
        """Find all chapters for a specific book."""
        return self.find_by_criteria({"book_index": book_index})


class ParagraphRepository(WeaviateRepository["Paragraph"]):
    """Repository for paragraph entities with vector search capabilities."""

    def __init__(self, config: WeaviateConfig):
        from ...data_models.entities import Paragraph

        super().__init__(
            config=config,
            collection_name="Paragraphs",  # Will be auto-capitalized
            entity_class=Paragraph,
        )

    def find_by_book_index(self, book_index: int) -> List["Paragraph"]:
        """Find all paragraphs for a specific book."""
        return self.find_by_criteria({"book_index": book_index})

    def find_by_chapter_index(
        self, book_index: int, chapter_index: int
    ) -> List["Paragraph"]:
        """Find all paragraphs for a specific chapter."""
        return self.find_by_criteria(
            {"book_index": book_index, "chapter_index": chapter_index}
        )

    def search_similar_paragraphs(
        self,
        query_text: str,
        limit: int = 10,
        book_index: int | None = None,
        threshold: float | None = None,
    ) -> List[Tuple["Paragraph", float]]:
        """
        Search for similar paragraphs using text similarity.

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            book_index: Optional book index to limit search scope
            threshold: Minimum similarity threshold

        Returns:
            List of (paragraph, similarity_score) tuples
        """
        # If book_index is specified, we need to filter results
        # For now, we'll do a simple similarity search and filter afterward
        # In a more advanced implementation, you could combine this with where filters

        results = self.similarity_search_by_text(
            query_text=query_text,
            limit=limit * 2
            if book_index is not None
            else limit,  # Get more results for filtering
            threshold=threshold,
        )

        if book_index is not None:
            # Filter results by book_index
            filtered_results = [
                (paragraph, score)
                for paragraph, score in results
                if paragraph.book_index == book_index
            ]
            return filtered_results[:limit]

        return results[:limit]

    def search_paragraphs_by_page_range(
        self, start_page: int, end_page: int, book_index: int | None = None
    ) -> List["Paragraph"]:
        """
        Find paragraphs within a specific page range.

        Args:
            start_page: Starting page number
            end_page: Ending page number
            book_index: Optional book index to limit search scope

        Returns:
            List of paragraphs in the page range
        """
        # This would require more complex filtering in Weaviate
        # For now, we'll implement a simple version
        all_paragraphs = self.list_all()

        filtered_paragraphs = [
            p
            for p in all_paragraphs
            if start_page <= p.page <= end_page
            and (book_index is None or p.book_index == book_index)
        ]

        return filtered_paragraphs


class ChatSessionRepository(WeaviateRepository["ChatSession"]):
    """Repository for chat session entities."""

    def __init__(self, config: WeaviateConfig):
        from ...data_models.entities import ChatSession

        super().__init__(
            config=config,
            collection_name="ChatSessions",
            entity_class=ChatSession,
        )

    def find_recent_sessions(self, limit: int = 10) -> List["ChatSession"]:
        """Find the most recently updated chat sessions."""
        # Note: This would need to be implemented in the base WeaviateRepository
        # For now, return all and sort in Python (not optimal for large datasets)
        all_sessions = self.list_all()
        return sorted(all_sessions, key=lambda s: s.updated_at, reverse=True)[:limit]


class ChatMessageRepository(WeaviateRepository["ChatMessage"]):
    """Repository for chat message entities with vector search capabilities."""

    def __init__(self, config: WeaviateConfig):
        from ...data_models.entities import ChatMessage

        super().__init__(
            config=config,
            collection_name="ChatMessages",
            entity_class=ChatMessage,
        )

    def find_by_session(self, session_id: str) -> List["ChatMessage"]:
        """Find all messages for a specific session, ordered by timestamp."""
        messages = self.find_by_criteria({"session_id": session_id})
        return sorted(messages, key=lambda m: m.timestamp)

    def find_by_role(self, session_id: str, role: str) -> List["ChatMessage"]:
        """Find all messages for a specific session and role."""
        messages = self.find_by_criteria({"session_id": session_id, "role": role})
        return sorted(messages, key=lambda m: m.timestamp)

    def search_message_content(
        self,
        query_text: str,
        session_id: str | None = None,
        limit: int = 10
    ) -> List["ChatMessage"]:
        """
        Search for messages by content similarity.
        
        Args:
            query_text: Text to search for
            session_id: Optional session to limit search to
            limit: Maximum number of results
            
        Returns:
            List of messages sorted by relevance
        """
        # Use the vector search from the base repository
        if session_id:
            # Search within a specific session
            where_filter = {"session_id": session_id}
            return self.vector_search(
                query_text=query_text,
                limit=limit,
                where_filter=where_filter
            )
        else:
            # Search across all messages
            return self.vector_search(query_text=query_text, limit=limit)


class BookRepositoryManager:
    """
    Manager class that provides access to all repositories.
    This simplifies dependency injection and provides a single entry point
    for all data operations.
    """

    def __init__(self, config: WeaviateConfig):
        self.config = config
        self._book_repo: BookRepository | None = None
        self._chapter_repo: ChapterRepository | None = None
        self._paragraph_repo: ParagraphRepository | None = None
        self._chat_session_repo: ChatSessionRepository | None = None
        self._chat_message_repo: ChatMessageRepository | None = None

    @property
    def books(self) -> BookRepository:
        """Get the book repository."""
        if self._book_repo is None:
            self._book_repo = BookRepository(self.config)
        return self._book_repo

    @property
    def chapters(self) -> ChapterRepository:
        """Get the chapter repository."""
        if self._chapter_repo is None:
            self._chapter_repo = ChapterRepository(self.config)
        return self._chapter_repo

    @property
    def paragraphs(self) -> ParagraphRepository:
        """Get the paragraph repository."""
        if self._paragraph_repo is None:
            self._paragraph_repo = ParagraphRepository(self.config)
        return self._paragraph_repo

    @property
    def chat_sessions(self) -> ChatSessionRepository:
        """Get the chat session repository."""
        if self._chat_session_repo is None:
            self._chat_session_repo = ChatSessionRepository(self.config)
        return self._chat_session_repo

    @property
    def chat_messages(self) -> ChatMessageRepository:
        """Get the chat message repository."""
        if self._chat_message_repo is None:
            self._chat_message_repo = ChatMessageRepository(self.config)
        return self._chat_message_repo

    def close_all(self):
        """Close all repository connections."""
        repositories = [
            self._book_repo, 
            self._chapter_repo, 
            self._paragraph_repo,
            self._chat_session_repo,
            self._chat_message_repo
        ]
        for repo in repositories:
            if repo is not None:
                try:
                    repo.close()
                except Exception as e:
                    logger.warning(f"Error closing repository: {e}")

        self._book_repo = None
        self._chapter_repo = None
        self._paragraph_repo = None
        self._chat_session_repo = None
        self._chat_message_repo = None
