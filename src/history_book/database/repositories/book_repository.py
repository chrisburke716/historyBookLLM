"""Book-specific repository implementation."""

import logging
from typing import List, Optional, Tuple
from ..repositories.weaviate_repository import WeaviateRepository
from ..config.database_config import WeaviateConfig
from ...data_models.book import BookDBModel, ChapterDBModel, ParagraphDBModel

logger = logging.getLogger(__name__)


class BookRepository(WeaviateRepository[BookDBModel]):
    """Repository for book entities."""
    
    def __init__(self, config: WeaviateConfig):
        super().__init__(
            config=config,
            collection_name="books",
            entity_class=BookDBModel
        )


class ChapterRepository(WeaviateRepository[ChapterDBModel]):
    """Repository for chapter entities."""
    
    def __init__(self, config: WeaviateConfig):
        super().__init__(
            config=config,
            collection_name="chapters",
            entity_class=ChapterDBModel
        )
    
    def find_by_book_index_sync(self, book_index: int) -> List[ChapterDBModel]:
        """Find all chapters for a specific book."""
        return self.find_by_criteria_sync({"book_index": book_index})


class ParagraphRepository(WeaviateRepository[ParagraphDBModel]):
    """Repository for paragraph entities with vector search capabilities."""
    
    def __init__(self, config: WeaviateConfig):
        super().__init__(
            config=config,
            collection_name="paragraphs",
            entity_class=ParagraphDBModel
        )
    
    def find_by_book_index_sync(self, book_index: int) -> List[ParagraphDBModel]:
        """Find all paragraphs for a specific book."""
        return self.find_by_criteria_sync({"book_index": book_index})
    
    def find_by_chapter_index_sync(self, book_index: int, chapter_index: int) -> List[ParagraphDBModel]:
        """Find all paragraphs for a specific chapter."""
        return self.find_by_criteria_sync({
            "book_index": book_index,
            "chapter_index": chapter_index
        })
    
    def search_similar_paragraphs_sync(
        self, 
        query_text: str, 
        limit: int = 10,
        book_index: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[ParagraphDBModel, float]]:
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
        
        results = self.similarity_search_by_text_sync(
            query_text=query_text,
            limit=limit * 2 if book_index is not None else limit,  # Get more results for filtering
            threshold=threshold
        )
        
        if book_index is not None:
            # Filter results by book_index
            filtered_results = [
                (paragraph, score) for paragraph, score in results 
                if paragraph.book_index == book_index
            ]
            return filtered_results[:limit]
        
        return results[:limit]
    
    def search_paragraphs_by_page_range_sync(
        self, 
        start_page: int, 
        end_page: int,
        book_index: Optional[int] = None
    ) -> List[ParagraphDBModel]:
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
        all_paragraphs = self.list_all_sync()
        
        filtered_paragraphs = [
            p for p in all_paragraphs
            if start_page <= p.page <= end_page
            and (book_index is None or p.book_index == book_index)
        ]
        
        return filtered_paragraphs


class BookRepositoryManager:
    """
    Manager class that provides access to all book-related repositories.
    This simplifies dependency injection and provides a single entry point
    for all book data operations.
    """
    
    def __init__(self, config: WeaviateConfig):
        self.config = config
        self._book_repo: Optional[BookRepository] = None
        self._chapter_repo: Optional[ChapterRepository] = None
        self._paragraph_repo: Optional[ParagraphRepository] = None
    
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
    
    def close_all(self):
        """Close all repository connections."""
        repositories = [self._book_repo, self._chapter_repo, self._paragraph_repo]
        for repo in repositories:
            if repo is not None:
                try:
                    repo.close()
                except Exception as e:
                    logger.warning(f"Error closing repository: {e}")
        
        self._book_repo = None
        self._chapter_repo = None
        self._paragraph_repo = None
