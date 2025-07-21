"""Service layer for paragraph operations with complex vector functionality."""

from typing import List, Optional, Tuple
import logging
from ..data_models.entities import Paragraph
from ..database.repositories import ParagraphRepository
from ..database.config import WeaviateConfig

logger = logging.getLogger(__name__)


class ParagraphService:
    """Service for paragraph operations with vector capabilities."""
    
    def __init__(self, config: Optional[WeaviateConfig] = None):
        """
        Initialize the paragraph service.
        
        Args:
            config: Optional WeaviateConfig. If not provided, will use environment-based config.
        """
        if config is None:
            config = WeaviateConfig.from_environment()
        self.config = config
        self._repository: Optional[ParagraphRepository] = None
    
    @property
    def repository(self) -> ParagraphRepository:
        """Get or create the paragraph repository."""
        if self._repository is None:
            self._repository = ParagraphRepository(self.config)
        return self._repository
    
    def create_paragraph(self, paragraph: Paragraph) -> str:
        """
        Create a new paragraph and handle vector embedding extraction.
        
        This method performs the complex multi-step process of:
        1. Creating the paragraph in the database
        2. Waiting for Weaviate to generate the vector embedding
        3. Fetching the generated embedding and updating the paragraph object
        
        Args:
            paragraph: The paragraph to create
            
        Returns:
            The ID of the created paragraph
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            # Step 1: Create the paragraph using repository
            paragraph_id = self.repository.create(paragraph)
            
            # Step 2: Fetch the created paragraph with its generated vector embedding
            # We need to get the vector from Weaviate's collection
            created_paragraph = self.repository.get_by_id(paragraph_id)
            
            # Step 3: Get the vector embedding from Weaviate
            vector = self.repository.get_vector(paragraph_id)
            
            # Step 4: Update the original paragraph object with the embedding and ID
            if created_paragraph:
                paragraph.embedding = vector
                paragraph.id = paragraph_id
                logger.debug(f"Created paragraph {paragraph_id} with embedding of {len(vector) if vector else 0} dimensions")
            else:
                logger.warning(f"Created paragraph {paragraph_id} but couldn't fetch it back")
            
            return paragraph_id
            
        except Exception as e:
            logger.error(f"Failed to create paragraph: {e}")
            raise

    def get_paragraph_by_id(self, paragraph_id: str) -> Optional[Paragraph]:
        """Get a paragraph by ID."""
        return self.repository.get_by_id(paragraph_id)
    
    def search_similar_paragraphs(
        self, 
        query_text: str, 
        limit: int = 10,
        book_index: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[Paragraph, float]]:
        """
        Search for paragraphs similar to the given text query.
        
        This method handles the complex vector search operations and can
        apply additional business logic like filtering by book.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            book_index: Optional book index to limit search scope
            threshold: Minimum similarity threshold
            
        Returns:
            List of (paragraph, similarity_score) tuples, ordered by relevance
        """
        try:
            # Use the repository's specialized search method
            results = self.repository.search_similar_paragraphs(
                query_text=query_text,
                limit=limit,
                book_index=book_index,
                threshold=threshold
            )
            
            logger.debug(f"Found {len(results)} similar paragraphs for query: '{query_text[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar paragraphs: {e}")
            return []
    
    def get_paragraphs_by_book(self, book_index: int) -> List[Paragraph]:
        """Get all paragraphs for a specific book."""
        return self.repository.find_by_book_index(book_index)
    
    def get_paragraphs_by_chapter(
        self, 
        book_index: int, 
        chapter_index: int
    ) -> List[Paragraph]:
        """Get all paragraphs for a specific chapter."""
        return self.repository.find_by_chapter_index(book_index, chapter_index)
    
    def get_paragraphs_by_page_range(
        self,
        start_page: int,
        end_page: int,
        book_index: Optional[int] = None
    ) -> List[Paragraph]:
        """
        Get paragraphs within a specific page range.
        
        This demonstrates business logic that might be too complex for
        a simple repository method.
        """
        return self.repository.search_paragraphs_by_page_range(
            start_page=start_page,
            end_page=end_page,
            book_index=book_index
        )
    
    def batch_create_paragraphs(self, paragraphs: List[Paragraph]) -> List[str]:
        """
        Create multiple paragraphs efficiently.
        
        This method demonstrates complex batch operations that benefit
        from a service layer to coordinate multiple repository calls.
        
        Args:
            paragraphs: List of paragraphs to create
            
        Returns:
            List of created paragraph IDs
        """
        try:
            # Prepare batch data for repository
            entities_and_vectors = [(p, None) for p in paragraphs]  # Let Weaviate generate vectors
            
            # Perform batch creation
            paragraph_ids = self.repository.batch_create_with_vectors(entities_and_vectors)
            
            # Update the paragraph objects with their new IDs and fetch embeddings
            for paragraph, paragraph_id in zip(paragraphs, paragraph_ids):
                paragraph.id = paragraph_id
                # Fetch the vector for this paragraph
                try:
                    vector = self.repository.get_vector(paragraph_id)
                    paragraph.embedding = vector
                    logger.debug(f"Updated paragraph {paragraph_id} with embedding of {len(vector) if vector else 0} dimensions")
                except Exception as e:
                    logger.warning(f"Could not fetch embedding for paragraph {paragraph_id}: {e}")
            
            logger.info(f"Batch created {len(paragraph_ids)} paragraphs")
            return paragraph_ids
            
        except Exception as e:
            logger.error(f"Failed to batch create paragraphs: {e}")
            raise
    
    def count_paragraphs(self) -> int:
        """Count total paragraphs."""
        return self.repository.count()
    
    def close(self):
        """Close repository connections."""
        if self._repository is not None:
            self._repository.close()
            self._repository = None
