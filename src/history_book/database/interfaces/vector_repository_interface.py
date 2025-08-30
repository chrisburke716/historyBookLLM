"""Vector repository interface for vector database operations."""

from abc import ABC, abstractmethod
from typing import Any

from .repository_interface import BaseRepository, T


class VectorRepository(BaseRepository[T], ABC):
    """
    Abstract interface for vector database operations.

    Extends the base repository with vector-specific operations like
    similarity search, embedding management, and vector indexing.
    """

    @abstractmethod
    def similarity_search(
        self,
        query_vector: list[float],
        limit: int = 10,
        threshold: float | None = None,
        where_filter: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[tuple[T, float]]:
        """
        Perform similarity search using a query vector.

        Args:
            query_vector: The vector to search with
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            where_filter: Optional filter criteria to limit search scope
            **kwargs: Additional search parameters

        Returns:
            List of tuples containing (entity, similarity_score)
        """
        pass

    @abstractmethod
    def similarity_search_by_text(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float | None = None,
        where_filter: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[tuple[T, float]]:
        """
        Perform similarity search using text query.

        Args:
            query_text: The text to search with
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            where_filter: Optional filter criteria to limit search scope
            **kwargs: Additional search parameters

        Returns:
            List of tuples containing (entity, similarity_score)
        """
        pass

    @abstractmethod
    def create_with_vector(
        self, entity: T, vector: list[float] | None = None, **kwargs
    ) -> str:
        """
        Create an entity with an associated vector.

        Args:
            entity: The entity to create
            vector: Optional pre-computed vector
            **kwargs: Additional parameters

        Returns:
            The ID of the created entity
        """
        pass

    @abstractmethod
    def update_vector(self, entity_id: str, vector: list[float], **kwargs) -> bool:
        """
        Update the vector for an existing entity.

        Args:
            entity_id: The unique identifier of the entity
            vector: The new vector
            **kwargs: Additional parameters

        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_vector(self, entity_id: str, **kwargs) -> list[float] | None:
        """
        Retrieve the vector for an entity.

        Args:
            entity_id: The unique identifier of the entity
            **kwargs: Additional parameters

        Returns:
            The vector if found, None otherwise
        """
        pass

    @abstractmethod
    def batch_create_with_vectors(
        self, entities_and_vectors: list[tuple[T, list[float] | None]], **kwargs
    ) -> list[str]:
        """
        Create multiple entities with their vectors in a batch operation.

        Args:
            entities_and_vectors: List of (entity, vector) tuples
            **kwargs: Additional parameters

        Returns:
            List of created entity IDs
        """
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        alpha: float = 0.5,
        limit: int = 10,
        **kwargs,
    ) -> list[tuple[T, float]]:
        """
        Perform hybrid search combining text and vector search.

        Args:
            query_text: Text query for keyword search
            query_vector: Optional vector for similarity search
            alpha: Weight for combining text and vector scores (0-1)
            limit: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of tuples containing (entity, combined_score)
        """
        pass
