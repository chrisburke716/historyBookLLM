"""Base repository interface for database operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base class for repository pattern implementation.

    This interface defines the basic CRUD operations that all repositories
    should implement, providing a consistent API regardless of the underlying
    database technology.
    """

    @abstractmethod
    def create(self, entity: T, **kwargs) -> str:
        """
        Create a new entity in the database.

        Args:
            entity: The entity to create
            **kwargs: Additional parameters for creation

        Returns:
            The ID of the created entity
        """
        pass

    @abstractmethod
    def get_by_id(self, entity_id: str, **kwargs) -> Optional[T]:
        """
        Retrieve an entity by its ID.

        Args:
            entity_id: The unique identifier of the entity
            **kwargs: Additional parameters for retrieval

        Returns:
            The entity if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, entity_id: str, updates: Dict[str, Any], **kwargs) -> bool:
        """
        Update an existing entity.

        Args:
            entity_id: The unique identifier of the entity
            updates: Dictionary of field updates
            **kwargs: Additional parameters for update

        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, entity_id: str, **kwargs) -> bool:
        """
        Delete an entity by its ID.

        Args:
            entity_id: The unique identifier of the entity
            **kwargs: Additional parameters for deletion

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    def list_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs
    ) -> List[T]:
        """
        List all entities with optional pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            **kwargs: Additional parameters for listing

        Returns:
            List of entities
        """
        pass

    @abstractmethod
    def count(self, **kwargs) -> int:
        """
        Count the total number of entities.

        Args:
            **kwargs: Additional parameters for counting

        Returns:
            Total count of entities
        """
        pass

    @abstractmethod
    def exists(self, entity_id: str, **kwargs) -> bool:
        """
        Check if an entity exists by its ID.

        Args:
            entity_id: The unique identifier of the entity
            **kwargs: Additional parameters for existence check

        Returns:
            True if entity exists, False otherwise
        """
        pass

    @abstractmethod
    def find_by_criteria(self, criteria: Dict[str, Any], **kwargs) -> List[T]:
        """
        Find entities matching specific criteria.

        Args:
            criteria: Dictionary of search criteria
            **kwargs: Additional parameters for search

        Returns:
            List of matching entities
        """
        pass
