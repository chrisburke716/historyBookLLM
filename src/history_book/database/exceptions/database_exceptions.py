"""Custom exceptions for database operations."""


class DatabaseError(Exception):
    """Base exception for all database-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class CollectionError(DatabaseError):
    """Raised when collection operations fail."""

    pass


class EntityNotFoundError(DatabaseError):
    """Raised when a requested entity is not found."""

    def __init__(self, entity_id: str, entity_type: str | None = None):
        self.entity_id = entity_id
        self.entity_type = entity_type
        message = f"Entity with ID '{entity_id}' not found"
        if entity_type:
            message = f"{entity_type} with ID '{entity_id}' not found"
        super().__init__(message)


class ValidationError(DatabaseError):
    """Raised when entity validation fails."""

    pass


class VectorError(DatabaseError):
    """Raised when vector operations fail."""

    pass


class ConfigurationError(DatabaseError):
    """Raised when database configuration is invalid."""

    pass


class QueryError(DatabaseError):
    """Raised when database queries fail."""

    pass


class BatchOperationError(DatabaseError):
    """Raised when batch operations fail."""

    def __init__(
        self,
        message: str,
        failed_items: list | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, original_error)
        self.failed_items = failed_items or []
