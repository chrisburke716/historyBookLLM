"""Database exceptions module."""

from .database_exceptions import (
    BatchOperationError,
    CollectionError,
    ConfigurationError,
    ConnectionError,
    DatabaseError,
    EntityNotFoundError,
    QueryError,
    ValidationError,
    VectorError,
)

__all__ = [
    "DatabaseError",
    "ConnectionError",
    "CollectionError",
    "EntityNotFoundError",
    "ValidationError",
    "VectorError",
    "ConfigurationError",
    "QueryError",
    "BatchOperationError",
]
