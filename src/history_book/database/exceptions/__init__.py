"""Database exceptions module."""

from .database_exceptions import (
    DatabaseError,
    ConnectionError,
    CollectionError,
    EntityNotFoundError,
    ValidationError,
    VectorError,
    ConfigurationError,
    QueryError,
    BatchOperationError,
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
