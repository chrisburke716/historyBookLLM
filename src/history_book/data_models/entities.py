"""Data models for the history book application."""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field


class Book(BaseModel):
    """Represents a single book within the complete history book volume."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    start_page: int
    end_page: int
    book_index: int


class Chapter(BaseModel):
    """Represents a section or chapter within a book."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    start_page: int
    end_page: int
    book_index: int
    chapter_index: int


class Paragraph(BaseModel):
    """Represents a text chunk with optional vector embeddings for semantic search."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: list[float] | None = None
    page: int
    paragraph_index: int
    book_index: int
    chapter_index: int

    # Specify which fields should be vectorized by Weaviate
    vectorize_fields: ClassVar[list[str]] = ["text"]


class MessageRole(str, Enum):
    """Enumeration for message roles in chat."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    retrieved_paragraphs: list[str] | None = None  # IDs of paragraphs used as context

    # Specify which fields should be vectorized by Weaviate (for semantic search of chat history)
    vectorize_fields: ClassVar[list[str]] = ["content"]


class ChatSession(BaseModel):
    """Represents a chat conversation session containing multiple messages."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str | None = None  # Auto-generated or user-provided
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Note: messages are stored separately and linked by session_id for better performance
