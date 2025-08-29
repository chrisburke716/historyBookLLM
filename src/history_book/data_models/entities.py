"""Data models for the history book application."""

from typing import List, ClassVar
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
import uuid


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
    embedding: List[float] | None = None
    page: int
    paragraph_index: int
    book_index: int
    chapter_index: int

    # Specify which fields should be vectorized by Weaviate
    vectorize_fields: ClassVar[List[str]] = ["text"]


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    retrieved_paragraphs: List[str] | None = None  # IDs of paragraphs used as context

    # Specify which fields should be vectorized by Weaviate (for semantic search of chat history)
    vectorize_fields: ClassVar[List[str]] = ["content"]


class ChatSession(BaseModel):
    """Represents a chat conversation session containing multiple messages."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str | None = None  # Auto-generated or user-provided
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Note: messages are stored separately and linked by session_id for better performance
