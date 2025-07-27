"""Pure data models without database operations."""

from typing import List, Optional, ClassVar
from pydantic import BaseModel, Field
import uuid


class Book(BaseModel):
    """Pure data model for a book (no database operations)."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    start_page: int
    end_page: int
    book_index: int


class Chapter(BaseModel):
    """Pure data model for a chapter (no database operations)."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    start_page: int
    end_page: int
    book_index: int
    chapter_index: int


class Paragraph(BaseModel):
    """Pure data model for a paragraph (no database operations)."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: Optional[List[float]] = None
    page: int
    paragraph_index: int
    book_index: int
    chapter_index: int

    # Specify which fields should be vectorized by Weaviate
    vectorize_fields: ClassVar[List[str]] = ["text"]
