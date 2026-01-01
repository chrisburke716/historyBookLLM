"""Pydantic models for API requests and responses."""

from datetime import datetime

from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    """Request to create a new chat session."""

    title: str | None = None


class MessageRequest(BaseModel):
    """Request to send a message in a chat session."""

    content: str
    enable_retrieval: bool = True
    max_context_paragraphs: int = 5


class SessionResponse(BaseModel):
    """Response containing chat session information."""

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """Response containing a chat message."""

    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    session_id: str
    citations: list[str] | None = None  # Simple page references like "Page 123"


class SessionListResponse(BaseModel):
    """Response containing a list of recent sessions."""

    sessions: list[SessionResponse]


class MessageListResponse(BaseModel):
    """Response containing a list of messages for a session."""

    messages: list[MessageResponse]


class ChatResponse(BaseModel):
    """Response after sending a message - includes the AI's reply."""

    message: MessageResponse


# Book reading API models


class BookResponse(BaseModel):
    """Response containing book information."""

    id: str
    title: str
    book_index: int
    start_page: int
    end_page: int


class ChapterResponse(BaseModel):
    """Response containing chapter information."""

    id: str
    title: str
    chapter_index: int
    book_index: int
    start_page: int
    end_page: int


class ParagraphResponse(BaseModel):
    """Response containing paragraph text and metadata."""

    text: str
    page: int
    paragraph_index: int


class BookListResponse(BaseModel):
    """Response containing a list of books."""

    books: list[BookResponse]


class ChapterListResponse(BaseModel):
    """Response containing a list of chapters."""

    chapters: list[ChapterResponse]


class ChapterContentResponse(BaseModel):
    """Response containing full chapter content with all paragraphs."""

    chapter: ChapterResponse
    paragraphs: list[ParagraphResponse]
