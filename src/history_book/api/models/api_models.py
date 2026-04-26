"""Pydantic models for the Book API."""

from pydantic import BaseModel

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
