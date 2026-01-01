"""Book API routes for browsing and reading book content."""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from history_book.api.models.api_models import (
    BookListResponse,
    BookResponse,
    ChapterContentResponse,
    ChapterListResponse,
    ChapterResponse,
    ParagraphResponse,
)
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/books", tags=["books"])


# Dependency to get BookRepositoryManager instance
# Cached to reuse the same instance and avoid creating new connections per request
@lru_cache
def get_repository_manager():
    """Get a BookRepositoryManager instance (cached)."""
    config = WeaviateConfig.from_environment()
    return BookRepositoryManager(config)


@router.get("", response_model=BookListResponse)
async def get_books(
    repo_manager: BookRepositoryManager = Depends(get_repository_manager),
):
    """Get list of all books."""
    try:
        books = repo_manager.books.list_all()
        # Sort by book_index for consistent ordering
        books_sorted = sorted(books, key=lambda b: b.book_index)
        book_responses = [
            BookResponse(
                id=book.id,
                title=book.title,
                book_index=book.book_index,
                start_page=book.start_page,
                end_page=book.end_page,
            )
            for book in books_sorted
        ]
        return BookListResponse(books=book_responses)
    except Exception as e:
        logger.error(f"Failed to get books: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve books") from e


@router.get("/{book_index}/chapters", response_model=ChapterListResponse)
async def get_chapters(
    book_index: int,
    repo_manager: BookRepositoryManager = Depends(get_repository_manager),
):
    """Get list of chapters for a specific book."""
    try:
        chapters = repo_manager.chapters.find_by_book_index(book_index)

        if not chapters:
            raise HTTPException(
                status_code=404, detail=f"No chapters found for book {book_index}"
            )

        # Sort by chapter_index for correct ordering
        chapters_sorted = sorted(chapters, key=lambda c: c.chapter_index)
        chapter_responses = [
            ChapterResponse(
                id=chapter.id,
                title=chapter.title,
                chapter_index=chapter.chapter_index,
                book_index=chapter.book_index,
                start_page=chapter.start_page,
                end_page=chapter.end_page,
            )
            for chapter in chapters_sorted
        ]
        return ChapterListResponse(chapters=chapter_responses)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chapters for book {book_index}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chapters"
        ) from e


@router.get(
    "/{book_index}/chapters/{chapter_index}", response_model=ChapterContentResponse
)
async def get_chapter_content(
    book_index: int,
    chapter_index: int,
    repo_manager: BookRepositoryManager = Depends(get_repository_manager),
):
    """Get full content of a specific chapter including all paragraphs."""
    try:
        # Get chapter metadata
        chapters = repo_manager.chapters.find_by_criteria(
            {
                "book_index": book_index,
                "chapter_index": chapter_index,
            }
        )

        if not chapters:
            raise HTTPException(
                status_code=404,
                detail=f"Chapter {chapter_index} not found in book {book_index}",
            )

        chapter = chapters[0]

        # Get paragraphs for this chapter
        paragraphs = repo_manager.paragraphs.find_by_chapter_index(
            book_index=book_index, chapter_index=chapter_index
        )

        if not paragraphs:
            logger.warning(
                f"No paragraphs found for book {book_index}, chapter {chapter_index}"
            )

        # Sort paragraphs by paragraph_index for correct reading order
        paragraphs_sorted = sorted(paragraphs, key=lambda p: p.paragraph_index)

        # Convert to response models
        chapter_response = ChapterResponse(
            id=chapter.id,
            title=chapter.title,
            chapter_index=chapter.chapter_index,
            book_index=chapter.book_index,
            start_page=chapter.start_page,
            end_page=chapter.end_page,
        )

        paragraph_responses = [
            ParagraphResponse(
                text=para.text, page=para.page, paragraph_index=para.paragraph_index
            )
            for para in paragraphs_sorted
        ]

        return ChapterContentResponse(
            chapter=chapter_response, paragraphs=paragraph_responses
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get content for book {book_index}, chapter {chapter_index}: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chapter content"
        ) from e
