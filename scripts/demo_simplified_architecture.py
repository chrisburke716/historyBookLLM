"""
Demo script showing the new simplified architecture:
- Pure data models (entities)
- Direct repository usage for Book/Chapter (simple CRUD)
- Service layer for Paragraph only (complex vector operations)
"""

import logging
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepository, ChapterRepository
from history_book.services import ParagraphService
from history_book.data_models.entities import Book, Chapter, Paragraph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_pure_entities_and_repositories():
    """Demonstrate using pure data models with repositories directly."""
    logger.info("=== Pure Entities + Direct Repository Usage ===")
    
    config = WeaviateConfig.from_environment()
    
    # Create repositories directly for simple entities
    book_repo = BookRepository(config)
    chapter_repo = ChapterRepository(config)
    
    try:
        # Example 1: Count existing data using repositories
        book_count = book_repo.count_sync()
        chapter_count = chapter_repo.count_sync()
        
        logger.info(f"Found {book_count} books and {chapter_count} chapters")
        
        # Example 2: List books using repository
        if book_count > 0:
            logger.info("Listing first 3 books...")
            books = book_repo.list_all_sync(limit=3)
            for book in books:
                logger.info(f"  Book {book.book_index}: {book.title} (pages {book.start_page}-{book.end_page})")
        
        # Example 3: Find chapters by book using repository  
        if chapter_count > 0:
            logger.info("Finding chapters for book 0...")
            chapters = chapter_repo.find_by_book_index_sync(book_index=0)
            for chapter in chapters[:3]:  # Show first 3
                logger.info(f"  Chapter {chapter.chapter_index}: {chapter.title}")
                
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        book_repo.close()
        chapter_repo.close()


def demo_paragraph_service():
    """Demonstrate using the service layer for complex paragraph operations."""
    logger.info("=== Paragraph Service Layer (Complex Operations) ===")
    
    # Use service layer for complex paragraph operations
    paragraph_service = ParagraphService()
    
    try:
        # Example 1: Count paragraphs
        paragraph_count = paragraph_service.count_paragraphs()
        logger.info(f"Found {paragraph_count} paragraphs")
        
        # Example 2: Complex vector search
        if paragraph_count > 0:
            logger.info("Searching for paragraphs about 'ancient Rome'...")
            results = paragraph_service.search_similar_paragraphs(
                query_text="ancient Rome",
                limit=3
            )
            
            for i, (paragraph, score) in enumerate(results):
                logger.info(f"  Result {i+1} (score: {score:.3f}): {paragraph.text[:80]}...")
        
        # Example 3: Get paragraphs by book (business logic)
        if paragraph_count > 0:
            logger.info("Getting paragraphs for book 0...")
            book_paragraphs = paragraph_service.get_paragraphs_by_book(book_index=0)
            logger.info(f"  Found {len(book_paragraphs)} paragraphs in book 0")
        
        # Example 4: Complex page range search
        if paragraph_count > 0:
            logger.info("Finding paragraphs in page range 100-110...")
            page_paragraphs = paragraph_service.get_paragraphs_by_page_range(
                start_page=100,
                end_page=110,
                book_index=0  # Optional filter
            )
            logger.info(f"  Found {len(page_paragraphs)} paragraphs in that range")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        paragraph_service.close()


def demo_creating_new_entities():
    """Demonstrate creating new entities with the new approach."""
    logger.info("=== Creating New Entities ===")
    
    config = WeaviateConfig.from_environment()
    
    # Create pure data models (no database operations)
    book = Book(
        title="Test Book - New Architecture",
        start_page=1,
        end_page=100,
        book_index=9999
    )
    
    chapter = Chapter(
        title="Test Chapter - New Architecture",
        start_page=1,
        end_page=10,
        book_index=9999,
        chapter_index=1
    )
    
    paragraph = Paragraph(
        text="This is a test paragraph created with the new architecture. It demonstrates how clean data models work without database coupling.",
        page=5,
        paragraph_index=1,
        book_index=9999,
        chapter_index=1
    )
    
    logger.info("Created pure data entities (no database operations yet)")
    logger.info(f"  Book: {book.title}")
    logger.info(f"  Chapter: {chapter.title}")
    logger.info(f"  Paragraph: {paragraph.text[:60]}...")
    
    # For simple entities, use repositories directly
    # book_repo = BookRepository(config)
    # book_id = book_repo.create_sync(book)
    
    # For complex entities, use service layer
    # paragraph_service = ParagraphService(config)
    # paragraph_id = paragraph_service.create_paragraph(paragraph)  # Handles embedding extraction
    
    logger.info("(Creation commented out to avoid modifying your database)")


def demo_architecture_comparison():
    """Show the difference between old and new approaches."""
    logger.info("=== Architecture Comparison ===")
    
    logger.info("OLD APPROACH:")
    logger.info("  BookDBModel() -> Auto-saves to database on __init__")
    logger.info("  Tight coupling between data and database")
    logger.info("  Hard to test, hard to mock")
    
    logger.info("\nNEW APPROACH:")
    logger.info("  Book() -> Pure data model, no database operations")
    logger.info("  BookRepository -> Simple CRUD operations")
    logger.info("  ParagraphService -> Complex business logic only")
    logger.info("  Clean separation, easy to test")


if __name__ == "__main__":
    logger.info("Starting simplified architecture demonstration...")
    
    demo_architecture_comparison()
    print()
    
    demo_pure_entities_and_repositories()
    print()
    
    demo_paragraph_service()
    print()
    
    demo_creating_new_entities()
    
    logger.info("\nDemonstration complete!")
    logger.info("Architecture benefits:")
    logger.info("  ✅ Pure data models (testable, mockable)")
    logger.info("  ✅ Simple CRUD via repositories")
    logger.info("  ✅ Complex logic only where needed (ParagraphService)")
    logger.info("  ✅ No over-engineering of simple operations")
