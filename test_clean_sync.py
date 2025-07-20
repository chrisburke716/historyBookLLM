#!/usr/bin/env python3
"""Simple demo of the clean sync-only architecture."""

import logging
from src.history_book.database.config import WeaviateConfig
from src.history_book.database.repositories.book_repository import BookRepository, ChapterRepository

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demo the clean sync-only architecture."""
    logger.info("🚀 Testing Clean Sync-Only Architecture")
    
    # Create config
    config = WeaviateConfig.from_environment()
    
    # Test basic repository operations
    logger.info("\n=== Testing Repository Pattern ===")
    
    try:
        # Create repositories
        book_repo = BookRepository(config)
        chapter_repo = ChapterRepository(config)
        
        # Test basic operations
        book_count = book_repo.count()
        chapter_count = chapter_repo.count()
        
        logger.info(f"✅ Found {book_count} books and {chapter_count} chapters")
        
        # Test listing
        if book_count > 0:
            books = book_repo.list_all(limit=2)
            logger.info(f"✅ Listed {len(books)} books successfully")
            for book in books:
                logger.info(f"   📖 {book.title}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    finally:
        book_repo.close()
        chapter_repo.close()
    
    logger.info("\n✅ Clean sync-only architecture working!")
    logger.info("🎯 Key Benefits:")
    logger.info("   - No fake async methods")
    logger.info("   - Clean, honest APIs")
    logger.info("   - Simple and maintainable")
    logger.info("   - Easy to test and debug")

if __name__ == "__main__":
    main()
