"""
Example script demonstrating the new repository interface.

This script shows how to use the new repository pattern alongside
the existing code, providing a migration path.
"""

import logging
from history_book.database.config import WeaviateConfig, DatabaseEnvironment
from history_book.database.repositories import BookRepositoryManager
from history_book.data_models.book import BookDBModel, ChapterDBModel, ParagraphDBModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_new_interface():
    """Demonstrate using the new repository interface."""
    
    # Create configuration (this will read from environment variables)
    config = WeaviateConfig.from_environment()
    logger.info(f"Using database environment: {config.environment}")
    logger.info(f"Connecting to: {config.connection_string}")
    
    # Create repository manager
    repo_manager = BookRepositoryManager(config)
    
    try:
        # Example 1: Count existing entities
        logger.info("=== Counting existing entities ===")
        book_count = repo_manager.books.count_sync()
        chapter_count = repo_manager.chapters.count_sync()
        paragraph_count = repo_manager.paragraphs.count_sync()
        
        logger.info(f"Books: {book_count}")
        logger.info(f"Chapters: {chapter_count}")
        logger.info(f"Paragraphs: {paragraph_count}")
        
        # Example 2: List some books
        if book_count > 0:
            logger.info("=== Listing first 3 books ===")
            books = repo_manager.books.list_all_sync(limit=3)
            for book in books:
                logger.info(f"Book {book.book_index}: {book.title} (pages {book.start_page}-{book.end_page})")
        
        # Example 3: Search paragraphs (if any exist)
        if paragraph_count > 0:
            logger.info("=== Searching paragraphs ===")
            try:
                search_results = repo_manager.paragraphs.search_similar_paragraphs_sync(
                    query_text="ancient Rome",
                    limit=3
                )
                
                for i, (paragraph, score) in enumerate(search_results):
                    logger.info(f"Result {i+1} (score: {score:.3f}): {paragraph.text[:100]}...")
            except Exception as e:
                logger.warning(f"Search failed: {e}")
                logger.info("Vector search may not be available for this collection")
        
        # Example 4: Find chapters by book
        if chapter_count > 0:
            logger.info("=== Finding chapters for book 0 ===")
            chapters = repo_manager.chapters.find_by_book_index_sync(book_index=0)
            for chapter in chapters[:3]:  # Show first 3
                logger.info(f"Chapter {chapter.chapter_index}: {chapter.title}")
                
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
    
    finally:
        # Clean up connections
        repo_manager.close_all()
        logger.info("Repository connections closed")


def demonstrate_configuration():
    """Demonstrate different configuration options."""
    
    logger.info("=== Configuration Examples ===")
    
    # Example 1: Production config
    prod_config = WeaviateConfig(
        host="localhost",
        port=8080,
        grpc_port=50051,
        environment=DatabaseEnvironment.PRODUCTION
    )
    logger.info(f"Production config: {prod_config.connection_string}")
    
    # Example 2: Test config
    test_config = WeaviateConfig(
        host="localhost",
        port=8081,
        grpc_port=50052,
        environment=DatabaseEnvironment.TEST
    )
    logger.info(f"Test config: {test_config.connection_string}")
    
    # Example 3: From environment
    env_config = WeaviateConfig.from_environment()
    logger.info(f"Environment config: {env_config.connection_string} ({env_config.environment})")


if __name__ == "__main__":
    logger.info("Starting repository interface demonstration...")
    
    # Show configuration options
    demonstrate_configuration()
    
    print()
    
    # Demonstrate the interface (only if you have data)
    try:
        demonstrate_new_interface()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.info("Make sure Weaviate is running and you have data loaded")
    
    logger.info("Demonstration complete!")
