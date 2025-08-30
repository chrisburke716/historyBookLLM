#!/usr/bin/env python3
"""
Collection management script for ensuring proper vectorization setup.

This script creates or recreates collections with proper vectorization configuration.
Use this when you need to reset collections or ensure they have the correct schema.
"""

import logging

from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def recreate_collections():
    """Recreate all collections with proper vectorization."""
    config = WeaviateConfig.from_environment()

    logger.info("🔧 COLLECTION MANAGEMENT")
    logger.info("=" * 50)
    logger.info(f"Environment: {config.environment.value}")
    logger.info(f"Weaviate URL: {config.connection_string}")

    # Check OpenAI API key
    if not config.openai_api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.error("Vector embeddings require OpenAI API access.")
        return False

    try:
        # Initialize repository manager (this will auto-create collections)
        manager = BookRepositoryManager(config)

        # Test each collection by checking if they exist and have correct config
        logger.info("\n=== Testing Collection Creation ===")

        # Check Books collection
        books_count = manager.books.count()
        logger.info(f"✅ Books collection: {books_count} entities")

        # Check Chapters collection
        chapters_count = manager.chapters.count()
        logger.info(f"✅ Chapters collection: {chapters_count} entities")

        # Check Paragraphs collection (this should have vectorization)
        paragraphs_count = manager.paragraphs.count()
        logger.info(f"✅ Paragraphs collection: {paragraphs_count} entities")

        # Test vectorization by checking if we can get a vector
        if paragraphs_count > 0:
            # Get first paragraph and check if it has a vector
            paragraphs = manager.paragraphs.list_all(limit=1)
            if paragraphs:
                test_paragraph = paragraphs[0]
                vector = manager.paragraphs.get_vector(test_paragraph.id)
                if vector:
                    logger.info(f"✅ Vectorization working: {len(vector)} dimensions")
                else:
                    logger.warning("⚠️  No vector found for existing paragraph")

        logger.info("\n=== Collection Schema Info ===")
        logger.info("Collections created with:")
        logger.info("  - Books: Basic properties, no vectorization")
        logger.info("  - Chapters: Basic properties, no vectorization")
        logger.info("  - Paragraphs: Text vectorization enabled (OpenAI)")

        manager.close_all()
        return True

    except Exception as e:
        logger.error(f"❌ Failed to manage collections: {e}")
        return False


def clear_and_recreate():
    """Clear all data and recreate collections."""
    config = WeaviateConfig.from_environment()

    logger.info("\n🗑️  CLEARING AND RECREATING COLLECTIONS")
    logger.info("=" * 50)

    try:
        # Connect to client directly to delete collections

        from history_book.database import server  # noqa: PLC0415

        client = server.get_client(config)

        with client:
            # List existing collections
            existing_collections = client.collections.list_all()
            logger.info(f"Found {len(existing_collections)} existing collections")

            # Delete collections if they exist
            for collection_name in ["Books", "Chapters", "Paragraphs"]:
                if collection_name in existing_collections:
                    logger.info(f"Deleting collection: {collection_name}")
                    client.collections.delete(collection_name)
                else:
                    logger.info(f"Collection {collection_name} doesn't exist")

            logger.info("✅ Collections cleared")

        # Now recreate them
        return recreate_collections()

    except Exception as e:
        logger.error(f"❌ Failed to clear collections: {e}")
        return False


def main():
    """Main collection management interface."""
    import sys  # noqa: PLC0415

    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        success = clear_and_recreate()
    else:
        success = recreate_collections()

    if success:
        logger.info("\n🎉 Collection management completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  - Run ingestion: python scripts/run_ingestion.py")
        logger.info(
            "  - Test search: python -c \"from history_book.database.repositories import BookRepositoryManager; from history_book.database.config import WeaviateConfig; manager = BookRepositoryManager(WeaviateConfig.from_env()); print(manager.paragraphs.similarity_search_by_text('test', limit=1))\""
        )
    else:
        logger.error("\n❌ Collection management failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
