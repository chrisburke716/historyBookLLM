#!/usr/bin/env python3
"""
Database inspection script to check what data exists before ingestion.

This script helps you see what's currently in your database and provides
options to clear data before running a fresh ingestion.

Note: This will delete all items in the specified collections, but will not delete the collections themselves.
If you need to delete the collections, use the `manage_collections.py` script.
"""

import logging
import os
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup_environment(env_name: str):
    """Set up environment variables for the specified environment."""
    os.environ["DB_ENVIRONMENT"] = env_name

    if env_name == "development":
        # Use default development ports
        if "WEAVIATE_PORT" not in os.environ:
            os.environ["WEAVIATE_PORT"] = "8080"
        if "WEAVIATE_GRPC_PORT" not in os.environ:
            os.environ["WEAVIATE_GRPC_PORT"] = "50051"
    elif env_name == "test":
        # Use test ports
        if "WEAVIATE_PORT" not in os.environ:
            os.environ["WEAVIATE_PORT"] = "8081"
        if "WEAVIATE_GRPC_PORT" not in os.environ:
            os.environ["WEAVIATE_GRPC_PORT"] = "50052"


def inspect_database(config: WeaviateConfig) -> dict:
    """Inspect the current database state."""
    logger.info(f"🔍 Inspecting database at {config.connection_string}")

    repo_manager = BookRepositoryManager(config)

    try:
        # Count entities in each collection
        books_count = repo_manager.books.count()
        chapters_count = repo_manager.chapters.count()
        paragraphs_count = repo_manager.paragraphs.count()

        # Get some sample data
        sample_books = repo_manager.books.list_all(limit=3)
        sample_chapters = repo_manager.chapters.list_all(limit=3)
        sample_paragraphs = repo_manager.paragraphs.list_all(limit=3)

        return {
            "counts": {
                "books": books_count,
                "chapters": chapters_count,
                "paragraphs": paragraphs_count,
            },
            "samples": {
                "books": sample_books,
                "chapters": sample_chapters,
                "paragraphs": sample_paragraphs,
            },
        }
    finally:
        repo_manager.close_all()


def clear_database(config: WeaviateConfig, collection_types: list = None):
    """Clear specified collections from the database."""
    if collection_types is None:
        collection_types = [
            "paragraphs",
            "chapters",
            "books",
        ]  # Order matters for dependencies

    logger.info(f"🗑️  Clearing collections: {collection_types}")

    repo_manager = BookRepositoryManager(config)

    try:
        deleted_counts = {}

        for collection_type in collection_types:
            if collection_type == "books":
                # Get all books and delete them
                all_books = repo_manager.books.list_all()
                for book in all_books:
                    if book.id:
                        repo_manager.books.delete(book.id)
                deleted_counts["books"] = len(all_books)

            elif collection_type == "chapters":
                # Get all chapters and delete them
                all_chapters = repo_manager.chapters.list_all()
                for chapter in all_chapters:
                    if chapter.id:
                        repo_manager.chapters.delete(chapter.id)
                deleted_counts["chapters"] = len(all_chapters)

            elif collection_type == "paragraphs":
                # Get all paragraphs and delete them
                all_paragraphs = repo_manager.paragraphs.list_all()
                for paragraph in all_paragraphs:
                    if paragraph.id:
                        repo_manager.paragraphs.delete(paragraph.id)
                deleted_counts["paragraphs"] = len(all_paragraphs)

        return deleted_counts

    finally:
        repo_manager.close_all()


def main():
    """Main inspection and clearing script."""
    logger.info("📊 DATABASE INSPECTION & CLEARING TOOL")
    logger.info("=" * 50)

    # Ask which environment to inspect
    print("\nWhich environment would you like to inspect?")
    print("1. Development (port 8080)")
    print("2. Test (port 8081)")
    print("3. Production (port 8080)")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        env_name = "development"
    elif choice == "2":
        env_name = "test"
    elif choice == "3":
        env_name = "production"
    else:
        logger.error("Invalid choice")
        return

    # Set up environment
    setup_environment(env_name)
    config = WeaviateConfig.from_environment()

    logger.info(f"\n🎯 Inspecting {env_name} environment")
    logger.info(f"Database: {config.connection_string}")

    try:
        # Inspect current state
        data = inspect_database(config)
        counts = data["counts"]
        samples = data["samples"]

        logger.info("\n📈 Current Database State:")
        logger.info(f"  📚 Books: {counts['books']}")
        logger.info(f"  📖 Chapters: {counts['chapters']}")
        logger.info(f"  📄 Paragraphs: {counts['paragraphs']}")

        total_entities = sum(counts.values())
        logger.info(f"  🔢 Total entities: {total_entities}")

        # Show sample data if available
        if samples["books"]:
            logger.info("\n📋 Sample Books:")
            for book in samples["books"][:3]:
                logger.info(
                    f"  - {book.title} (pages {book.start_page}-{book.end_page})"
                )

        if samples["paragraphs"]:
            logger.info("\n📝 Sample Paragraphs:")
            for para in samples["paragraphs"][:2]:
                logger.info(f"  - Page {para.page}: {para.text[:60]}...")

        # Ask if user wants to clear data
        if total_entities > 0:
            logger.info(f"\n🤔 Database contains {total_entities} entities.")
            print("\nWould you like to clear the database before ingestion?")
            print("This will permanently delete all data in this environment.")

            clear_choice = input("Clear database? (yes/no): ").strip().lower()

            if clear_choice == "yes":
                logger.info("🗑️  Clearing database...")
                deleted_counts = clear_database(config)

                logger.info("✅ Database cleared!")
                for collection, count in deleted_counts.items():
                    logger.info(f"  Deleted {count} {collection}")

                # Verify it's empty
                logger.info("\n🔍 Verifying database is empty...")
                final_data = inspect_database(config)
                final_total = sum(final_data["counts"].values())

                if final_total == 0:
                    logger.info(
                        "✅ Database is now empty and ready for fresh ingestion!"
                    )
                else:
                    logger.warning(f"⚠️  Database still contains {final_total} entities")
            else:
                logger.info("Database left unchanged.")
        else:
            logger.info("✅ Database is already empty - ready for ingestion!")

    except Exception as e:
        logger.error(f"❌ Error inspecting database: {e}")
        raise


if __name__ == "__main__":
    main()
