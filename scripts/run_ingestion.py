#!/usr/bin/env python3
"""
Modern ingestion script using the new repository pattern.

This script demonstrates the new clean architecture:
- Pure entity models (no auto-saving)
- Repository pattern for data access
- Service layer for complex operations
- Clear separation of concerns
"""

import logging
import os
from pathlib import Path
from history_book.services import IngestionService
from history_book.database.config import WeaviateConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Constants
FINAL_PAGE = 1699  # Last page of the book
PROJECT_ROOT = Path(__file__).parents[1]
BOOK_FILE = PROJECT_ROOT / "data" / "penguin_history_6.pdf"


def check_environment_safety():
    """Check which environment we're running in and warn if necessary."""
    current_env = os.getenv("DB_ENVIRONMENT", "production")
    current_port = os.getenv("WEAVIATE_PORT", "8080")

    logger.info("🔍 ENVIRONMENT CHECK")
    logger.info("=" * 30)
    logger.info(f"Current environment: {current_env}")
    logger.info(f"Weaviate port: {current_port}")

    if current_env == "production":
        logger.warning("⚠️  WARNING: Running in PRODUCTION environment!")
        logger.warning("This will modify your production data.")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Aborted by user.")
            return False
    elif current_env == "test":
        logger.info("✅ Running in TEST environment - safe to proceed")
    elif current_env == "development":
        logger.info("🔧 Running in DEVELOPMENT environment")
        logger.warning("⚠️  This may modify development data")
        response = input("Continue with development ingestion? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Aborted by user.")
            return False

    return True


def check_existing_data_and_ask_to_clear(ingestion_service: IngestionService) -> bool:
    """Check if data exists and ask user if they want to clear it."""
    logger.info("\n🔍 Checking for existing data...")

    existing_counts = ingestion_service.check_existing_data()
    total_existing = sum(existing_counts.values())

    if total_existing == 0:
        logger.info("✅ Database is empty - ready for fresh ingestion!")
        return False

    logger.info("📊 Found existing data:")
    logger.info(f"  📚 Books: {existing_counts['books']}")
    logger.info(f"  📖 Chapters: {existing_counts['chapters']}")
    logger.info(f"  📄 Paragraphs: {existing_counts['paragraphs']}")
    logger.info(f"  🔢 Total: {total_existing} entities")

    logger.warning("⚠️  Proceeding without clearing will create duplicate data!")

    print("\nWhat would you like to do?")
    print("1. Clear existing data and run fresh ingestion")
    print("2. Add to existing data (may create duplicates)")
    print("3. Cancel ingestion")

    while True:
        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            logger.info("🗑️  User chose to clear existing data")
            return True
        elif choice == "2":
            logger.warning("⚠️  User chose to add to existing data")
            return False
        elif choice == "3":
            logger.info("Ingestion cancelled by user")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Run the modern ingestion process."""
    logger.info("🚀 MODERN BOOK INGESTION PROCESS")
    logger.info("=" * 50)

    # Check environment safety first
    if not check_environment_safety():
        return

    # Check if book file exists
    if not BOOK_FILE.exists():
        logger.error(f"Book file not found: {BOOK_FILE}")
        logger.info("Available files in data directory:")
        data_dir = PROJECT_ROOT / "data"
        if data_dir.exists():
            for file in data_dir.glob("*.pdf"):
                logger.info(f"  - {file.name}")
        return

    # Initialize the ingestion service
    config = WeaviateConfig.from_environment()
    logger.info(f"Using Weaviate at: {config.connection_string}")
    logger.info(f"Environment: {config.environment.value}")

    ingestion_service = IngestionService(config)

    try:
        # Check for existing data and ask user what to do
        clear_existing = check_existing_data_and_ask_to_clear(ingestion_service)

        if clear_existing is None:  # User cancelled
            return

        logger.info(f"📖 Starting ingestion of: {BOOK_FILE.name}")
        logger.info(f"📄 Final page: {FINAL_PAGE}")

        # Run the ingestion
        book_ids, chapter_ids, paragraph_ids = ingestion_service.ingest_book_from_pdf(
            pdf_path=BOOK_FILE, final_page=FINAL_PAGE, clear_existing=clear_existing
        )

        logger.info("🎉 INGESTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info("📊 Results:")
        logger.info(f"  📚 Books created: {len(book_ids)}")
        logger.info(f"  📖 Chapters created: {len(chapter_ids)}")
        logger.info(f"  📄 Paragraphs created: {len(paragraph_ids)}")

        # Show some sample IDs
        if book_ids:
            logger.info("\n📋 Sample IDs:")
            logger.info(f"  First book ID: {book_ids[0]}")
        if chapter_ids:
            logger.info(f"  First chapter ID: {chapter_ids[0]}")
        if paragraph_ids:
            logger.info(f"  First paragraph ID: {paragraph_ids[0]}")
            logger.info(f"  Last paragraph ID: {paragraph_ids[-1]}")

        # Show summary statistics
        logger.info("\n📈 Summary:")
        logger.info(
            f"  Average chapters per book: {len(chapter_ids) / len(book_ids):.1f}"
        )
        logger.info(
            f"  Average paragraphs per chapter: {len(paragraph_ids) / len(chapter_ids):.1f}"
        )
        logger.info(
            f"  Total entities created: {len(book_ids) + len(chapter_ids) + len(paragraph_ids)}"
        )

        logger.info("\n✨ New repository pattern benefits demonstrated:")
        logger.info("  ✅ Pure entity creation (no auto-saving)")
        logger.info("  ✅ Explicit database operations")
        logger.info("  ✅ Batch operations for efficiency")
        logger.info("  ✅ Proper error handling")
        logger.info("  ✅ Clean separation of concerns")

    except Exception as e:
        logger.error(f"❌ INGESTION FAILED: {e}")
        logger.error("Check the logs above for more details.")
        raise
    finally:
        # Always clean up connections
        ingestion_service.close()
        logger.info("🔌 Connections closed.")


if __name__ == "__main__":
    main()
