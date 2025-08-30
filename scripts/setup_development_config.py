#!/usr/bin/env python3
"""
Development environment configuration script.

This script helps set up and validate configuration for the development environment
so you can easily switch between development and test environments.
"""

import logging
import os

from history_book.database.config import DatabaseEnvironment, WeaviateConfig
from history_book.database.repositories.weaviate_repository import WeaviateRepository

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_development_config() -> WeaviateConfig:
    """
    Create a development configuration for your main Weaviate instance.
    """
    # Development instance configuration
    dev_config = WeaviateConfig(
        host="localhost",
        port=8080,  # Standard port for main instance
        grpc_port=50051,  # Standard gRPC port
        scheme="http",
        api_key=None,  # No API key for local development instance
        openai_api_key=os.getenv("OPENAI_APIKEY"),  # Need this for embeddings
        timeout=30,
        environment=DatabaseEnvironment.DEVELOPMENT,
    )

    return dev_config


def validate_development_connection(config: WeaviateConfig) -> bool:
    """
    Validate that we can connect to the development Weaviate instance.

    Args:
        config: Development configuration to validate

    Returns:
        True if connection successful, False otherwise
    """
    logger.info(f"Testing connection to {config.connection_string}")

    try:
        # Try to create a basic repository to test connection
        from history_book.data_models.entities import Book  # noqa: PLC0415

        test_repo = WeaviateRepository(
            config=config, collection_name="TestBooks", entity_class=Book
        )

        # Try to count items (this will fail gracefully if collection doesn't exist)
        try:
            count = test_repo.count()
            logger.info(
                f"✅ Connected successfully! Found {count} items in TestBooks collection"
            )
        except Exception:
            logger.info(
                "✅ Connected successfully! (TestBooks collection may not exist yet)"
            )

        test_repo.close()
        return True

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False


def setup_development_environment():
    """Set up environment variables for development."""
    logger.info("=== Setting up development environment ===")

    # Set environment to development mode
    os.environ["DB_ENVIRONMENT"] = "development"

    # Set development-specific ports (if not already set)
    if "WEAVIATE_PORT" not in os.environ:
        os.environ["WEAVIATE_PORT"] = "8080"
    if "WEAVIATE_GRPC_PORT" not in os.environ:
        os.environ["WEAVIATE_GRPC_PORT"] = "50051"

    logger.info("Environment variables set:")
    logger.info(f"  DB_ENVIRONMENT: {os.getenv('DB_ENVIRONMENT')}")
    logger.info(f"  WEAVIATE_PORT: {os.getenv('WEAVIATE_PORT')}")
    logger.info(f"  WEAVIATE_GRPC_PORT: {os.getenv('WEAVIATE_GRPC_PORT')}")
    logger.info(
        f"  OPENAI_APIKEY: {'***' if os.getenv('OPENAI_APIKEY') else 'Not set'}"
    )


def show_docker_commands():
    """Show Docker commands to start the development Weaviate instance."""
    logger.info("\n=== Development Instance Docker Commands ===")
    logger.info(
        "Your main development instance should be configured in docker-compose.yml"
    )
    logger.info("")
    logger.info("To start your development instance:")
    logger.info("   docker-compose up -d")
    logger.info("")
    logger.info("To stop your development instance:")
    logger.info("   docker-compose down")
    logger.info("")
    logger.info("To check if it's running:")
    logger.info("   curl http://localhost:8080/v1/meta")
    logger.info("")
    logger.info("Your development instance configuration:")
    logger.info("   - Port: 8080 (standard)")
    logger.info("   - gRPC Port: 50051 (standard)")
    logger.info("   - Environment: development")
    logger.info("   - Volume: weaviate_data (standard)")


def show_environment_switching():
    """Show how to switch between environments."""
    logger.info("\n=== ENVIRONMENT SWITCHING ===")
    logger.info("You can easily switch between environments:")
    logger.info("")
    logger.info("🔧 For DEVELOPMENT environment:")
    logger.info("   export DB_ENVIRONMENT=development")
    logger.info("   python scripts/setup_dev_config.py")
    logger.info("   python scripts/run_modern_ingestion.py")
    logger.info("")
    logger.info("🧪 For TEST environment:")
    logger.info("   export DB_ENVIRONMENT=test")
    logger.info("   python scripts/setup_test_config.py")
    logger.info("   python scripts/run_modern_ingestion.py")
    logger.info("")
    logger.info("📊 To inspect/clear data:")
    logger.info("   python scripts/inspect_and_clear_database.py")
    logger.info("")
    logger.info("The scripts will automatically use the correct ports:")
    logger.info("   - Development: localhost:8080")
    logger.info("   - Test: localhost:8081")


def main():
    """Set up and validate development configuration."""
    logger.info("🔧 DEVELOPMENT ENVIRONMENT SETUP")
    logger.info("=" * 50)

    # Show Docker commands first
    show_docker_commands()

    # Set up environment
    setup_development_environment()

    # Create development config
    logger.info("\n=== Creating development configuration ===")
    dev_config = create_development_config()
    logger.info(f"Development instance: {dev_config.connection_string}")
    logger.info(f"Environment: {dev_config.environment.value}")

    # Validate connection
    logger.info("\n=== Validating connection ===")
    if validate_development_connection(dev_config):
        logger.info("✅ Development configuration is ready!")
        logger.info("\nYou can now run:")
        logger.info("  python scripts/run_modern_ingestion.py")
        logger.info("  python scripts/inspect_and_clear_database.py")

        # Show environment switching info
        show_environment_switching()

        return True
    else:
        logger.error("❌ Development configuration failed!")
        logger.error("\nTroubleshooting:")
        logger.error(
            "  1. Make sure development Weaviate instance is running on port 8080"
        )
        logger.error("  2. Check: docker-compose up -d")
        logger.error("  3. Verify OPENAI_APIKEY environment variable is set")
        return False


if __name__ == "__main__":
    main()
