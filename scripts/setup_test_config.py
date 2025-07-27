#!/usr/bin/env python3
"""
Test configuration for Weaviate test instance.

This script helps set up and validate configuration for testing the new ingestion architecture
on a separate test Weaviate instance to avoid affecting your production data.
"""

import os
import logging
from history_book.database.config import WeaviateConfig, DatabaseEnvironment
from history_book.database.repositories.weaviate_repository import WeaviateRepository

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_test_config() -> WeaviateConfig:
    """
    Create a test configuration for your Weaviate test instance.

    This uses a separate port/instance to avoid affecting production data.
    """
    # Test instance configuration - modify these as needed for your setup
    test_config = WeaviateConfig(
        host="localhost",
        port=8081,  # Different port for test instance
        grpc_port=50052,  # Different gRPC port
        scheme="http",
        api_key=None,  # No API key for local test instance
        openai_api_key=os.getenv("OPENAI_APIKEY"),  # Still need this for embeddings
        timeout=30,
        environment=DatabaseEnvironment.TEST,
    )

    return test_config


def validate_test_connection(config: WeaviateConfig) -> bool:
    """
    Validate that we can connect to the test Weaviate instance.

    Args:
        config: Test configuration to validate

    Returns:
        True if connection successful, False otherwise
    """
    logger.info(f"Testing connection to {config.connection_string}")

    try:
        # Try to create a basic repository to test connection
        from history_book.data_models.entities import Book

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


def setup_test_environment():
    """Set up environment variables for testing."""
    logger.info("=== Setting up test environment ===")

    # Set environment to test mode
    os.environ["DB_ENVIRONMENT"] = "test"

    # Set test-specific ports (if not already set)
    if "WEAVIATE_PORT" not in os.environ:
        os.environ["WEAVIATE_PORT"] = "8081"
    if "WEAVIATE_GRPC_PORT" not in os.environ:
        os.environ["WEAVIATE_GRPC_PORT"] = "50052"

    logger.info("Environment variables set:")
    logger.info(f"  DB_ENVIRONMENT: {os.getenv('DB_ENVIRONMENT')}")
    logger.info(f"  WEAVIATE_PORT: {os.getenv('WEAVIATE_PORT')}")
    logger.info(f"  WEAVIATE_GRPC_PORT: {os.getenv('WEAVIATE_GRPC_PORT')}")
    logger.info(
        f"  OPENAI_APIKEY: {'***' if os.getenv('OPENAI_APIKEY') else 'Not set'}"
    )


def show_docker_commands():
    """Show Docker commands to start the test Weaviate instance."""
    logger.info("\n=== Test Instance Docker Commands ===")
    logger.info("You have a test instance configured in docker-compose.test.yml")
    logger.info("")
    logger.info("To start your test instance:")
    logger.info("   docker-compose -f docker-compose.test.yml up -d")
    logger.info("")
    logger.info("To stop your test instance:")
    logger.info("   docker-compose -f docker-compose.test.yml down")
    logger.info("")
    logger.info("To check if it's running:")
    logger.info("   curl http://localhost:8081/v1/meta")
    logger.info("")
    logger.info("Your test instance configuration:")
    logger.info("   - Port: 8081 (mapped from container 8080)")
    logger.info("   - gRPC Port: 50052 (mapped from container 50051)")
    logger.info("   - Version: 1.31.0")
    logger.info("   - Volume: weaviate_data_test")


def main():
    """Set up and validate test configuration."""
    logger.info("🧪 WEAVIATE TEST CONFIGURATION SETUP")
    logger.info("=" * 50)

    # Show Docker commands first
    show_docker_commands()

    # Set up environment
    setup_test_environment()

    # Create test config
    logger.info("\n=== Creating test configuration ===")
    test_config = create_test_config()
    logger.info(f"Test instance: {test_config.connection_string}")
    logger.info(f"Environment: {test_config.environment.value}")

    # Validate connection
    logger.info("\n=== Validating connection ===")
    if validate_test_connection(test_config):
        logger.info("✅ Test configuration is ready!")
        logger.info("\nYou can now run:")
        logger.info("  python scripts/test_new_ingestion_architecture.py")
        return True
    else:
        logger.error("❌ Test configuration failed!")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Make sure test Weaviate instance is running on port 8081")
        logger.error("  2. Check Docker commands above to start test instance")
        logger.error("  3. Verify OPENAI_APIKEY environment variable is set")
        return False


if __name__ == "__main__":
    main()
