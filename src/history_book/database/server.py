import logging

import weaviate

from .config.database_config import WeaviateConfig

logger = logging.getLogger(__name__)


class WeaviateClientManager:
    """Manages a singleton Weaviate client connection."""
    
    def __init__(self):
        self._client = None
        self._config = None
    
    def get_client(self, config: WeaviateConfig | None = None) -> weaviate.WeaviateClient:
        """
        Create and return a Weaviate client instance.

        Args:
            config: Optional WeaviateConfig. If not provided, will use environment-based config.

        Returns:
            weaviate.WeaviateClient: An instance of the Weaviate client.
        """
        # If config is provided and different from current, reset client
        if config is not None and config != self._config:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception as e:
                    logger.warning(f"Error closing previous client: {e}")
            self._client = None
            self._config = config

        # Create config from environment if none provided
        if self._config is None:
            self._config = WeaviateConfig.from_environment()

        if self._client is None or not self._client.is_connected():
            try:
                if self._config.is_local:
                    self._client = weaviate.connect_to_local(
                        port=self._config.port, grpc_port=self._config.grpc_port
                    )
                    logger.info(f"Connected to Weaviate at {self._config.connection_string}")
                else:
                    # For remote connections, you'd configure differently
                    # This is a placeholder for future remote connection support
                    raise NotImplementedError(
                        "Remote Weaviate connections not yet implemented"
                    )
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {e}")
                raise

        return self._client

    def close_client(self):
        """Close the global client connection."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("Weaviate client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {e}")
            finally:
                self._client = None
                self._config = None

    def get_config(self) -> WeaviateConfig | None:
        """Get the current configuration."""
        return self._config


# Global instance for backward compatibility
_manager = WeaviateClientManager()


def get_client(config: WeaviateConfig | None = None) -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client instance.

    Args:
        config: Optional WeaviateConfig. If not provided, will use environment-based config.

    Returns:
        weaviate.WeaviateClient: An instance of the Weaviate client.
    """
    return _manager.get_client(config)


def close_client():
    """Close the global client connection."""
    _manager.close_client()


def get_config() -> WeaviateConfig | None:
    """Get the current configuration."""
    return _manager.get_config()
