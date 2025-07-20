import weaviate
import logging
from typing import Optional
from .config.database_config import WeaviateConfig

logger = logging.getLogger(__name__)

_client = None
_config = None


def get_client(config: Optional[WeaviateConfig] = None) -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client instance.
    
    Args:
        config: Optional WeaviateConfig. If not provided, will use environment-based config.

    Returns:
        weaviate.WeaviateClient: An instance of the Weaviate client.
    """
    global _client, _config
    
    # If config is provided and different from current, reset client
    if config is not None and config != _config:
        if _client is not None:
            try:
                _client.close()
            except Exception as e:
                logger.warning(f"Error closing previous client: {e}")
        _client = None
        _config = config
    
    # Create config from environment if none provided
    if _config is None:
        _config = WeaviateConfig.from_environment()
    
    if _client is None:
        try:
            if _config.is_local:
                _client = weaviate.connect_to_local(
                    port=_config.port,
                    grpc_port=_config.grpc_port
                )
                logger.info(f"Connected to Weaviate at {_config.connection_string}")
            else:
                # For remote connections, you'd configure differently
                # This is a placeholder for future remote connection support
                raise NotImplementedError("Remote Weaviate connections not yet implemented")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
            
    return _client


def close_client():
    """Close the global client connection."""
    global _client, _config
    if _client is not None:
        try:
            _client.close()
            logger.info("Weaviate client connection closed")
        except Exception as e:
            logger.warning(f"Error closing Weaviate client: {e}")
        finally:
            _client = None
            _config = None


def get_config() -> Optional[WeaviateConfig]:
    """Get the current configuration."""
    return _config
