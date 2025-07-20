"""Database configuration management."""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DatabaseEnvironment(Enum):
    """Database environment types."""
    PRODUCTION = "production"
    TEST = "test"
    DEVELOPMENT = "development"


@dataclass
class WeaviateConfig:
    """Weaviate database configuration."""
    
    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    scheme: str = "http"
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    timeout: int = 30
    environment: DatabaseEnvironment = DatabaseEnvironment.PRODUCTION
    
    @classmethod
    def from_environment(cls) -> "WeaviateConfig":
        """Create configuration from environment variables."""
        env = os.getenv("DB_ENVIRONMENT", "production").lower()
        
        # Default configurations for different environments
        if env == "test":
            return cls(
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8081")),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50052")),
                scheme=os.getenv("WEAVIATE_SCHEME", "http"),
                api_key=os.getenv("WEAVIATE_API_KEY"),
                openai_api_key=os.getenv("OPENAI_APIKEY"),
                timeout=int(os.getenv("WEAVIATE_TIMEOUT", "30")),
                environment=DatabaseEnvironment.TEST
            )
        elif env == "development":
            return cls(
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                scheme=os.getenv("WEAVIATE_SCHEME", "http"),
                api_key=os.getenv("WEAVIATE_API_KEY"),
                openai_api_key=os.getenv("OPENAI_APIKEY"),
                timeout=int(os.getenv("WEAVIATE_TIMEOUT", "30")),
                environment=DatabaseEnvironment.DEVELOPMENT
            )
        else:  # production
            return cls(
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8080")),
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                scheme=os.getenv("WEAVIATE_SCHEME", "http"),
                api_key=os.getenv("WEAVIATE_API_KEY"),
                openai_api_key=os.getenv("OPENAI_APIKEY"),
                timeout=int(os.getenv("WEAVIATE_TIMEOUT", "30")),
                environment=DatabaseEnvironment.PRODUCTION
            )
    
    @property
    def is_local(self) -> bool:
        """Check if this is a local Weaviate instance."""
        return self.host in ["localhost", "127.0.0.1"]
    
    @property
    def connection_string(self) -> str:
        """Get connection string for logging/debugging."""
        return f"{self.scheme}://{self.host}:{self.port}"
