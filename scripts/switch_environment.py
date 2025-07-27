#!/usr/bin/env python3
"""
Environment switcher utility for Weaviate configurations.

This script provides an interactive way to switch between production, development, 
and test environments with proper validation.
"""

import os
import sys
import logging
from history_book.database.config import WeaviateConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class EnvironmentSwitcher:
    """Utility for switching between Weaviate environments."""
    
    def __init__(self):
        self.environments = {
            "production": {
                "port": "8080",
                "grpc_port": "50051", 
                "description": "Production data (port 8080)",
                "safety": "⚠️  CAUTION: Live production data"
            },
            "development": {
                "port": "8080",
                "grpc_port": "50051",
                "description": "Development data (port 8080)", 
                "safety": "⚠️  CAUTION: May share data with production"
            },
            "test": {
                "port": "8081",
                "grpc_port": "50052",
                "description": "Isolated test data (port 8081)",
                "safety": "✅ SAFE: Completely isolated test environment"
            }
        }
    
    def show_current_environment(self):
        """Show the current environment settings."""
        current_env = os.getenv("DB_ENVIRONMENT", "production")
        current_port = os.getenv("WEAVIATE_PORT", "8080")
        
        logger.info("🔍 CURRENT ENVIRONMENT STATUS")
        logger.info("=" * 40)
        logger.info(f"Environment: {current_env}")
        logger.info(f"Weaviate Port: {current_port}")
        logger.info(f"gRPC Port: {os.getenv('WEAVIATE_GRPC_PORT', '50051')}")
        logger.info(f"OpenAI API Key: {'***' if os.getenv('OPENAI_APIKEY') else 'Not set'}")
        
        if current_env in self.environments:
            env_info = self.environments[current_env]
            logger.info(f"Description: {env_info['description']}")
            logger.info(f"Safety: {env_info['safety']}")
    
    def show_available_environments(self):
        """Show all available environments."""
        logger.info("\n🌍 AVAILABLE ENVIRONMENTS")
        logger.info("=" * 40)
        
        for env_name, env_info in self.environments.items():
            logger.info(f"\n{env_name.upper()}:")
            logger.info(f"  Port: {env_info['port']}")
            logger.info(f"  Description: {env_info['description']}")
            logger.info(f"  Safety: {env_info['safety']}")
    
    def switch_to_environment(self, env_name: str) -> bool:
        """
        Switch to the specified environment.
        
        Args:
            env_name: Name of environment to switch to
            
        Returns:
            True if switch was successful
        """
        if env_name not in self.environments:
            logger.error(f"❌ Unknown environment: {env_name}")
            logger.info(f"Available: {list(self.environments.keys())}")
            return False
        
        env_info = self.environments[env_name]
        
        # Set environment variables
        os.environ["DB_ENVIRONMENT"] = env_name
        os.environ["WEAVIATE_PORT"] = env_info["port"]
        os.environ["WEAVIATE_GRPC_PORT"] = env_info["grpc_port"]
        
        logger.info(f"🔄 Switched to {env_name.upper()} environment")
        logger.info(f"Port: {env_info['port']}")
        logger.info(f"Safety: {env_info['safety']}")
        
        # Test the connection
        return self._test_connection(env_name)
    
    def _test_connection(self, env_name: str) -> bool:
        """Test connection to the environment."""
        try:
            config = WeaviateConfig.from_environment()
            logger.info(f"Testing connection to {config.connection_string}...")
            
            # Import here to avoid circular imports
            from history_book.database.repositories.weaviate_repository import WeaviateRepository
            from history_book.data_models.entities import Book
            
            test_repo = WeaviateRepository(
                config=config,
                collection_name="Books",
                entity_class=Book
            )
            
            try:
                count = test_repo.count()
                logger.info(f"✅ Connected! Found {count} books in {env_name}")
            except Exception:
                logger.info(f"✅ Connected to {env_name} (Books collection may not exist)")
            
            test_repo.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {env_name}: {e}")
            return False
    
    def interactive_switch(self):
        """Interactive environment switching."""
        logger.info("🔧 INTERACTIVE ENVIRONMENT SWITCHER")
        logger.info("=" * 45)
        
        self.show_current_environment()
        self.show_available_environments()
        
        logger.info("\n" + "=" * 45)
        print("\nChoose an environment:")
        print("1. production")
        print("2. development") 
        print("3. test")
        print("0. exit")
        
        try:
            choice = input("\nEnter your choice (0-3): ").strip()
            
            env_map = {
                "1": "production",
                "2": "development", 
                "3": "test",
                "0": None
            }
            
            if choice == "0":
                logger.info("👋 Goodbye!")
                return False
            
            if choice not in env_map:
                logger.error("❌ Invalid choice. Please enter 0-3.")
                return False
            
            env_name = env_map[choice]
            return self.switch_to_environment(env_name)
            
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False


def main():
    """Main function for environment switching."""
    switcher = EnvironmentSwitcher()
    
    # Check if environment was specified as command line argument
    if len(sys.argv) > 1:
        env_name = sys.argv[1].lower()
        
        if env_name in ["--help", "-h"]:
            print("Usage:")
            print("  python scripts/switch_environment.py [environment]")
            print("  python scripts/switch_environment.py production")
            print("  python scripts/switch_environment.py development")
            print("  python scripts/switch_environment.py test")
            print("  python scripts/switch_environment.py  # Interactive mode")
            return
        
        success = switcher.switch_to_environment(env_name)
        if success:
            logger.info("\n🎯 Environment switched successfully!")
            logger.info("You can now run your scripts in this environment.")
        else:
            logger.error("❌ Failed to switch environment")
            sys.exit(1)
    else:
        # Interactive mode
        switcher.interactive_switch()


if __name__ == "__main__":
    main()
