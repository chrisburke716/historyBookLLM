"""LLM configuration management."""

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL_NAME = "gpt-4o-mini"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    # Provider settings
    provider: str = "openai"  # openai, anthropic, etc.
    model_name: str = DEFAULT_MODEL_NAME
    api_key: str | None = None
    api_base: str | None = None

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Chat-specific settings
    system_message: str = (
        "You are a helpful AI assistant that answers questions about history "
        "using the provided context from historical documents. Always base your "
        "answers on the context provided and cite specific information when possible."
    )
    max_context_length: int = 4000  # Max characters for context
    max_conversation_length: int = 20  # Max messages to include in history

    # Provider-specific settings
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environment(cls, provider: str | None = None) -> "LLMConfig":
        """
        Create LLM configuration from environment variables.

        Args:
            provider: Override the provider from environment

        Returns:
            LLMConfig instance
        """
        config = cls()

        # Provider and model
        config.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        config.model_name = os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)

        # API settings
        config.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.api_base = os.getenv("LLM_API_BASE")

        # Generation parameters
        if temp := os.getenv("LLM_TEMPERATURE"):
            config.temperature = float(temp)
        if max_tokens := os.getenv("LLM_MAX_TOKENS"):
            config.max_tokens = int(max_tokens)
        if top_p := os.getenv("LLM_TOP_P"):
            config.top_p = float(top_p)

        # Chat settings
        if system_msg := os.getenv("LLM_SYSTEM_MESSAGE"):
            config.system_message = system_msg
        if max_context := os.getenv("LLM_MAX_CONTEXT_LENGTH"):
            config.max_context_length = int(max_context)
        if max_conv := os.getenv("LLM_MAX_CONVERSATION_LENGTH"):
            config.max_conversation_length = int(max_conv)

        return config

    def validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.provider:
            raise ValueError("Provider must be specified")

        if not self.model_name:
            raise ValueError("Model name must be specified")

        if self.provider == "openai" and not self.api_key:
            raise ValueError("API key is required for OpenAI provider")

        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        if not 0 <= self.top_p <= 1:
            raise ValueError("Top-p must be between 0 and 1")
