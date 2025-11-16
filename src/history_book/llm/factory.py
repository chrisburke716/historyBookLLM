"""Factory for creating LangChain chat models from LLMConfig.

This module provides a reusable function to create LangChain chat models
in a provider-agnostic way, eliminating code duplication across services.
"""

import logging

from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMConnectionError, LLMValidationError

logger = logging.getLogger(__name__)


def create_chat_model(config: LLMConfig | None = None, streaming: bool = False):
    """
    Create a LangChain chat model from configuration.

    This is the canonical way to create LLM instances across the application.
    Supports both OpenAI and Anthropic providers.

    Args:
        config: LLM configuration. If None, loads from environment.
        streaming: Whether to enable streaming mode.

    Returns:
        LangChain chat model instance (ChatOpenAI or ChatAnthropic)

    Raises:
        LLMValidationError: If provider is not supported
        LLMConnectionError: If required dependencies are missing

    Example:
        >>> from history_book.llm.factory import create_chat_model
        >>> from history_book.llm.config import LLMConfig
        >>>
        >>> # Use environment config
        >>> llm = create_chat_model()
        >>>
        >>> # Use custom config
        >>> config = LLMConfig(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
        >>> llm = create_chat_model(config)
        >>>
        >>> # Enable streaming
        >>> llm = create_chat_model(streaming=True)
    """
    # Load config from environment if not provided
    if config is None:
        config = LLMConfig.from_environment()

    try:
        if config.provider == "openai":
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            return ChatOpenAI(
                model=config.model_name,
                api_key=config.api_key,
                base_url=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                streaming=streaming,
                **config.provider_kwargs,
            )

        elif config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            return ChatAnthropic(
                model=config.model_name,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                streaming=streaming,
                **config.provider_kwargs,
            )

        else:
            raise LLMValidationError(f"Unsupported provider: {config.provider}")

    except ImportError as e:
        raise LLMConnectionError(
            f"Missing dependency for {config.provider}: {e}"
        ) from e
    except Exception as e:
        raise LLMConnectionError(f"Failed to create chat model: {e}") from e
