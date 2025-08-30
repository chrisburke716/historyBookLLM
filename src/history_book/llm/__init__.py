"""LLM abstraction layer for the history book application."""

from . import utils
from .config import LLMConfig
from .exceptions import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTokenLimitError,
    LLMValidationError,
)
from .interfaces import LLMInterface
from .providers import MockLLMProvider

# Try to import LangChain provider if available
try:
    from .providers import LangChainProvider

    langchain_available = True
except ImportError:
    langchain_available = False

__all__ = [
    "LLMConfig",
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMTokenLimitError",
    "LLMValidationError",
    "LLMResponseError",
    "LLMInterface",
    "MockLLMProvider",
    "utils",
]

if langchain_available:
    __all__.append("LangChainProvider")
