"""LLM-specific exceptions."""


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when LLM provider connection fails."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM provider rate limits are exceeded."""
    pass


class LLMTokenLimitError(LLMError):
    """Raised when token limits are exceeded."""
    pass


class LLMValidationError(LLMError):
    """Raised when input validation fails."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or malformed."""
    pass
