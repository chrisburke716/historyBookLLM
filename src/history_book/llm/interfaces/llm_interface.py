"""Abstract interface for LLM operations."""

from abc import ABC, abstractmethod
from typing import Any

from ...data_models.entities import ChatMessage


class LLMInterface(ABC):
    """Abstract interface for LLM operations."""

    @abstractmethod
    async def generate_response(
        self, messages: list[ChatMessage], context: str | None = None, **kwargs: Any
    ) -> str:
        """
        Generate a response based on chat history and optional context.

        Args:
            messages: List of chat messages forming the conversation history
            context: Optional context text (e.g., retrieved paragraphs)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream_response(
        self, messages: list[ChatMessage], context: str | None = None, **kwargs: Any
    ) -> Any:  # AsyncIterator[str] - but avoiding complex typing for now
        """
        Generate a streaming response based on chat history and optional context.

        Args:
            messages: List of chat messages forming the conversation history
            context: Optional context text (e.g., retrieved paragraphs)
            **kwargs: Additional provider-specific parameters

        Yields:
            Response text chunks

        Raises:
            LLMError: If generation fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def validate_messages(self, messages: list[ChatMessage]) -> bool:
        """
        Validate that the message list is compatible with this LLM provider.

        Args:
            messages: Messages to validate

        Returns:
            True if valid, False otherwise
        """
        pass
