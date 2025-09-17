"""Utility functions for LLM operations."""

from history_book.data_models.entities import ChatMessage, MessageRole


def format_messages_for_llm(
    messages: list[ChatMessage],
    system_message: str | None = None,
    max_messages: int | None = None,
) -> list[ChatMessage]:
    """
    Format chat messages for LLM consumption.

    Args:
        messages: Raw chat messages
        system_message: Optional system message to prepend
        max_messages: Maximum number of messages to include

    Returns:
        Formatted messages ready for LLM
    """
    formatted_messages = []

    # Add system message if provided
    if system_message:
        formatted_messages.append(
            ChatMessage(
                content=system_message,
                role=MessageRole.SYSTEM,
                session_id="system",  # Placeholder session ID
            )
        )

    # Sort messages by timestamp and limit if needed
    sorted_messages = sorted(messages, key=lambda m: m.timestamp)
    if max_messages:
        # Keep the most recent messages
        sorted_messages = sorted_messages[-max_messages:]

    formatted_messages.extend(sorted_messages)
    return formatted_messages


def format_context_for_llm(
    context: str | None, max_length: int | None = None
) -> str | None:
    """
    Format context text for LLM consumption.

    Args:
        context: Raw context text
        max_length: Maximum length of context

    Returns:
        Formatted context or None
    """
    if not context:
        return None

    # Truncate if too long
    if max_length and len(context) > max_length:
        # Try to truncate at sentence boundaries
        truncated = context[:max_length]
        last_period = truncated.rfind(".")
        if last_period > max_length * 0.8:  # Only if we don't lose too much
            truncated = truncated[: last_period + 1]
        context = truncated + "..."

    return f"""
Context from historical documents:

{context}

Please answer the question based on the context provided above.
""".strip()


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text (rough approximation).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximate token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4  # Rough approximation
    if len(text) <= max_chars:
        return text

    # Try to truncate at sentence boundaries
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:  # Only if we don't lose too much
        return truncated[: last_period + 1]

    return truncated + "..."
