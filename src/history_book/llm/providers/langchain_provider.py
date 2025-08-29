"""LangChain implementation of LLM interface."""

import logging
from typing import List, Dict, Any, AsyncIterator
from ..interfaces.llm_interface import LLMInterface
from ..config import LLMConfig
from ..exceptions import (
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTokenLimitError,
    LLMValidationError,
    LLMResponseError,
)
from ..utils import (
    format_messages_for_llm,
    format_context_for_llm,
    estimate_token_count,
)
from ...data_models.entities import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class LangChainProvider(LLMInterface):
    """LangChain implementation of LLM interface."""

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the LangChain provider.

        Args:
            config: LLM configuration. If None, will load from environment.
        """
        self.config = config or LLMConfig.from_environment()
        self.config.validate()
        self._llm = None
        self._chat_model = None

    @property
    def llm(self):
        """Lazy-load the LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    @property
    def chat_model(self):
        """Lazy-load the chat model instance."""
        if self._chat_model is None:
            self._chat_model = self._create_chat_model()
        return self._chat_model

    def _create_llm(self):
        """Create the appropriate LLM instance based on configuration."""
        try:
            if self.config.provider == "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=self.config.model_name,
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    **self.config.provider_kwargs,
                )
            elif self.config.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=self.config.model_name,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    **self.config.provider_kwargs,
                )
            else:
                raise LLMValidationError(
                    f"Unsupported provider: {self.config.provider}"
                )

        except ImportError as e:
            raise LLMConnectionError(
                f"Missing dependency for {self.config.provider}: {e}"
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to create LLM: {e}")

    def _create_chat_model(self):
        """Create chat model (same as LLM for now, but allows for future differentiation)."""
        return self._create_llm()

    def _convert_to_langchain_messages(
        self, messages: List[ChatMessage], context: str | None = None
    ):
        """Convert ChatMessage objects to LangChain message format."""
        from langchain.schema import HumanMessage, AIMessage, SystemMessage

        # Format messages for LLM
        formatted_messages = format_messages_for_llm(
            messages,
            system_message=self.config.system_message,
            max_messages=self.config.max_conversation_length,
        )

        lc_messages = []

        for msg in formatted_messages:
            content = msg.content

            # Add context to the last user message if provided
            # TODO: handle context more gracefully? seems awkward to insert into user message?
            # TODO: what if last message is not user? shouldn't happen...
            if (
                context
                and msg.role == MessageRole.USER
                and msg == formatted_messages[-1]
            ):
                formatted_context = format_context_for_llm(
                    context, self.config.max_context_length
                )
                if formatted_context:
                    content = f"{formatted_context}\n\nUser Question: {content}"

            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=content))
            elif msg.role == MessageRole.SYSTEM:
                lc_messages.append(SystemMessage(content=content))

        return lc_messages

    async def generate_response(
        self, messages: List[ChatMessage], context: str | None = None, **kwargs: Any
    ) -> str:
        """Generate a response using LangChain."""
        try:
            # Validate input
            if not self.validate_messages(messages):
                raise LLMValidationError("Invalid message format")

            # Convert to LangChain format
            lc_messages = self._convert_to_langchain_messages(messages, context)

            # Generate response
            response = await self.chat_model.ainvoke(lc_messages, **kwargs)

            if not response or not response.content:
                raise LLMResponseError("Empty response from LLM")

            return response.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise LLMTokenLimitError(f"Token limit exceeded: {e}")
            elif isinstance(e, (LLMError,)):
                raise
            else:
                raise LLMError(f"Generation failed: {e}")

    async def generate_stream_response(
        self, messages: List[ChatMessage], context: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate a streaming response using LangChain."""
        try:
            # Validate input
            if not self.validate_messages(messages):
                raise LLMValidationError("Invalid message format")

            # Convert to LangChain format
            lc_messages = self._convert_to_langchain_messages(messages, context)

            # Generate streaming response
            async for chunk in self.chat_model.astream(lc_messages, **kwargs):
                if chunk and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            if "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise LLMTokenLimitError(f"Token limit exceeded: {e}")
            elif isinstance(e, (LLMError,)):
                raise
            else:
                raise LLMError(f"Streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens using LangChain's tokenizer if available, otherwise estimate."""
        try:
            # Try to use LangChain's token counting if available
            if hasattr(self.llm, "get_num_tokens"):
                return self.llm.get_num_tokens(text)
            else:
                # Fall back to estimation
                return estimate_token_count(text)
        except Exception:
            # Fall back to estimation if anything goes wrong
            return estimate_token_count(text)

    def validate_messages(self, messages: List[ChatMessage]) -> bool:
        """Validate message format and content."""
        if not messages:
            return False

        for msg in messages:
            if not isinstance(msg, ChatMessage):
                return False
            if not msg.content or not msg.content.strip():
                return False
            if msg.role not in [
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.SYSTEM,
            ]:
                return False

        return True
