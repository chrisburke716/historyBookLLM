"""Factory for creating LangChain chat models from LLMConfig."""

import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from history_book.llm.config import LLMConfig

logger = logging.getLogger(__name__)


def build_chat_model(
    config: LLMConfig, *, temperature_override: float | None = None
) -> BaseChatModel:
    """Build a chat model from LLMConfig via init_chat_model.

    When `config.reasoning_effort` is set (gpt-5 family, o1/o3), passes
    `reasoning_effort` + `use_responses_api=True` and omits `temperature`
    (reasoning models reject anything other than the default 1.0). Otherwise
    passes `temperature_override` if provided, else `config.temperature`.

    Used by the agent graph and the title-generation chain.
    """
    model_id = f"{config.provider}:{config.model_name}"
    kwargs: dict = {}
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
        # Reasoning + bound tools requires the Responses API on gpt-5 family;
        # Chat Completions rejects the combination.
        kwargs["use_responses_api"] = True
    else:
        kwargs["temperature"] = (
            temperature_override
            if temperature_override is not None
            else config.temperature
        )
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["base_url"] = config.api_base
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens
    kwargs.update(config.provider_kwargs)
    return init_chat_model(model_id, **kwargs)
