# LLM Configuration and Utilities

## Overview

Simplified LLM module providing configuration and utilities for direct LangChain integration. No provider-abstraction layers — `ChatService` and the LangGraph agent consume LangChain chat models directly via the `build_chat_model` factory.

## Components

- **`LLMConfig`**: Environment-driven configuration for LLM providers
- **`utils`**: Message formatting and context handling helpers
- **`exceptions`**: Typed LLM error handling

## Usage

```python
from history_book.llm.config import LLMConfig

# Load configuration
config = LLMConfig.from_environment()
config.validate()

# Build a LangChain chat model via the canonical factory
from history_book.llm.factory import build_chat_model
chat_model = build_chat_model(config)
```

## Environment Variables

```bash
LLM_PROVIDER=openai                    # openai, anthropic
LLM_MODEL_NAME=gpt-5.4-mini           # model to use
LLM_API_KEY=your-key                  # provider API key
LLM_TEMPERATURE=0.7                   # response randomness
LLM_SYSTEM_MESSAGE="You are..."       # system prompt
LLM_MAX_CONTEXT_LENGTH=4000           # max context chars
LLM_MAX_CONVERSATION_LENGTH=20        # max messages in history
```

## Integration

The agent graph and the title-generation chain both go through one canonical
factory:

```python
from history_book.llm.factory import build_chat_model
from history_book.llm.config import LLMConfig

llm = build_chat_model(LLMConfig.from_environment())
```

`build_chat_model` handles the gpt-5 / o1 / o3 conditional (reasoning_effort +
Responses API ↔ temperature) so callers don't repeat it.

## Migration Notes

**Removed**: LLMInterface, custom provider classes, ResponseChain — replaced with the single `build_chat_model(LLMConfig)` factory and direct LangChain consumption.
**Kept**: Configuration, utilities, exceptions — still needed for LLM operations.