# LLM Configuration and Utilities

## Overview

Simplified LLM module providing configuration and utilities for direct LangChain integration. No abstraction layers - RagService uses LangChain models directly.

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

# Used by RagService to create LangChain models directly
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(
    model=config.model_name,
    api_key=config.api_key,
    temperature=config.temperature
)
```

## Environment Variables

```bash
LLM_PROVIDER=openai                    # openai, anthropic
LLM_MODEL_NAME=gpt-4o-mini            # model to use
LLM_API_KEY=your-key                  # provider API key
LLM_TEMPERATURE=0.7                   # response randomness
LLM_SYSTEM_MESSAGE="You are..."       # system prompt
LLM_MAX_CONTEXT_LENGTH=4000           # max context chars
LLM_MAX_CONVERSATION_LENGTH=20        # max messages in history
```

## Integration

RagService creates LangChain models and LCEL chains directly:

```python
# In RagService.__init__()
self.chat_model = self._create_chat_model()  # Direct LangChain model
self.rag_chain = prompt | self.chat_model | StrOutputParser()  # LCEL chain
```

## Migration Notes

**Removed**: LLMInterface, providers, ResponseChain - all replaced with direct LangChain usage in RagService.
**Kept**: Configuration, utilities, exceptions - still needed for LLM operations.