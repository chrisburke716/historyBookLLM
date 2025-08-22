# LLM Abstraction Layer

## Overview

The LLM abstraction layer provides a clean, provider-agnostic interface for working with Large Language Models in the history book application.

## Architecture

### Components

1. **Interfaces** (`src/history_book/llm/interfaces/`)
   - `LLMInterface`: Abstract base class defining the contract for LLM providers

2. **Providers** (`src/history_book/llm/providers/`)
   - `LangChainProvider`: Production provider using LangChain (requires langchain dependencies)
   - `MockLLMProvider`: Development/testing provider with simulated responses

3. **Configuration** (`src/history_book/llm/config.py`)
   - `LLMConfig`: Centralized configuration management with environment variable support

4. **Exceptions** (`src/history_book/llm/exceptions.py`)
   - Specialized exceptions for different types of LLM errors

5. **Utilities** (`src/history_book/llm/utils.py`)
   - Helper functions for message formatting, context handling, and token estimation

## Key Features

### Provider Abstraction
- Consistent interface across different LLM providers (OpenAI, Anthropic, etc.)
- Easy to swap providers without changing application code
- Graceful fallback to mock provider when dependencies unavailable

### Configuration Management
- Environment variable support for all settings
- Provider-specific configuration options
- Validation and sensible defaults

### Error Handling
- Specialized exceptions for rate limits, token limits, connection errors
- Proper error propagation and logging

### Message Handling
- Automatic formatting of chat messages for LLM consumption
- Context injection with length limits
- Token counting and text truncation

### Streaming Support
- Both synchronous and streaming response generation
- Consistent interface for real-time chat experiences

## Usage Examples

### Basic Setup
```python
from src.history_book.llm import LLMConfig, MockLLMProvider

config = LLMConfig.from_environment()
provider = MockLLMProvider(config)
```

### Generate Response
```python
messages = [ChatMessage(content="What caused WWI?", role=MessageRole.USER, session_id="test")]
context = "Historical context about WWI..."

response = await provider.generate_response(messages, context=context)
```

### Streaming Response
```python
async for chunk in provider.generate_stream_response(messages, context=context):
    print(chunk, end="")
```

## Environment Variables

- `LLM_PROVIDER`: Provider to use (default: "openai")
- `LLM_MODEL_NAME`: Model name (default: "gpt-3.5-turbo")
- `LLM_API_KEY` or `OPENAI_API_KEY`: API key for the provider
- `LLM_TEMPERATURE`: Response temperature (default: 0.7)
- `LLM_MAX_TOKENS`: Maximum response tokens
- `LLM_SYSTEM_MESSAGE`: System message for the LLM
- `LLM_MAX_CONTEXT_LENGTH`: Maximum context length (default: 4000)
- `LLM_MAX_CONVERSATION_LENGTH`: Maximum messages in conversation (default: 20)

## Integration with Chat Service

The LLM abstraction layer is designed to integrate seamlessly with the ChatService:

1. **Dependency Injection**: ChatService accepts any LLMInterface implementation
2. **Configuration**: Uses same config system as other components
3. **Error Handling**: LLM errors are properly caught and handled
4. **Context Integration**: Automatically formats retrieved paragraphs as context

## Future Extensibility

- Easy to add new providers (Cohere, local models, etc.)
- Support for function calling and tools
- Custom prompt templates and strategies
- Advanced context management and retrieval integration
