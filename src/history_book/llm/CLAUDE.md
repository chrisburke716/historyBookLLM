# CLAUDE.md - LLM Configuration

Provider-agnostic LLM configuration for OpenAI and Anthropic models. Used by RagService for direct LangChain integration.

## Quick Reference

**Key Files**:
- `config.py` - LLMConfig dataclass for provider settings
- `utils.py` - Message and context formatting helpers
- `exceptions.py` - Typed LLM error handling
- `README.md` - Overview and migration notes

**Architecture**: No abstraction layer - RagService creates LangChain models directly using LLMConfig settings.

## LLMConfig

**File**: `config.py`

### Configuration Options

```python
@dataclass
class LLMConfig:
    # Provider settings
    provider: str = "openai"  # "openai" or "anthropic"
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None  # Optional custom endpoint

    # Generation parameters
    temperature: float = 0.7  # 0.0-2.0, higher = more random
    max_tokens: int | None = None  # Max response length
    top_p: float = 1.0  # Nucleus sampling
    frequency_penalty: float = 0.0  # OpenAI only
    presence_penalty: float = 0.0  # OpenAI only

    # Chat settings
    system_message: str = "You are a helpful AI assistant..."
    max_context_length: int = 100000  # Max chars for context
    max_conversation_length: int = 20  # Max messages in history

    # Provider-specific
    provider_kwargs: dict[str, Any] = {}
```

### From Environment

```python
from history_book.llm.config import LLMConfig

# Load from environment variables
config = LLMConfig.from_environment()
config.validate()  # Raises ValueError if invalid

# Override provider
config = LLMConfig.from_environment(provider="anthropic")
```

### Environment Variables

```bash
# Provider and model
LLM_PROVIDER=openai                    # openai, anthropic
LLM_MODEL_NAME=gpt-4o-mini            # Model to use
LLM_API_KEY=sk-...                    # API key (or OPENAI_API_KEY)
LLM_API_BASE=https://...              # Optional custom endpoint

# Generation parameters
LLM_TEMPERATURE=0.7                   # 0.0-2.0
LLM_MAX_TOKENS=1000                   # Optional limit
LLM_TOP_P=1.0                         # 0.0-1.0

# Chat settings
LLM_SYSTEM_MESSAGE="You are..."       # System prompt
LLM_MAX_CONTEXT_LENGTH=100000         # Max context chars
LLM_MAX_CONVERSATION_LENGTH=20        # Max messages
```

### Supported Providers

**OpenAI**:
```bash
export LLM_PROVIDER=openai
export LLM_MODEL_NAME=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo
export LLM_API_KEY=sk-...
```

**Anthropic**:
```bash
export LLM_PROVIDER=anthropic
export LLM_MODEL_NAME=claude-3-5-sonnet-20241022
export LLM_API_KEY=sk-ant-...
```

### Validation

```python
config.validate()  # Raises ValueError if:
# - Missing required fields (provider, model_name)
# - Invalid ranges (temperature, top_p)
# - Missing API key for OpenAI
```

## Utilities

**File**: `utils.py`

### format_messages_for_llm()

```python
def format_messages_for_llm(
    messages: list[ChatMessage],
    system_message: str | None = None,
    max_messages: int | None = None,
) -> list[ChatMessage]
```

**Purpose**: Format chat history for LLM consumption.

**Features**:
- Adds optional system message
- Sorts by timestamp
- Limits to most recent N messages

**Used by**: RagService to prepare conversation history.

### format_context_for_llm()

```python
def format_context_for_llm(
    context: str | None,
    max_length: int | None = None
) -> str | None
```

**Purpose**: Format retrieved context with wrapper text.

**Output**:
```
Context from historical documents:

[context text here]

Please answer the question based on the context provided above.
```

**Features**:
- Truncates at sentence boundaries if too long
- Adds ellipsis for truncated text

### Token Estimation

```python
estimate_token_count(text: str) -> int
truncate_to_token_limit(text: str, max_tokens: int) -> str
```

**Purpose**: Rough token counting (1 token ≈ 4 chars).

**Note**: Approximation only - use for estimates, not strict limits.

## Exceptions

**File**: `exceptions.py`

```python
class LLMError(Exception):                    # Base exception
class LLMConnectionError(LLMError):           # Connection failed
class LLMRateLimitError(LLMError):            # Rate limit exceeded
class LLMTokenLimitError(LLMError):           # Token limit exceeded
class LLMValidationError(LLMError):           # Invalid config/input
class LLMResponseError(LLMError):             # Malformed response
```

**Usage**:
```python
from history_book.llm.exceptions import LLMConnectionError

try:
    model = create_chat_model()
except LLMConnectionError as e:
    logger.error(f"Failed to connect: {e}")
```

## Integration with RagService

**File**: `/src/history_book/services/rag_service.py`

### Creating LangChain Models

```python
# RagService.__init__()
self.config = llm_config or LLMConfig.from_environment()
self.chat_model = self._create_chat_model()

# RagService._create_chat_model()
if self.config.provider == "openai":
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=self.config.model_name,
        api_key=self.config.api_key,
        temperature=self.config.temperature,
        max_tokens=self.config.max_tokens,
        # ... other params from config
    )
elif self.config.provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=self.config.model_name,
        api_key=self.config.api_key,
        temperature=self.config.temperature,
        # ... other params from config
    )
```

### Building LCEL Chains

```python
# RagService._build_rag_chain()
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", self.config.system_message),  # From LLMConfig
    MessagesPlaceholder("chat_history"),
    ("human", "{context}\n\nQuestion: {query}"),
])
rag_chain = rag_prompt | self.chat_model | StrOutputParser()
```

## Common Tasks

### Switching Providers

```bash
# OpenAI → Anthropic
export LLM_PROVIDER=anthropic
export LLM_MODEL_NAME=claude-3-5-sonnet-20241022
export LLM_API_KEY=sk-ant-...

# Restart services
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload
```

### Changing Models

```bash
# Use GPT-4 instead of GPT-4o-mini
export LLM_MODEL_NAME=gpt-4

# Use different Claude model
export LLM_MODEL_NAME=claude-3-opus-20240229
```

### Adjusting Temperature

```bash
# More deterministic (less creative)
export LLM_TEMPERATURE=0.3

# More random (more creative)
export LLM_TEMPERATURE=1.0
```

### Custom System Prompt

```bash
export LLM_SYSTEM_MESSAGE="You are a history expert specializing in ancient civilizations. Always cite specific dates and sources when possible."
```

### Programmatic Override

```python
from history_book.llm.config import LLMConfig
from history_book.services import ChatService

# Custom config
custom_config = LLMConfig(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
    temperature=0.5,
    system_message="Custom prompt"
)

# Use in service
chat_service = ChatService(llm_config=custom_config)
```

## Metadata Export

**For Evaluations**: ChatService exports LLM config for experiment tracking.

```python
# ChatService.get_eval_metadata()
{
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": None,
        "system_message": "..."
    },
    # ... other metadata
}
```

Used by evaluation scripts to track configuration across runs.

## Design Philosophy

**No Abstraction Layer**: Previous versions had `LLMInterface`, custom provider classes, etc. Now:
- RagService creates LangChain models directly
- LLMConfig provides settings only
- Direct LCEL chain usage
- Simpler, more maintainable

**What Remains**:
- Configuration management (environment-driven)
- Utility functions (message/context formatting)
- Exception types (typed error handling)

## Related Files

- RagService: `/src/history_book/services/CLAUDE.md` - Uses LLMConfig to create models
- ChatService: `/src/history_book/services/CLAUDE.md` - Exports config for evals
- Evaluations: `/src/history_book/evals/CLAUDE.md` - Tracks LLM configuration
- Entity Models: `/src/history_book/data_models/entities.py` - ChatMessage for history formatting
