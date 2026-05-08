# CLAUDE.md - LLM Configuration

Provider-agnostic LLM configuration for OpenAI and Anthropic models. Consumed by `ChatService` and the LangGraph agent via the `build_chat_model` factory.

## Quick Reference

**Key Files**:
- `config.py` - LLMConfig dataclass for provider settings
- `utils.py` - Message and context formatting helpers
- `exceptions.py` - Typed LLM error handling
- `README.md` - Overview and migration notes

**Architecture**: No provider-abstraction layer — `build_chat_model(LLMConfig)` constructs a LangChain chat model via `init_chat_model` and is the single call site for both the agent's per-turn LLM and the title-generation chain.

## LLMConfig

**File**: `config.py`

### Configuration Options

```python
@dataclass
class LLMConfig:
    # Provider settings
    provider: str = "openai"  # "openai" or "anthropic"
    model_name: str = "gpt-5.4-mini"
    api_key: str | None = None
    api_base: str | None = None  # Optional custom endpoint

    # Generation parameters
    temperature: float = 0.7  # 0.0-2.0, higher = more random
    max_tokens: int | None = None  # Max response length
    top_p: float = 1.0  # Nucleus sampling
    frequency_penalty: float = 0.0  # OpenAI only
    presence_penalty: float = 0.0  # OpenAI only
    reasoning_effort: str | None = "low"  # gpt-5 / o1 / o3: "none" | "low" | "medium" | "high"

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
LLM_MODEL_NAME=gpt-5.4-mini           # Model to use
LLM_API_KEY=sk-...                    # API key (or OPENAI_API_KEY)
LLM_API_BASE=https://...              # Optional custom endpoint

# Generation parameters
LLM_TEMPERATURE=0.7                   # 0.0-2.0 (ignored when LLM_REASONING_EFFORT is set)
LLM_MAX_TOKENS=1000                   # Optional limit
LLM_TOP_P=1.0                         # 0.0-1.0
LLM_REASONING_EFFORT=low              # gpt-5 / o1 / o3 only: none | low | medium | high

# Chat settings
LLM_SYSTEM_MESSAGE="You are..."       # System prompt
LLM_MAX_CONTEXT_LENGTH=100000         # Max context chars
LLM_MAX_CONVERSATION_LENGTH=20        # Max messages
```

### Supported Providers

**OpenAI**:
```bash
export LLM_PROVIDER=openai
export LLM_MODEL_NAME=gpt-5.4-mini  # or gpt-4o, gpt-4o-mini, etc.
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

**Used by**: `ChatService` to prepare conversation history.

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
    model = build_chat_model(LLMConfig.from_environment())
except LLMConnectionError as e:
    logger.error(f"Failed to connect: {e}")
```

## Factory: `build_chat_model`

**File**: `factory.py`

```python
def build_chat_model(
    config: LLMConfig, *, temperature_override: float | None = None
) -> BaseChatModel
```

Single canonical builder used by the agent graph and the title-generation
chain. Wraps `langchain.chat_models.init_chat_model("provider:model", ...)` and
handles the gpt-5 / o1 / o3 reasoning-model conditional:

- When `config.reasoning_effort` is set: passes `reasoning_effort=` and
  `use_responses_api=True` (Chat Completions rejects reasoning + bound tools);
  omits `temperature` (reasoning models require the default 1.0).
- Otherwise: passes `temperature_override` if provided, else
  `config.temperature`.

Always forwards `api_key`, `base_url`, `max_tokens` (when set), and
`provider_kwargs`.

**Call sites**:
- `services/agents/rag_agent.py` — agent's per-turn LLM
- `services/chat_service.py:maybe_regenerate_title` — title generation
  (passes `temperature_override=0.3` for the non-reasoning path)

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
# Switch to a non-reasoning OpenAI model
export LLM_MODEL_NAME=gpt-4o
export LLM_REASONING_EFFORT=                  # unset to honor LLM_TEMPERATURE

# Use a Claude model
export LLM_MODEL_NAME=claude-3-opus-20240229
export LLM_REASONING_EFFORT=                  # Anthropic ignores this
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
        "model": "gpt-5.4-mini",
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
- `build_chat_model(LLMConfig)` returns a LangChain chat model directly
- LLMConfig provides settings only
- The agent and chains consume the chat model directly (no service-level wrapper)
- Simpler, more maintainable

**What Remains**:
- Configuration management (environment-driven)
- Utility functions (message/context formatting)
- Exception types (typed error handling)

## Related Files

- ChatService / agent: `/src/history_book/services/CLAUDE.md` and `/src/history_book/services/agents/CLAUDE.md` — call `build_chat_model(LLMConfig)`
- Evaluations: `/src/history_book/evals/CLAUDE.md` — tracks LLM configuration via `ChatService.get_eval_metadata()`
- Entity Models: `/src/history_book/data_models/entities.py` — `ChatMessage` for history formatting
