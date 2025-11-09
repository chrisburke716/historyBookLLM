# CLAUDE.md - Service Layer

Business logic layer for the History Book RAG application. Services orchestrate repositories, LLM calls, and complex workflows.

## Quick Commands

```bash
# Start backend (uses all services)
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# Test chat interactively
jupyter notebook notebooks/chat_service_demo.ipynb

# Run ingestion
poetry run python scripts/run_ingestion.py

# Run evaluations
poetry run python scripts/run_evals.py
```

## Services Overview

### ChatService (`chat_service.py` - 462 lines)

**Purpose**: Chat session management and RAG orchestration.

**Key Methods**:
- `create_session(title)` - Create new chat session
- `send_message(session_id, message)` - Process message, return `ChatResult` with AI response + retrieved paragraphs
- `get_session_history(session_id)` - Get all messages for a session
- `get_eval_metadata()` - Export config for evaluation tracking

**Configuration** (module constants):
```python
CONTEXT_MIN_RESULTS = 5
CONTEXT_MAX_RESULTS = 40
CONTEXT_SIMILARITY_CUTOFF = 0.4
```

**ChatResult Dataclass**:
```python
@dataclass
class ChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]  # For evaluation and citations
```

**Tuning Retrieval**:
```python
chat_service = ChatService(
    min_context_results=10,
    max_context_results=50,
    context_similarity_cutoff=0.5  # Higher = stricter matching
)
```

**Integration**: Called by API layer → calls RagService → uses ChatMessage/ChatSession repositories.

---

### RagService (`rag_service.py` - 405 lines)

**Purpose**: Retrieval-Augmented Generation using LangChain LCEL chains.

**Key Methods**:
- `generate_response(query, chat_history, min_results, max_results, similarity_cutoff)` - Main RAG execution, returns `RAGResult`

**RAGResult**:
```python
class RAGResult(NamedTuple):
    response: str
    source_paragraphs: list[Paragraph]
```

**LCEL Chain Structure**:
```python
# RAG chain (uses context + history)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{context}\n\nQuestion: {query}"),
])
rag_chain = rag_prompt | chat_model | StrOutputParser()
```

**Supported LLM Providers**:
- OpenAI: `ChatOpenAI` (GPT-4, GPT-3.5, etc.)
- Anthropic: `ChatAnthropic` (Claude models)

**Retrieval**: Uses `ParagraphRepository.hybrid_search()` (vector + BM25).

**Context Formatting**: `format_context_for_llm()` from `llm.utils` formats paragraphs with book/chapter/page metadata.

**Integration**: Called by ChatService → uses LLMConfig and ParagraphRepository.

---

### IngestionService (`ingestion_service.py` - 627 lines)

**Purpose**: PDF processing pipeline - extract, chunk, and store in vector DB.

**Key Methods**:
- `ingest_pdf(pdf_path, book_title, author, clear_existing)` - Main ingestion entry point
- `clear_all_data()` - Delete all books/chapters/paragraphs
- `check_existing_data()` - Get counts of existing entities

**Pipeline Flow**:
```
PDF → PyMuPDF extraction → Text cleaning → Chapter detection →
Paragraph chunking → Entity creation → Batch storage → Weaviate (auto-generates embeddings)
```

**Example**:
```python
from pathlib import Path

service = IngestionService()
result = service.ingest_pdf(
    pdf_path=Path("data/book.pdf"),
    book_title="History Book",
    author="Author Name",
    clear_existing=True  # Fresh start
)
# Returns: {"book_id": "uuid", "chapters_created": 15, "paragraphs_created": 342, ...}
```

**Integration**: Used by `scripts/run_ingestion.py` → uses BookRepositoryManager and text_processing module.

---

### ParagraphService (`paragraph_service.py` - 220 lines)

**Purpose**: High-level paragraph operations and vector search.

**Key Methods**:
- `create_paragraph(paragraph)` - Create paragraph and fetch generated embedding
- `search_similar_paragraphs(query_text, limit, book_index, threshold)` - Vector search with business logic

**Note**: RagService typically uses `ParagraphRepository` directly, not ParagraphService.

---

## Common Tasks

### Tuning RAG Parameters

**Similarity Cutoff**: Lower = more results, less strict matching.
```python
# Recommended: Use notebooks/investigate_vector_search_cutoff.ipynb to analyze
# Then adjust in ChatService initialization
```

**Test Changes**: Run evals before/after tuning.
```bash
poetry run python scripts/run_evals.py
# View results in LangSmith
```

### Changing LLM Provider

```python
from history_book.llm.config import LLMConfig

llm_config = LLMConfig(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.5
)
chat_service = ChatService(llm_config=llm_config)
```

See `/src/history_book/llm/` for details.

### Modifying Prompts

Edit `RagService._build_rag_chain()`:
```python
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Custom system message"),
    MessagesPlaceholder("chat_history"),
    ("human", "Context: {context}\n\nQuestion: {query}"),
])
```

### Adding New PDF

```bash
poetry run python scripts/run_ingestion.py
# Or programmatically via IngestionService.ingest_pdf()
```

---

## Agent Services (LangGraph-based)

### GraphRagService (`graph_rag_service.py` - 308 lines)

**Purpose**: Executes RAG using a LangGraph state machine with retrieve → generate nodes.

**Key Features**:
- **Graph Structure**: Simple RAG graph: `START → retrieve → generate → END`
- **Checkpointing**: Uses MemorySaver to maintain state across executions
- **Streaming**: Token-by-token output via `stream_mode="messages"`
- **Tracing**: Automatic LangSmith tracing with tags `["agent", "langgraph", "simple_rag"]`
- **Composition**: Reuses RagService methods (DRY principle)

**Graph State** (`AgentState` TypedDict):
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Chat history
    question: str  # Current user query
    retrieved_paragraphs: list[Paragraph]  # Retrieved context
    generation: str  # Generated response
    session_id: str  # Maps to thread_id for checkpointing
    metadata: dict  # Execution metadata
```

**Graph Nodes**:
1. **retrieve_node**: Fetches relevant paragraphs using ParagraphRepository
2. **generate_node**: Creates response using LLM + context

**Key Methods**:
```python
# Execute graph (non-streaming)
result = await service.invoke(
    question="What is history?",
    messages=[],  # LangChain message history
    session_id="session-123"
)

# Stream graph execution (token-by-token)
async for chunk in service.stream(
    question="What is history?",
    messages=[],
    session_id="session-123"
):
    print(chunk)  # Token chunks
```

**Comparison with RagService**:
| Feature | RagService (LCEL) | GraphRagService (LangGraph) |
|---------|-------------------|----------------------------|
| Execution | LCEL chains | LangGraph state machine |
| Checkpointing | None | MemorySaver (in-memory) |
| Visualization | None | Mermaid diagrams |
| Tracing | LangSmith (@traceable) | Automatic (built-in) |
| Extensibility | Limited | Easy (add nodes/edges) |
| Performance | 9.50s avg | 8.97s avg (5.6% faster) |

**When to Use**:
- Multi-step reasoning (future)
- Tool calling (future)
- Need checkpointing or graph visualization
- Want better performance

---

### GraphChatService (`graph_chat_service.py` - 381 lines)

**Purpose**: Orchestrates chat sessions using GraphRagService, with hybrid memory strategy.

**Hybrid Memory Strategy**:
- **LangGraph MemorySaver**: In-memory state during graph execution (keyed by session_id)
- **Weaviate**: Long-term persistence of sessions and messages

**Key Methods**:
```python
# Send message
result = await service.send_message(
    session_id="session-123",
    user_message="Tell me about Julius Caesar"
)
# Returns: GraphChatResult(message, retrieved_paragraphs)

# Session management
session = await service.create_session(title="My Chat")
sessions = await service.list_recent_sessions(limit=10)
await service.delete_session(session_id)
```

**GraphChatResult**:
```python
@dataclass
class GraphChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
```

**Data Flow**:
1. Save user message to Weaviate
2. Load chat history from Weaviate
3. Execute graph with history (MemorySaver maintains state)
4. Save AI response to Weaviate
5. Return result

**Integration**: Called by Agent API layer → calls GraphRagService → uses ChatMessage/ChatSession repositories.

**Checkpointing Benefits**:
- Maintains conversation context automatically
- Supports multi-turn dialogues (e.g., "Who was he?" after asking about Julius Caesar)
- State isolated per session_id (maps to thread_id)

**Future Extensibility**:
The graph architecture supports easy addition of:
- **Tool calling**: Add tools_node for search, calculations, APIs
- **Planning**: Add plan_node for complex query decomposition
- **Reflection**: Add reflect_node for self-critique
- **Adaptive RAG**: Add routing nodes for dynamic strategy selection

Example - Adding a tool node:
```python
def tools_node(state: AgentState) -> dict:
    """Execute requested tools"""
    # Tool execution logic
    return {"tool_results": results}

# Add to graph
workflow.add_node("tools", tools_node)
workflow.add_edge("retrieve", "tools")
workflow.add_edge("tools", "generate")
```

**When to Use Agent vs Chat Services**:

**Use ChatService (LCEL) when**:
- Simple RAG queries without conversation context
- Existing integrations depend on it
- Don't need checkpointing or graph visualization

**Use GraphChatService (LangGraph) when**:
- Multi-turn conversations requiring context
- Need to visualize execution flow
- Want LangSmith tracing with graph structure
- Planning to add tools or multi-step reasoning
- Better performance is desired (5.6% faster)

---

## Architecture Notes

**Clean Architecture Pattern**:
```
API Layer → Service Layer → Repository Layer → Database
```

**Services are stateless**: State stored in database via repositories.

**Async patterns**: All I/O operations use `async def` for performance.

**LangSmith tracing**: `@traceable` decorator on `send_message()` and automatic LCEL chain tracing.

---

## Related Files

- API Layer: `/src/history_book/api/` - REST endpoints
- Database Layer: `/src/history_book/database/` - Repositories
- LLM Config: `/src/history_book/llm/` - LLM providers and configuration
- Evaluations: `/src/history_book/evals/` - Quality measurement
- Entity Models: `/src/history_book/data_models/entities.py` - Data classes
