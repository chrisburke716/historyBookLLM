# CLAUDE.md - Service Layer

Business logic layer for the History Book RAG application.

## Quick Commands

```bash
# Start backend
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# Run ingestion
poetry run python scripts/run_ingestion.py

# Run evaluations
poetry run python scripts/run_evals.py

# KG pipeline
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py chapter --book 3 --chapter 4
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py book --book 3
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py volume
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py list
```

## Services Overview

### ChatService (`chat_service.py`)

Session management and LangGraph agent orchestration.

**Key Methods**:
- `create_session(title)` → `ChatSession`
- `send_message(session_id, user_message)` → `ChatResult(message, retrieved_paragraphs, metadata)`
- `send_message_stream(session_id, user_message)` → `(AsyncIterator[str], list[Paragraph])`
- `get_session_messages(session_id)` → `list[ChatMessage]`
- `get_eval_metadata()` → dict

**Flow**:
1. Save user message to Weaviate
2. Load chat history, convert to LangChain messages
3. `agent.ainvoke(messages, context=AgentContext(...), config={"thread_id": session_id})`
4. Save AI response + update session timestamp
5. Regenerate session title (async, via `create_title_generation_chain`)

**Memory strategy**:
- `MemorySaver` (LangGraph): fast ephemeral state during graph execution
- Weaviate: durable session + message storage across restarts

**ChatResult**:
```python
@dataclass
class ChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
    metadata: dict | None = None
```

**Tuning**:
```python
ChatService(
    min_context_results=10,
    max_context_results=50,
    context_similarity_cutoff=0.5,
)
```

**Integration**: Called by `api/routes/chat.py` → invokes agent from `services/agents/` → uses `BookRepositoryManager`.

---

### IngestionService (`ingestion_service.py`)

PDF processing pipeline: extract, chunk, store in Weaviate.

**Key Methods**:
- `ingest_pdf(pdf_path, book_title, author, clear_existing)` → stats dict
- `clear_all_data()` — delete all books/chapters/paragraphs
- `check_existing_data()` — get counts

**Pipeline**:
```
PDF → PyMuPDF extraction → Text cleaning → Chapter detection →
Paragraph chunking → Entity creation → Batch storage → Weaviate (auto-embeddings)
```

---

### KGIngestionService (`kg_ingestion_service.py`)

Multi-stage knowledge graph extraction and merge pipeline.

See root `CLAUDE.md` for CLI usage and pipeline details.

---

### KGService (`kg_service.py`)

Read-only KG queries for the KG Explorer frontend.

**Key Methods**:
- `list_graphs()` → `list[KGGraph]`
- `get_graph(graph_name)` → all nodes + links; builds & caches `nx.MultiDiGraph`
- `get_subgraph(entity_id, hops, graph_name)` → N-hop ego subgraph
- `get_entity(entity_id)` → `EntityDetail` with relationship summaries
- `search(query, graph_name, entity_types, limit)` → hybrid entity search

---

### ParagraphService (`paragraph_service.py`)

High-level paragraph operations.

**Note**: `ChatService` uses `BookRepositoryManager.paragraphs` directly (via agent tools), not this service.

---

## Architecture Notes

**Async patterns**: All I/O uses `async def`. Repository calls are synchronous Weaviate I/O (wrapped as needed).

**LangSmith tracing**: `@traceable` on `ChatService.send_message()` + automatic LangGraph tracing.

**Context flow**: `LLMConfig` + `BookRepositoryManager` built once in `ChatService.__init__`, passed to agent via `AgentContext` on each invocation.

## Related Files

- Agent system: `services/agents/CLAUDE.md`
- API layer: `api/CLAUDE.md`
- Database layer: `database/CLAUDE.md`
- LLM config: `llm/CLAUDE.md`
- Evaluations: `evals/CLAUDE.md`
