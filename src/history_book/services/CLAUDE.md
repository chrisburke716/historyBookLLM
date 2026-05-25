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

Session management and agent orchestration. Two entry points for running
the agent serve different consumers:

**Key methods**:
- `create_session(title)` → `ChatSession`
- `get_session(session_id)`, `list_recent_sessions(limit)`, `delete_session(session_id)`, `get_session_messages(session_id)` — session/history CRUD
- `send_message_agui(input_data)` → `AsyncIterator[ProcessedEvents]` — primary chat path. Owns the full turn lifecycle: pre-persists the user message, streams AG-UI events from the wrapped graph, persists the assistant message and regenerates the title in a `finally`. Used by `routes/agent.py`.
- `send_message(session_id, user_message)` → `ChatResult` — synchronous one-shot used by `scripts/run_evals.py`. Not exposed via HTTP. Same persistence semantics as the AG-UI path.
- `save_user_message`, `save_ai_message`, `maybe_regenerate_title`, `build_context`, `agent_config` — public helpers shared by both entry points.
- `get_eval_metadata()` → dict — eval surfacing of model + retrieval config.

**Lifecycle (both paths)**:
1. Save user message to Weaviate.
2. Run the graph (AG-UI wrapped for `send_message_agui`, direct `ainvoke` for `send_message`).
3. Save AI response, update session timestamp.
4. Regenerate session title (via `create_title_generation_chain`).

**Memory strategy**:
- `MemorySaver` (LangGraph): in-process state with full message history including tool calls/results. Lost on restart.
- Weaviate: durable session + user/assistant message storage. Tool messages are not persisted here.

**ChatResult** (eval path only):
```python
@dataclass
class ChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
    metadata: dict | None = None
```

**Init / tuning**:
```python
ChatService(
    min_context_results=10,
    max_context_results=50,
    context_similarity_cutoff=0.5,
    enabled_tools=None,                  # None = all tools; pass list for A/B
)
```

`__init__` instantiates one `KGService` (sharing the repo manager so its NX
cache stays warm across requests) and resolves `volume_graph_name` once via
`_resolve_volume_graph_name()` (volume → largest book → None with warning).
Both flow into `AgentContext` on every invocation.

**Integration**: HTTP via `api/routes/agent.py` (streaming) and `api/routes/chat.py` (session CRUD only); evals via `scripts/run_evals.py`. All paths share the same compiled graph and checkpointer.

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

Read-only KG queries. Used by both the KG Explorer frontend and the chat
agent's KG tools (via `AgentContext.kg_service`).

**Key Methods**:
- `list_graphs()` → `list[KGGraph]`
- `get_graph(graph_name)` → all nodes + links; builds & caches `nx.MultiDiGraph`
- `get_subgraph(entity_id, hops, graph_name)` → N-hop ego subgraph
- `get_entity(entity_id)` → `EntityDetail` with relationship summaries
  (each summary includes `other_entity_occurrence_count` and `page` for
  prominence sorting and citations); also populates `source_paragraph_ids`
- `search(query, graph_name, entity_types, limit)` → hybrid entity search

---

### ParagraphService (`paragraph_service.py`)

High-level paragraph operations.

**Note**: `ChatService` uses `BookRepositoryManager.paragraphs` directly (via agent tools), not this service.

---

## Architecture Notes

**Async patterns**: All I/O uses `async def`. Repository calls are synchronous Weaviate I/O (wrapped as needed).

**LangSmith tracing**: `@traceable` on `ChatService.send_message()` (eval path) + automatic LangGraph tracing on every run regardless of entry point.

**Context flow**: `LLMConfig` + `BookRepositoryManager` built once in `ChatService.__init__`, passed to agent via `AgentContext` on each invocation.

## Related Files

- Agent system: `services/agents/CLAUDE.md`
- API layer: `api/CLAUDE.md`
- Database layer: `database/CLAUDE.md`
- LLM config: `llm/CLAUDE.md`
- Evaluations: `evals/CLAUDE.md`
