# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (Python)
```bash
# Code quality and linting
poetry run ruff check
poetry run ruff format

# Verify functionality
poetry run python scripts/verify/verify_api.py
poetry run python scripts/verify/verify_integration.py

# Start backend server
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# Database and ingestion
poetry run python scripts/run_ingestion.py
poetry run python scripts/setup_development_config.py
poetry run python scripts/setup_test_config.py

# Knowledge graph extraction
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py chapter --book 3 --chapter 4
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py book --book 3
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py volume        # all books
PYTHONPATH=src poetry run python scripts/run_kg_extraction.py list
```

### Frontend (React TypeScript)
```bash
# Development server
cd frontend && npm start

# Build and test
cd frontend && npm run build
cd frontend && npm test

# Type checking
cd frontend && npx tsc --noEmit
```

## Architecture Overview

This is a full-stack RAG-powered chat application with a Python FastAPI backend and React TypeScript frontend.

### Backend Architecture (Clean Architecture Pattern)

**Service Layer** (`src/history_book/services/`):
- `ChatService`: Session management + LangGraph agent orchestration for RAG responses
- `IngestionService`: Orchestrates PDF processing and data storage
- `ParagraphService`: High-level paragraph query operations
- `KGIngestionService`: Knowledge graph extraction + multi-level merge pipeline
- `KGService`: Read-only KG queries — full graph, N-hop subgraphs (NetworkX ego_graph), entity detail, hybrid search; caches `nx.MultiDiGraph` per graph_name

**Chain Layer** (`src/history_book/chains/`):
- LCEL chain factories for KG operations: extraction, merge (LLM + rule-filter), temporal parsing
- See `/src/history_book/chains/CLAUDE.md` for details

**Repository Layer** (`src/history_book/database/repositories/`):
- `WeaviateRepository<T>`: Generic base repository with type-safe CRUD operations
- `BookRepositoryManager`: Aggregates all repositories (books, chapters, paragraphs, KG collections)
- KG repositories: `KGEntityRepository`, `KGRelationshipRepository`, `KGGraphRepository`, `KGMergeDecisionRepository`

**Entity Layer** (`src/history_book/data_models/entities.py` + `kg_entities.py`):
- Book models: `Book`, `Chapter`, `Paragraph`, `ChatSession`, `ChatMessage`
- KG models: `KGEntity`, `KGRelationship`, `KGGraph`, `KGMergeDecision`, `NormalizedEntity`

### Key Data Flow

**Ingestion Pipeline**:
```
PDF Input → Text Processing → Entity Creation → Repository Storage → Vector Indexing
```

**Chat Pipeline** (LangGraph):
```
User Message → ChatService → agent.ainvoke → [agent_node ↔ tools_node loop] → AI Response
                                               ↓ MemorySaver checkpointing
```

**KG Pipeline**:
```
Chapter paragraphs → LLM extraction → Rule merge (name/alias, LLM-filtered)
                   → Embed new entities → Similarity candidates → LLM merge
                   → Chapter KGGraph → Cross-chapter merge → Book KGGraph → Cross-book merge → Volume KGGraph
```

### Chat API

Live chat goes through one AG-UI streaming endpoint; everything else is plain JSON session/history CRUD.

**Endpoints**:
- `POST /api/chat/agent` — AG-UI streaming. Accepts `RunAgentInput`, returns SSE event stream (token deltas, tool calls, state snapshots). The only live-chat surface.
- `POST /api/chat/sessions` — create session
- `GET /api/chat/sessions` — list recent sessions
- `GET /api/chat/sessions/{id}/messages` — fetch persisted history
- `DELETE /api/chat/sessions/{id}` — delete session
- `GET /api/chat/sessions/{id}/graph` — Mermaid visualization of the agent graph

**Key features**:
- **Auto-generated titles**: regenerated server-side after each turn (synchronous, via the title chain)
- **Checkpointing**: `MemorySaver` keyed by `thread_id == session_id`. Full message history including tool calls/results survives within a process; Weaviate is the durable store across restarts
- **LangSmith tracing**: every run, regardless of entry point
- **AG-UI protocol**: wire-compatible with CopilotKit; the frontend uses `HttpAgent` from `@ag-ui/client` to consume the stream

See `/src/history_book/api/CLAUDE.md` for endpoint details and a curl example; `/src/history_book/services/agents/CLAUDE.md` for the agent + AG-UI wrapper.

### Frontend Architecture

- **React 19** with TypeScript and Material-UI
- **Three-page interface**: Chat (RAG), Book (browse content), KG Explorer (graph visualization)
- **Chat surface**: custom MUI components driven by CopilotKit hooks (`useCopilotChatInternal`, `useCoAgent`). Key files: `ChatPage`, `CopilotProvider` (sets up `HttpAgent` + provider), `ChatThreadController` (history hydration + run-end signal), `MessageList` (turn-grouped renderer with markdown + tool-call chips + citations), `MessageInput`, `ToolCallCard`, `SessionDropdown`
- **Book components**: `BookSelector`, `ChapterView`, `BookPage`
- **KG components**: `KGTopBar`, `ForceGraphPanel`, `EntityPanel`, `KGPage`
- **State management**: TanStack Query owns server state for chat (sessions, history) and KG; Redux Toolkit for KG Explorer's UI state; CopilotKit owns the live chat message thread
- Axios for the REST surface (sessions, history). Live chat goes through CopilotKit's HttpAgent, not Axios

### Evaluation Framework

**Evaluation Layer** (`src/history_book/evals/`):
- **LLM-based evaluators**: Helpfulness, Hallucination, Factual Accuracy, Coherence, IDK handling, Relevance
- **Function evaluators**: Document count tracking
- **LangSmith integration**: Experiment tracking and comparison
- **Dataset**: 100 evaluation queries (user + synthetic)
- See `/src/history_book/evals/CLAUDE.md` for details

## Detailed Documentation

For in-depth information about specific subsystems, see:

- **[Services](/src/history_book/services/CLAUDE.md)** - Business logic layer (ChatService, IngestionService, KGIngestionService)
- **[Chains](/src/history_book/chains/CLAUDE.md)** - LLM chain factories for KG extraction, merge, temporal parsing
- **[Agent System](/src/history_book/services/agents/CLAUDE.md)** - LangGraph agent: graph, state, context, tools
- **[Database](/src/history_book/database/CLAUDE.md)** - Repository pattern, Weaviate integration, KG repositories
- **[API](/src/history_book/api/CLAUDE.md)** - FastAPI REST endpoints
- **[LLM Configuration](/src/history_book/llm/CLAUDE.md)** - LLM provider setup (OpenAI, Anthropic)
- **[Evaluations](/src/history_book/evals/CLAUDE.md)** - RAG evaluation framework
- **[Frontend](/frontend/CLAUDE.md)** - React TypeScript UI

## Environment Configuration

The application supports multiple environments through configuration files:

- **Development**: Local Weaviate instance, verbose logging
- **Test**: Isolated test collections, temporary data
- **Production**: Currently points to development config

Required environment variables:
```bash
WEAVIATE_URL=http://localhost:8080
ENVIRONMENT=development
OPENAI_API_KEY=your-api-key  # Required for chat functionality
```

## Development Workflow

1. **Setup**: Run `poetry run python scripts/setup_development_config.py` (provides Weaviate Docker instructions)
2. **Ingest Data**: Run `poetry run python scripts/run_ingestion.py` to process PDFs into vector database
3. **Backend**: Start API with `PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000`
4. **Frontend**: Start UI with `cd frontend && npm start`
5. **Access**:
   - Chat interface: http://localhost:3000/chat
   - Book browsing: http://localhost:3000/book
   - KG Explorer: http://localhost:3000/kg
   - API docs: http://localhost:8000/docs

## Code Style and Quality

- **Python**: Uses ruff for linting and formatting with strict configuration
- **TypeScript**: Create React App defaults with Material-UI patterns
- **Pre-commit hooks**: Configured for ruff linter and formatter
- Always run `poetry run ruff check` and `poetry run ruff format` before committing

## Testing

- **Eval suite**: `poetry run python scripts/run_evals.py` — LangSmith-tracked evaluation against the dataset. Calls `ChatService.send_message` directly (synchronous Python path), not the HTTP API.
- **Frontend tests**: React Testing Library setup in `frontend/`.

## Key Libraries and Dependencies

**Backend**:
- FastAPI for REST API with automatic OpenAPI docs
- Weaviate for vector database operations
- LangChain + LangGraph for the RAG agent (tool-using, stateful, checkpointed)
- `ag-ui-langgraph` for the AG-UI wrapper around the compiled graph
- `ag-ui-protocol` for the AG-UI event types + SSE encoder
- PyMuPDF for PDF text extraction

**Frontend**:
- React 19 with TypeScript
- Material-UI for component library
- `@copilotkit/react-core` for chat hooks (custom MUI on top, no prebuilt UI)
- `@ag-ui/client`'s `HttpAgent` for AG-UI transport
- TanStack Query for server state (chat session list/history + KG data)
- Redux Toolkit + react-redux (KG Explorer UI state)
- react-markdown + remark-gfm for assistant message rendering
- Axios for the REST surface
- react-force-graph-2d (KG visualization)
- NetworkX (backend, subgraph computation via `nx.MultiDiGraph` + `ego_graph`)

## Project Documentation

**Planning Files**: The `/docs/plans/` directory contains Claude-generated planning documents for major features and documentation efforts. These serve as reference for understanding design decisions and project evolution.

## Important Notes

- `ChatService` orchestrates a LangGraph agent that consumes a LangChain chat model directly via `build_chat_model(LLMConfig)`. Two entry points: `send_message_agui` (HTTP / AG-UI streaming, used by the frontend) and `send_message` (synchronous, used by evals). Both share the same compiled graph + checkpointer.
- The AG-UI integration lives entirely in `services/agents/agui_wrapper.py` + `api/routes/agent.py`. The graph itself doesn't know AG-UI exists.
- Layer-boundary doc lives in `api/CLAUDE.md` — routes are HTTP/transport only; orchestration belongs in services.
- Repository pattern provides clean separation between business logic and data access.
- Chat functionality requires OpenAI or Anthropic API keys.
- Vector embeddings are generated during PDF ingestion and stored in Weaviate.

## Guiding Principles

- DRY - avoid duplicated code to keep things simple and maintainable
- KISS - don't overcomplicate things - avoid premature abstraction
- YAGNI - focus on building what's needed now, not what might be needed later