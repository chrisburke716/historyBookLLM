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
- `IngestionService`: Orchestrates PDF processing and data storage
- `ChatService`: Manages chat sessions and coordinates RAG responses (LCEL-based)
- `RagService`: Direct LangChain integration with LCEL chains (PromptTemplate | ChatModel | OutputParser)
- `ParagraphService`: High-level paragraph query operations
- `GraphRagService`: LangGraph-based RAG with graph execution and checkpointing
- `GraphChatService`: Session orchestration for graph-based chat with MemorySaver

**Repository Layer** (`src/history_book/database/repositories/`):
- `WeaviateRepository<T>`: Generic base repository with type-safe CRUD operations
- `BookRepositoryManager`: Aggregates specialized repositories (books, chapters, paragraphs)
- All repositories implement consistent interfaces for testability

**Entity Layer** (`src/history_book/data_models/entities.py`):
- Pure data models: `Book`, `Chapter`, `Paragraph`, `ChatSession`, `ChatMessage`
- No business logic or database dependencies

### Key Data Flow

**Ingestion Pipeline**:
```
PDF Input → Text Processing → Entity Creation → Repository Storage → Vector Indexing
```

**Chat Pipeline** (LCEL):
```
User Message → ChatService → RagService → [Retrieval → LCEL Chain → LLM] → AI Response
```

**Agent Pipeline** (LangGraph):
```
User Message → GraphChatService → GraphRagService → [Graph: retrieve_node → generate_node] → AI Response
                                                     ↓ (MemorySaver checkpointing)
```

### Agent API (LangGraph-based)

The `/api/agent/*` endpoints provide LangGraph-based chat with enhanced capabilities:

**Key Features**:
- **Auto-generated Titles**: Sessions get descriptive titles based on conversation content
- **Checkpointing**: Maintains conversation context across messages using LangGraph MemorySaver
- **Graph Visualization**: View execution flow via Mermaid diagrams
- **LangSmith Tracing**: Full observability of graph execution, timing, and state transitions
- **Better Performance**: 5.6% faster than LCEL implementation (8.97s vs 9.50s average)
- **Extensibility**: Easy to add tools, planning, reflection, and multi-step reasoning

**Endpoints**:
- `POST /api/agent/sessions` - Create new agent session
- `GET /api/agent/sessions` - List recent sessions
- `POST /api/agent/sessions/{id}/messages` - Send message (non-streaming)
- `GET /api/agent/sessions/{id}/messages` - Get conversation history
- `GET /api/agent/sessions/{id}/graph` - Get graph visualization (Mermaid)
- `DELETE /api/agent/sessions/{id}` - Delete session

**Quick Start**:
```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/agent/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "History Chat"}' | jq -r '.id')

# Send message
curl -X POST http://localhost:8000/api/agent/sessions/$SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "What is the history of Ancient Rome?"}'

# Get graph visualization
curl http://localhost:8000/api/agent/sessions/$SESSION_ID/graph
```

**When to Use Agent vs Chat API**:
- Use **Agent API** for multi-turn conversations, graph visualization, or when planning to add tools
- Use **Chat API** for simple one-off queries or existing integrations

See `/src/history_book/services/agents/CLAUDE.md` for implementation details.

### Frontend Architecture

- **React 19** with TypeScript and Material-UI
- **Dual-page interface**: Chat page (RAG conversations) and Book page (browse/read book content)
- **Chat Components**: `MessageInput`, `MessageList`, `SessionDropdown`, `ChatPage`
- **Book Components**: `BookSelector`, `ChapterView`, `BookPage`
- **Features**: URL-based routing, scroll position persistence, tab navigation
- API client with Axios for backend communication
- State management through React hooks

### Evaluation Framework

**Evaluation Layer** (`src/history_book/evals/`):
- **LLM-based evaluators**: Helpfulness, Hallucination, Factual Accuracy, Coherence, IDK handling, Relevance
- **Function evaluators**: Document count tracking
- **LangSmith integration**: Experiment tracking and comparison
- **Dataset**: 100 evaluation queries (user + synthetic)
- See `/src/history_book/evals/CLAUDE.md` for details

## Detailed Documentation

For in-depth information about specific subsystems, see:

- **[Services](/src/history_book/services/CLAUDE.md)** - Business logic layer (ChatService, RagService, IngestionService, ParagraphService)
- **[Agent System](/src/history_book/services/agents/CLAUDE.md)** - LangGraph-based agent implementation (GraphRagService, GraphChatService)
- **[Database](/src/history_book/database/CLAUDE.md)** - Repository pattern and Weaviate integration
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
   - API docs: http://localhost:8000/docs

## Code Style and Quality

- **Python**: Uses ruff for linting and formatting with strict configuration
- **TypeScript**: Create React App defaults with Material-UI patterns
- **Pre-commit hooks**: Configured for ruff linter and formatter
- Always run `poetry run ruff check` and `poetry run ruff format` before committing

## Testing

- **Backend API Verification**: `scripts/verify/verify_api.py` - API endpoint testing
- **Integration Verification**: `scripts/verify/verify_integration.py` - End-to-end testing (requires both servers)
- **Performance Benchmarks**: `scripts/verify/benchmark_langgraph.py` - LangGraph vs LCEL comparison
- **Frontend Tests**: React Testing Library setup in frontend

## Key Libraries and Dependencies

**Backend**:
- FastAPI for REST API with automatic OpenAPI docs
- Weaviate for vector database operations
- LangChain for RAG implementation with direct LCEL chains
- LangGraph for stateful agent execution with checkpointing
- PyMuPDF for PDF text extraction

**Frontend**:
- React 19 with TypeScript
- Material-UI for component library
- Axios for HTTP client

## Project Documentation

**Planning Files**: The `/docs/plans/` directory contains Claude-generated planning documents for major features and documentation efforts. These serve as reference for understanding design decisions and project evolution.

## Important Notes

- The system has moved away from complex abstraction layers to use LangChain LCEL directly in `RagService`
- Repository pattern provides clean separation between business logic and data access
- Environment-specific configurations handle different deployment scenarios
- Chat functionality requires OpenAI or Anthropic API keys
- Vector embeddings are generated during PDF ingestion and stored in Weaviate

## Guiding Principles

- DRY - avoid duplicated code to keep things simple and maintainable
- KISS - don't overcomplicate things - avoid premature abstraction
- YAGNI - focus on building what's needed now, not what might be needed later