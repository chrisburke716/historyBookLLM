# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (Python)
```bash
# Code quality and linting
poetry run ruff check
poetry run ruff format

# Run tests
poetry run python test_api.py
poetry run python test_full_integration.py

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
- `ChatService`: Manages chat sessions and coordinates RAG responses
- `RagService`: Direct LangChain integration with LCEL chains (PromptTemplate | ChatModel | OutputParser)
- `ParagraphService`: High-level paragraph query operations

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

**Chat Pipeline**:
```
User Message → ChatService → RagService → [Retrieval → LCEL Chain → LLM] → AI Response
```

### Frontend Architecture

- **React 19** with TypeScript and Material-UI
- Components: `MessageInput`, `MessageList`, `SessionDropdown`, `App`
- API client with Axios for backend communication
- State management through React hooks

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
5. **Access**: Chat interface at http://localhost:3000, API docs at http://localhost:8000/docs

## Code Style and Quality

- **Python**: Uses ruff for linting and formatting with strict configuration
- **TypeScript**: Create React App defaults with Material-UI patterns
- **Pre-commit hooks**: Configured for ruff linter and formatter
- Always run `poetry run ruff check` and `poetry run ruff format` before committing

## Testing

- **Backend API Tests**: `test_api.py` - API endpoint testing
- **Integration Tests**: `test_full_integration.py` - End-to-end testing (requires both servers)
- **Frontend Tests**: React Testing Library setup in frontend

## Key Libraries and Dependencies

**Backend**:
- FastAPI for REST API with automatic OpenAPI docs
- Weaviate for vector database operations
- LangChain for RAG implementation with direct LCEL chains
- PyMuPDF for PDF text extraction

**Frontend**:
- React 19 with TypeScript
- Material-UI for component library
- Axios for HTTP client

## Important Notes

- The system has moved away from complex abstraction layers to use LangChain LCEL directly in `RagService`
- Repository pattern provides clean separation between business logic and data access
- Environment-specific configurations handle different deployment scenarios
- Chat functionality requires OpenAI or Anthropic API keys
- Vector embeddings are generated during PDF ingestion and stored in Weaviate