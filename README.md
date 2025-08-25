# History Book Chat Application

A full-stack RAG-powered chat application for conversational interactions with historical documents. The system combines a Python FastAPI backend with vector search capabilities and a React TypeScript frontend for an intuitive chat experience.

## Features

- **🤖 RAG-Powered Chat**: Conversational interface with retrieval-augmented generation
- **📚 Real Citations**: Responses include actual page numbers from source documents  
- **💬 Session Management**: Create, switch between, and persist conversation sessions
- **🎨 Modern UI**: Clean Material-UI interface with TypeScript and accessibility
- **📡 REST API**: FastAPI backend with automatic OpenAPI documentation
- **🔍 Vector Search**: Semantic search using Weaviate vector database
- **⚡ Real-time**: Live responses with loading states and error handling

## Architecture

### Full-Stack Architecture

```
├── Backend (Python FastAPI)
│   ├── src/history_book/
│   │   ├── api/              # FastAPI routes and models
│   │   │   ├── main.py       # FastAPI application
│   │   │   ├── routes/       # API endpoints (chat, sessions)
│   │   │   └── models/       # Pydantic request/response models
│   │   ├── services/         # Business logic (chat, ingestion)
│   │   ├── database/         # Weaviate integration & repositories
│   │   ├── llm/             # LangChain LLM providers
│   │   └── data_models/     # Entity definitions
│
├── Frontend (React + TypeScript)
│   ├── frontend/src/
│   │   ├── components/       # Chat UI components
│   │   ├── pages/           # Main chat page
│   │   ├── services/        # API client (Axios)
│   │   ├── hooks/           # React state management
│   │   └── types/           # TypeScript interfaces
│
└── Tests
    ├── test_api.py          # Backend API test suite
    └── test_full_integration.py  # End-to-end tests
```

### Data Model

- **Book**: Top-level documents with title, page range
- **Chapter**: Document sections with title, page range
- **Paragraph**: Text chunks with content, embeddings, and metadata

## Quick Start

### Prerequisites

- **Backend**: Python 3.11+, Poetry, Weaviate database (Docker)
- **Frontend**: Node.js 16+, npm
- **LLM**: OpenAI API key (optional, uses Mock provider by default)

### Installation & Setup

1. **Clone and install backend dependencies:**
   ```bash
   poetry install
   ```

2. **Set up your environment:**
   ```bash
   # For development
   poetry run python scripts/setup_development_config.py
   
   # For testing
   poetry run python scripts/setup_test_config.py
   ```
   Note: These scripts provide instructions for running Weaviate through Docker.

3. **Ingest historical documents:**
   ```bash
   poetry run python scripts/run_ingestion.py
   ```
   This processes the PDF and ingests it into your Weaviate instance with vector embeddings.

4. **Install frontend dependencies:**
   ```bash
   cd frontend && npm install
   ```

### Running the Chat Application

**Start both servers (in separate terminals):**

1. **Backend API:**
   ```bash
   PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000
   ```

2. **Frontend UI:**
   ```bash
   cd frontend && npm start
   ```

**Access the application:**
- **💬 Chat Interface**: http://localhost:3000
- **📚 API Documentation**: http://localhost:8000/docs  
- **🔍 Health Check**: http://localhost:8000

### Direct Database Usage

You can also query the database directly:

```python
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

# Initialize
config = WeaviateConfig.from_environment()
manager = BookRepositoryManager(config)

# Search paragraphs
results = manager.paragraphs.similarity_search_by_text(
    query_text="causes of World War I",
    limit=5
)
```

### Testing

```bash
# Test backend API endpoints
poetry run python test_api.py

# Test full-stack integration (requires both servers running)
poetry run python test_full_integration.py
```

## Scripts

- `scripts/run_ingestion.py`: Main ingestion pipeline
- `scripts/setup_development_config.py`: Configure for local development
- `scripts/setup_test_config.py`: Configure for testing
- `scripts/switch_environment.py`: Switch between dev/test environments
- `scripts/inspect_and_clear_database.py`: Database utilities - inspect and clear items in collection
- `scripts/manage_collections.py`: Delete and/or create Weaviate collections

## Development

### Environment Variables

Create `.env` file with:
```bash
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-api-key  # Optional for local
ENVIRONMENT=development
```

### Code Quality

```bash
poetry run ruff check
poetry run ruff format
```

## Configuration

The system supports multiple environments through configuration files:

- **Development**: Local Weaviate instance, verbose logging
- **Test**: Isolated test collections, temporary data
- **Production**: Currently points to development config

## Contributing

1. Ensure all tests pass (once they exist)
2. Run code formatting: `poetry run ruff format`
3. Check for issues: `poetry run ruff check`
4. Update documentation for significant changes

## License

See LICENSE file for details.