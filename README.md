# History Book Vector Database

A sophisticated Python application for ingesting, storing, and querying a specific history book using Weaviate vector database. The system processes PDF documents into a structured hierarchy (Books → Chapters → Paragraphs) with vector embeddings for semantic search.

## Features

- **PDF Processing**: Extract structured content from PDF historical documents
- **Vector Storage**: Store books, chapters, and paragraphs with semantic embeddings in Weaviate
- **Repository Pattern**: Clean separation of concerns with repository and service layers
- **Environment Management**: Flexible configuration for development, testing, and production
- **Text Processing**: Advanced text cleaning and normalization utilities

## Architecture

### Core Components

```
src/history_book/
├── services/           # Business logic layer
│   ├── ingestion_service.py    # Main ingestion orchestration
│   └── paragraph_service.py    # Paragraph query operations
├── database/           # Data persistence layer
│   ├── repositories/   # Data access abstractions
│   ├── config/        # Database configuration
│   ├── collections.py # Weaviate collection definitions
│   └── interfaces/    # Repository contracts
├── entities/          # Pure data models
├── text_processing/   # Text normalization utilities
└── utils/            # Common utilities
```

### Data Model

- **Book**: Top-level document with title, total pages
- **Chapter**: Document sections with title, page range
- **Paragraph**: Text chunks with content, embeddings, and metadata

## Quick Start

### Prerequisites

- Python 3.11+
- Weaviate database (local Docker or cloud instance)
- Poetry for dependency management

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up your environment:
   ```bash
   # For development
   poetry run python scripts/setup_development_config.py
   
   # For testing
   poetry run python scripts/setup_test_config.py
   ```

### Usage

#### Ingest a Book

```bash
poetry run python scripts/run_ingestion.py
```

This will process the default PDF and ingest it into your configured Weaviate instance.

#### Query the Database

```python
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

# Initialize
config = WeaviateConfig.from_env()
manager = BookRepositoryManager(config)

# Search paragraphs
results = manager.paragraph_repository.similarity_search(
    query="ancient civilizations",
    limit=5
)
```

## Scripts

- `scripts/run_ingestion.py`: Main ingestion pipeline
- `scripts/setup_development_config.py`: Configure for local development
- `scripts/setup_test_config.py`: Configure for testing
- `scripts/inspect_and_clear_database.py`: Database utilities

## Development

### Environment Variables

Create `.env` file with:
```bash
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-api-key  # Optional for local
ENVIRONMENT=development
```

### Running Tests

```bash
poetry run python -m pytest
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

1. Ensure all tests pass
2. Run code formatting: `poetry run ruff format`
3. Check for issues: `poetry run ruff check`
4. Update documentation for significant changes

## License

See LICENSE file for details.