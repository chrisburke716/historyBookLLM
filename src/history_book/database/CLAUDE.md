# CLAUDE.md - Database Layer

Repository pattern implementation for clean data access. Uses Weaviate vector database with type-safe generic repositories.

## Quick Commands

```bash
# Start Weaviate (Docker)
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.28.3

# Setup development environment
poetry run python scripts/setup_development_config.py

# Inspect database
poetry run python scripts/inspect_and_clear_database.py

# Manage collections
poetry run python scripts/manage_collections.py
```

## Directory Structure

```
database/
├── repositories/           # Repository implementations
│   ├── weaviate_repository.py    # Generic base (645 lines)
│   └── book_repository.py        # Specialized repos (286 lines)
├── interfaces/            # Abstract contracts
│   ├── repository_interface.py
│   └── vector_repository_interface.py
├── config/               # Configuration
│   └── database_config.py
├── exceptions/           # Custom errors
│   └── database_exceptions.py
├── server.py            # Weaviate client singleton
└── collections.py       # Schema generation from Pydantic
```

## Core Components

### WeaviateRepository<T> (Generic Base)

**File**: `repositories/weaviate_repository.py`

**Purpose**: Type-safe generic repository with CRUD + vector search.

**Generic Type Pattern**:
```python
from history_book.database.repositories import WeaviateRepository
from history_book.data_models.entities import Book

class BookRepository(WeaviateRepository[Book]):
    def __init__(self, config: WeaviateConfig):
        super().__init__(
            config=config,
            collection_name="Books",
            entity_class=Book,
        )
```

**Key Methods**:
- `create(entity)` - Create single entity, returns ID
- `get_by_id(id)` - Retrieve by ID
- `update(entity)` - Update existing entity
- `delete(id)` - Delete by ID
- `list(limit, offset)` - Paginated list
- `list_all()` - Get all entities
- `count()` - Count entities
- `find_by_criteria(criteria)` - Filter by field values
- `batch_create(entities)` - Bulk insert
- `similarity_search_by_text(query_text, limit, filters)` - Vector search
- `hybrid_search(query_text, limit, alpha)` - Vector + BM25 combined
- `get_vector(id)` - Get vector embedding for entity

**Auto-creates collections**: If collection doesn't exist, creates it from Pydantic schema.

### Specialized Repositories

**File**: `repositories/book_repository.py`

All inherit from `WeaviateRepository[T]`:

#### BookRepository
```python
class BookRepository(WeaviateRepository[Book]):
    collection_name = "Books"
```

#### ChapterRepository
```python
class ChapterRepository(WeaviateRepository[Chapter]):
    collection_name = "Chapters"

    # Additional method
    def find_by_book_index(self, book_index: int) -> list[Chapter]
```

#### ParagraphRepository
```python
class ParagraphRepository(WeaviateRepository[Paragraph]):
    collection_name = "Paragraphs"

    # Additional methods
    def find_by_book_index(self, book_index: int) -> list[Paragraph]
    def find_by_chapter_index(self, book_index: int, chapter_index: int) -> list[Paragraph]
    def search_similar_paragraphs(query_text, limit, book_index, threshold) -> list[tuple[Paragraph, float]]
```

#### ChatSessionRepository
```python
class ChatSessionRepository(WeaviateRepository[ChatSession]):
    collection_name = "ChatSessions"
```

#### ChatMessageRepository
```python
class ChatMessageRepository(WeaviateRepository[ChatMessage]):
    collection_name = "ChatMessages"

    # Additional method
    def find_by_session_id(self, session_id: str) -> list[ChatMessage]
```

### BookRepositoryManager

**File**: `repositories/book_repository.py`

**Purpose**: Aggregates all specialized repositories for convenient access.

```python
class BookRepositoryManager:
    def __init__(self, config: WeaviateConfig | None = None):
        if config is None:
            config = WeaviateConfig.from_environment()

        self.books = BookRepository(config)
        self.chapters = ChapterRepository(config)
        self.paragraphs = ParagraphRepository(config)
        self.chat_sessions = ChatSessionRepository(config)
        self.chat_messages = ChatMessageRepository(config)
```

**Usage**:
```python
from history_book.database.repositories import BookRepositoryManager

manager = BookRepositoryManager()

# Access any repository
books = manager.books.list(limit=10)
paragraphs = manager.paragraphs.hybrid_search("first civilizations", limit=5)
messages = manager.chat_messages.find_by_session_id("session-uuid")
```

### WeaviateConfig

**File**: `config/database_config.py`

**Purpose**: Environment-based database configuration.

**Environments**:
```python
class DatabaseEnvironment(Enum):
    PRODUCTION = "production"
    TEST = "test"
    DEVELOPMENT = "development"
```

**Configuration**:
```python
@dataclass
class WeaviateConfig:
    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    scheme: str = "http"
    api_key: str | None = None
    openai_api_key: str | None = None  # For Weaviate vectorization
    timeout: int = 30
    environment: DatabaseEnvironment = DatabaseEnvironment.PRODUCTION
```

**From Environment**:
```python
config = WeaviateConfig.from_environment()
# Reads DB_ENVIRONMENT env var: "development", "test", or "production"
```

**Environment Variables**:
```bash
DB_ENVIRONMENT=development  # or test, production
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
OPENAI_APIKEY=sk-...  # For vectorization
```

**Helpers**:
```python
config.is_local  # True if localhost
config.connection_string  # "http://localhost:8080"
```

## Common Tasks

### Adding New Entity Type

1. **Define Entity** (`data_models/entities.py`):
```python
from pydantic import BaseModel

class Author(BaseModel):
    id: str | None = None
    name: str
    birth_year: int
```

2. **Create Repository** (`database/repositories/book_repository.py`):
```python
class AuthorRepository(WeaviateRepository[Author]):
    def __init__(self, config: WeaviateConfig):
        from history_book.data_models.entities import Author
        super().__init__(
            config=config,
            collection_name="Authors",
            entity_class=Author,
        )
```

3. **Add to Manager**:
```python
class BookRepositoryManager:
    def __init__(self, config: WeaviateConfig | None = None):
        # ... existing repos ...
        self.authors = AuthorRepository(config)
```

4. **Collection auto-created**: Schema generated from Pydantic model on first use.

### Vector Search

**Similarity Search** (pure vector):
```python
results = manager.paragraphs.similarity_search_by_text(
    query_text="ancient civilizations",
    limit=10,
    filters={"book_index": 0},  # Optional filtering
    distance_threshold=0.7  # Similarity cutoff
)
# Returns: list[tuple[Paragraph, float]]  # (entity, similarity_score)
```

**Hybrid Search** (vector + BM25):
```python
results = manager.paragraphs.hybrid_search(
    query_text="mesopotamia",
    limit=10,
    alpha=0.75  # 1.0 = pure vector, 0.0 = pure BM25, 0.75 = blend
)
# Returns: list[Paragraph]
```

**Tuning**: Use `/notebooks/investigate_vector_search_cutoff.ipynb` to analyze optimal thresholds and alpha values.

### Batch Operations

**Batch Insert** (much faster than one-by-one):
```python
paragraphs = [Paragraph(...), Paragraph(...), ...]
ids = manager.paragraphs.batch_create(paragraphs)
# Returns: list[str]  # IDs of created entities
```

### Filtering

```python
# Find by single field
chapters = manager.chapters.find_by_criteria({"book_index": 0})

# Find by multiple fields
paragraphs = manager.paragraphs.find_by_criteria({
    "book_index": 0,
    "chapter_index": 1
})
```

### Switching Environments

```bash
# Development (default)
export DB_ENVIRONMENT=development
poetry run python scripts/setup_development_config.py

# Test (isolated collections)
export DB_ENVIRONMENT=test
poetry run python scripts/setup_test_config.py

# Production
export DB_ENVIRONMENT=production
```

### Managing Collections

```bash
# View collections and counts
poetry run python scripts/inspect_and_clear_database.py

# Delete all data
poetry run python scripts/delete_collections.py

# Manual collection management
poetry run python scripts/manage_collections.py
```

## Repository Pattern Benefits

**Abstraction**: Services don't know about Weaviate - easy to swap database.

**Type Safety**: Generic types catch errors at development time.

**Testability**: Mock repositories for unit testing services.

**Consistency**: Same interface across all entity types.

**Reusability**: Base repository handles 90% of operations.

## Vector Embeddings

**Auto-generation**: Weaviate generates embeddings during insert using OpenAI's text-embedding model.

**Retrieval**: Embeddings stored in Weaviate, used for similarity search.

**Access**: `get_vector(id)` returns embedding for an entity.

**Configuration**: Set `OPENAI_APIKEY` environment variable for vectorization.

## Error Handling

**Custom Exceptions** (`exceptions/database_exceptions.py`):
- `ConnectionError` - Database connection issues
- `CollectionError` - Collection operations failed
- `QueryError` - Query execution failed
- `ValidationError` - Entity validation failed
- `VectorError` - Vector operations failed
- `BatchOperationError` - Batch operation failed

**Usage**:
```python
from history_book.database.exceptions import ConnectionError

try:
    entity = repository.get_by_id(id)
except ConnectionError as e:
    logger.error(f"Database connection failed: {e}")
```

## Related Files

- Services: `/src/history_book/services/` - Uses repositories for data access
- Entity Models: `/src/history_book/data_models/entities.py` - Data classes
- Scripts: `/scripts/` - Database setup and management utilities
- Schema Generation: `collections.py` - Auto-generates Weaviate schemas from Pydantic
- Client Singleton: `server.py` - Manages Weaviate client connection
