# Architecture Documentation

## System Overview

The History Book application follows a clean architecture pattern with clear separation of concerns across multiple layers. The system is designed for scalability, maintainability, and testability.

## Layer Architecture

### 1. Service Layer (`src/history_book/services/`)

**Purpose**: Orchestrates business logic and coordinates between repositories.

- `IngestionService`: Main coordinator for the book ingestion pipeline
  - Processes PDF documents
  - Extracts hierarchical structure (books → chapters → paragraphs)  
  - Coordinates with multiple repositories for data persistence
  - Handles text processing and vector generation

- `ParagraphService`: Specialized service for paragraph operations
  - Provides high-level paragraph query methods
  - Abstracts complex repository interactions

### 2. Repository Layer (`src/history_book/database/repositories/`)

**Purpose**: Abstracts data access and provides consistent interfaces.

- `WeaviateRepository<T>`: Generic base repository for Weaviate operations
  - Type-safe CRUD operations
  - Vector similarity search
  - Schema management
  - Error handling and connection management

- `BookRepositoryManager`: Aggregates specialized repositories
  - `book_repository`: Book entity operations
  - `chapter_repository`: Chapter entity operations  
  - `paragraph_repository`: Paragraph entity operations
  - Provides unified access point for all data operations

### 3. Entity Layer (`src/history_book/entities/`)

**Purpose**: Pure data models without business logic or database dependencies.

- `Book`: Represents a complete historical document
- `Chapter`: Represents a section within a book
- `Paragraph`: Represents a text chunk with vector embeddings

### 4. Configuration Layer (`src/history_book/database/config/`)

**Purpose**: Environment-specific database configuration.

- `WeaviateConfig`: Centralized configuration management
  - Connection parameters
  - Environment-specific settings
  - Validation and defaults

### 5. Interface Layer (`src/history_book/database/interfaces/`)

**Purpose**: Contracts and abstractions for loose coupling.

- `RepositoryInterface<T>`: Generic repository contract
- `VectorRepositoryInterface<T>`: Vector-specific operations contract

## Data Flow

### Ingestion Pipeline

```
PDF Input → Text Processing → Entity Creation → Repository Storage → Vector Indexing
```

1. **PDF Processing**: Extract text and structure from PDF documents
2. **Text Normalization**: Clean and standardize text content
3. **Entity Construction**: Create Book, Chapter, and Paragraph entities
4. **Repository Persistence**: Store entities via appropriate repositories
5. **Vector Generation**: Generate embeddings for semantic search

### Query Pipeline

```
Search Query → Repository Interface → Weaviate Vector Search → Entity Mapping → Results
```

1. **Query Input**: User provides search terms or vectors
2. **Repository Routing**: Request directed to appropriate repository
3. **Vector Search**: Weaviate performs similarity search
4. **Result Mapping**: Raw results mapped to entity objects
5. **Response**: Type-safe entities returned to caller

## Design Patterns

### Repository Pattern
- Encapsulates data access logic
- Provides consistent interface across different storage mechanisms
- Enables easy testing through interface mocking

### Service Layer Pattern
- Coordinates business operations across multiple repositories
- Maintains transaction boundaries
- Provides coarse-grained API for external consumers

### Dependency Injection
- Configuration injected into repositories and services
- Enables environment-specific behavior
- Facilitates testing with mock dependencies

### Generic Programming
- Type-safe repository implementations
- Reusable patterns across different entity types
- Compile-time type checking

## Error Handling

### Database Exceptions (`src/history_book/database/exceptions/`)
- `DatabaseConnectionError`: Connection failures
- `CollectionNotFoundError`: Missing collections
- `InvalidSchemaError`: Schema validation failures

### Error Propagation
- Repository layer catches low-level exceptions
- Service layer handles business logic errors
- Clear error messages propagated to callers

## Configuration Management

### Environment Support
- **Development**: Local Weaviate, debug logging, verbose output
- **Test**: Isolated collections, temporary data, minimal logging  
- **Production**: Cloud Weaviate, optimized settings, monitoring

### Configuration Sources
1. Environment variables (highest priority)
2. Configuration files
3. Application defaults (lowest priority)

## Performance Considerations

### Vector Operations
- Batch processing for bulk operations
- Async support for I/O bound operations
- Connection pooling and reuse

### Memory Management
- Streaming PDF processing for large documents
- Lazy loading of vector embeddings
- Efficient text chunk processing

### Scalability
- Horizontal scaling through Weaviate clustering
- Stateless service design
- Configuration-driven resource allocation

## Testing Strategy

### Unit Testing
- Repository interfaces enable easy mocking
- Pure entity objects require no special testing infrastructure
- Service layer tests focus on business logic

### Integration Testing
- Test environment provides isolated Weaviate instance
- End-to-end pipeline testing
- Performance and load testing

## Future Extensibility

### Additional Vector Stores
- Repository pattern enables multiple backend support
- Interface-based design allows storage engine swapping
- Configuration-driven provider selection

### Enhanced Processing
- Pluggable text processing pipeline
- Custom entity types through generic repositories
- Extensible metadata and annotations

### API Layer
- Service layer provides foundation for REST/GraphQL APIs
- Clean separation enables multiple presentation layers
- Standardized error handling and response formats
