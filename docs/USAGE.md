# Usage Guide

## Installation and Setup

### 1. Install Dependencies

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Set Up Weaviate Database

#### Option A: Local Docker Instance (Recommended for Development)

```bash
# Start Weaviate with Docker
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  -e ENABLE_MODULES='text2vec-openai,text2vec-huggingface,text2vec-transformers' \
  cr.weaviate.io/semitechnologies/weaviate:1.24.0
```

#### Option B: Weaviate Cloud Services (WCS)

1. Sign up at [Weaviate Cloud Services](https://console.weaviate.cloud/)
2. Create a new cluster
3. Note your cluster URL and API key

### 3. Configure Environment

#### For Development:
```bash
poetry run python scripts/setup_development_config.py
```

#### For Testing:
```bash
poetry run python scripts/setup_test_config.py
```

#### Manual Configuration:
Create a `.env` file in the project root:
```bash
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-api-key-here  # Only needed for cloud instances
ENVIRONMENT=development
```

## Basic Usage

### Ingesting Books

#### 1. Prepare Your PDF
Place your PDF file in the `data/` directory or note its path.

#### 2. Run the Ingestion Pipeline
```bash
poetry run python scripts/run_ingestion.py
```

The default script processes `data/penguin_history_6.pdf.pdf`. To process a different file:

```python
# Edit scripts/run_ingestion.py
from pathlib import Path
from history_book.services import IngestionService
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

def main():
    config = WeaviateConfig.from_env()
    manager = BookRepositoryManager(config)
    service = IngestionService(manager)
    
    # Change this path to your PDF
    pdf_path = Path("data/your-book.pdf")
    final_page = 1000  # Adjust as needed
    
    book_ids, chapter_ids, paragraph_ids = service.ingest_book_from_pdf(
        pdf_path=pdf_path,
        final_page=final_page,
        clear_existing=True  # Set to False to append to existing data
    )
    
    print(f"Ingested: {len(book_ids)} books, {len(chapter_ids)} chapters, {len(paragraph_ids)} paragraphs")

if __name__ == "__main__":
    main()
```

### Querying the Database

#### 1. Text-Based Search
```python
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

# Initialize
config = WeaviateConfig.from_env()
manager = BookRepositoryManager(config)

# Search for paragraphs containing specific topics
results = manager.paragraph_repository.search_by_text(
    query="ancient civilizations",
    limit=10
)

for paragraph, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {paragraph.text[:200]}...")
    print(f"Book: {paragraph.book_title}, Chapter: {paragraph.chapter_title}")
    print("---")
```

#### 2. Vector-Based Similarity Search
```python
# If you have a query vector (from embeddings)
query_vector = [0.1, 0.2, -0.3, ...]  # Your embedding vector

results = manager.paragraph_repository.similarity_search(
    query_vector=query_vector,
    limit=5,
    threshold=0.7
)

for paragraph, distance in results:
    print(f"Distance: {distance:.3f}")
    print(f"Text: {paragraph.text[:200]}...")
    print("---")
```

#### 3. Using the Paragraph Service
```python
from history_book.services import ParagraphService

# Higher-level service interface
paragraph_service = ParagraphService(manager)

# Get paragraphs from a specific book and chapter
book_2_chapter_3 = paragraph_service.get_paragraphs_by_chapter(
    book_index=2,
    chapter_index=3
)

for paragraph in book_2_chapter_3:
    print(f"Page {paragraph.page_number}: {paragraph.text[:100]}...")
```

#### 4. Book and Chapter Queries
```python
# Get all books
books = manager.book_repository.get_all()
for book in books:
    print(f"Book: {book.title} ({book.total_pages} pages)")

# Get chapters from first book
if books:
    book_id = books[0].id
    chapters = manager.chapter_repository.get_by_book_id(book_id)
    for chapter in chapters:
        print(f"Chapter: {chapter.title} (pages {chapter.start_page}-{chapter.end_page})")
```

## Advanced Usage

### Custom Ingestion

```python
from pathlib import Path
from history_book.entities import Book, Chapter, Paragraph
from history_book.text_processing import process_text

# Create entities manually
book = Book(
    title="My Custom Book",
    total_pages=100,
    book_index=0
)

chapter = Chapter(
    title="Introduction",
    start_page=1,
    end_page=10,
    book_id=book.id,
    chapter_index=0
)

paragraph = Paragraph(
    text="This is my custom paragraph content.",
    page_number=1,
    book_id=book.id,
    chapter_id=chapter.id,
    book_title=book.title,
    chapter_title=chapter.title
)

# Save to database
manager.book_repository.create(book)
manager.chapter_repository.create(chapter)
manager.paragraph_repository.create(paragraph)
```

### Batch Operations

```python
# Create multiple paragraphs at once
paragraphs = [
    Paragraph(text=f"Paragraph {i}", page_number=i, book_id=book.id, chapter_id=chapter.id)
    for i in range(1, 11)
]

# Batch insert
manager.paragraph_repository.create_many(paragraphs)
```

### Database Management

#### Clear Collections
```python
# Clear all data (use with caution!)
poetry run python scripts/inspect_and_clear_database.py
```

#### Inspect Database State
```python
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

config = WeaviateConfig.from_env()
manager = BookRepositoryManager(config)

# Count entities
book_count = len(manager.book_repository.get_all())
chapter_count = len(manager.chapter_repository.get_all())  
paragraph_count = len(manager.paragraph_repository.get_all())

print(f"Database contains:")
print(f"  Books: {book_count}")
print(f"  Chapters: {chapter_count}")
print(f"  Paragraphs: {paragraph_count}")
```

## Notebooks and Interactive Exploration

### Available Notebooks
- `notebooks/check_repo_interface.py`: Marimo notebook for testing repository interfaces
- `notebooks/paragraph_stats.ipynb`: Jupyter notebook for analyzing paragraph statistics
- `notebooks/paragraph_vector_eda.py`: Marimo notebook for vector analysis

### Running Marimo Notebooks
```bash
poetry run marimo run notebooks/check_repo_interface.py
```

### Running Jupyter Notebooks
```bash
poetry run jupyter lab notebooks/
```

## Environment Configuration

### Development Environment
- Local Weaviate instance
- Verbose logging
- Debug mode enabled
- Test collections prefixed with "Test_"

### Test Environment  
- Isolated test collections
- Minimal logging
- Fast teardown/setup
- Deterministic behavior

### Production Environment
- Cloud Weaviate instance
- Optimized logging
- Performance monitoring
- Error tracking

## Troubleshooting

### Common Issues

#### 1. Connection Errors
```
DatabaseConnectionError: Could not connect to Weaviate
```
**Solution**: Ensure Weaviate is running and accessible at the configured URL.

#### 2. Schema Errors
```
InvalidSchemaError: Collection schema mismatch
```
**Solution**: Clear existing collections or update schema migration.

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'history_book'
```
**Solution**: Ensure you're running commands with `poetry run` prefix.

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

For large PDFs:
1. Process in smaller page ranges
2. Use `clear_existing=False` for incremental ingestion
3. Monitor Weaviate resource usage
4. Consider batch size tuning

## Best Practices

1. **Always backup** your Weaviate data before major operations
2. **Use test environment** for experimentation
3. **Monitor resource usage** during large ingestions
4. **Validate PDF quality** before processing
5. **Use meaningful collection names** for organization
