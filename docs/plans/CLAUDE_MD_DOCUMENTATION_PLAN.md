# CLAUDE.md Documentation Plan

**Created**: 2025-11-07
**Purpose**: Comprehensive plan for adding CLAUDE.md documentation files throughout the History Book RAG codebase
**Status**: In Progress

---

## Current State

### Existing Documentation
✅ **Root `/CLAUDE.md`** - High-level overview, development commands, architecture summary
✅ **`/src/history_book/evals/CLAUDE.md`** - Evaluation framework documentation (created 2025-11-07)

### What's Missing
The codebase has 8+ major subsystems without dedicated documentation. Developers and Claude Code would benefit from detailed, subsystem-specific CLAUDE.md files that explain architecture, common tasks, and integration points.

---

## Architecture Context

### Technology Stack
- **Backend**: Python, FastAPI, LangChain, Weaviate (vector DB)
- **Frontend**: React 19, TypeScript, Material-UI, Axios
- **Infrastructure**: Docker (Weaviate), Poetry (Python deps), npm (frontend deps)
- **LLMs**: OpenAI (GPT-4, GPT-3.5), Anthropic (Claude)

### Clean Architecture Pattern
```
User Request
    ↓
API Layer (FastAPI routes)
    ↓
Service Layer (Business logic)
    ↓
Repository Layer (Data access)
    ↓
Database (Weaviate vector DB)
```

### Key Data Flows

**Chat Pipeline**:
```
User Message → API → ChatService → RagService → [Retrieval → LCEL Chain → LLM] → Response
```

**Ingestion Pipeline**:
```
PDF → IngestionService → Text Processing → Entity Creation → Repository → Weaviate
```

---

## Documentation Plan

### Tier 1: Essential Documentation (Create First)

#### 1. `/src/history_book/services/CLAUDE.md`
**Priority**: HIGHEST - This is the heart of the application
**Size**: ~1,700 lines across 4 service files
**Status**: Not created

**Content Outline**:
```markdown
# CLAUDE.md - Service Layer

## Overview
- Service layer responsibilities in clean architecture
- How services orchestrate repositories and external dependencies
- Async patterns and error handling

## Quick Commands
```bash
# Test specific service
poetry run python -c "from history_book.services import ChatService; ..."

# Start backend with services
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload
```

## ChatService (`chat_service.py` - 462 lines)

### Responsibilities
- Chat session lifecycle management
- Conversation history tracking
- RAG response orchestration
- Integration with RagService for retrieval

### Key Classes and Methods
- `ChatService` class
  - `create_session()` - Creates new chat session
  - `send_message()` - Processes user message, returns ChatResult
  - `get_session_history()` - Retrieves conversation history
  - `get_eval_metadata()` - Exports config for evaluation tracking

### ChatResult Dataclass
```python
@dataclass
class ChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
```
- Why: Enables evaluation framework to access retrieved context
- Where used: API layer, evaluation scripts

### Configuration
- System prompts
- Conversation history limits
- Session management

### Common Tasks
1. Modifying system prompts
2. Adjusting conversation history length
3. Adding new session metadata
4. Changing RAG orchestration logic

### Integration Points
- Calls `RagService` for retrieval and generation
- Uses `ChatRepository` for persistence
- Returns results to API layer

## RagService (`rag_service.py` - 405 lines)

### Responsibilities
- Retrieval-Augmented Generation logic
- Vector similarity search
- LangChain LCEL chain execution
- Context formatting and system prompts

### Key Components
- Direct LangChain integration (LCEL chains)
- Vector retrieval strategies
- Context window management
- LLM provider abstraction

### Configuration
- Similarity cutoff thresholds
- Min/max context results
- Retrieval strategies
- LLM model and temperature

### Common Tasks
1. Tuning retrieval parameters (similarity cutoff, max results)
2. Modifying context formatting
3. Changing LLM provider or model
4. Adjusting prompts and chain logic

### Integration Points
- Called by ChatService
- Uses ParagraphService/Repository for retrieval
- Integrates with LLMConfig (llm/)
- Uses LangChain LCEL chains

## IngestionService (`ingestion_service.py` - 627 lines)

### Responsibilities
- PDF processing pipeline orchestration
- Text extraction and chunking
- Entity creation (Book, Chapter, Paragraph)
- Batch storage in vector database

### Pipeline Flow
```
PDF File → PyMuPDF extraction → Text chunking →
Entity creation → Repository batch operations → Weaviate storage
```

### Key Methods
- `ingest_pdf()` - Main ingestion entry point
- Batch processing for efficiency
- Error handling and rollback

### Configuration
- Chunking parameters (size, overlap)
- Batch sizes
- Text processing rules

### Common Tasks
1. Adding new PDF sources
2. Adjusting chunk sizes
3. Modifying entity structure
4. Handling ingestion errors

### Integration Points
- Uses text_processing module
- Calls BookRepositoryManager
- Used by run_ingestion.py script

## ParagraphService (`paragraph_service.py` - 220 lines)

### Responsibilities
- High-level paragraph query operations
- Search and retrieval helpers
- Abstraction over repository layer

### Key Methods
- Search and filtering
- Pagination helpers
- Query optimization

### Common Tasks
1. Adding new search methods
2. Modifying query logic
3. Performance optimization

### Integration Points
- Used by RagService
- Calls ParagraphRepository

## Development Workflow

### Testing Services
```bash
# Unit tests (if they exist)
poetry run pytest tests/services/

# Integration testing
poetry run python test_api.py
```

### Adding New Service
1. Create new file in services/
2. Define service class with dependencies
3. Implement business logic methods
4. Add dependency injection in API layer
5. Update this documentation

### Modifying RAG Logic
1. Identify component: retrieval (ParagraphService) vs generation (RagService)
2. Modify parameters or logic
3. Test with chat_service_demo.ipynb
4. Run evals to measure impact: `poetry run python scripts/run_evals.py`

## Best Practices
- Keep services stateless (use repositories for state)
- Use async patterns consistently
- Inject dependencies (don't instantiate in services)
- Handle errors gracefully with custom exceptions
- Log important operations for debugging

## Related Files
- API Layer: `/src/history_book/api/CLAUDE.md`
- Database Layer: `/src/history_book/database/CLAUDE.md`
- LLM Config: `/src/history_book/llm/CLAUDE.md`
- Evaluation: `/src/history_book/evals/CLAUDE.md`
```

---

#### 2. `/src/history_book/database/CLAUDE.md`
**Priority**: HIGH - Foundation infrastructure layer
**Size**: ~1,000 lines across multiple files
**Status**: Not created

**Content Outline**:
```markdown
# CLAUDE.md - Database Layer

## Overview
- Repository pattern for clean separation
- Type-safe generic repositories with Pydantic models
- Weaviate vector database integration
- Environment-based configuration

## Quick Commands
```bash
# Start Weaviate
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.28.3

# Setup development config
poetry run python scripts/setup_development_config.py

# Inspect collections
poetry run python scripts/inspect_and_clear_database.py
```

## Directory Structure
```
database/
├── repositories/        # Repository implementations
│   ├── weaviate_repository.py    # Generic base
│   └── book_repository.py        # Specialized repos
├── interfaces/         # Abstract contracts
│   ├── repository_interface.py
│   └── vector_repository_interface.py
├── config/            # Configuration
│   └── database_config.py
├── exceptions/        # Custom errors
│   └── database_exceptions.py
├── server.py          # Weaviate client singleton
└── collections.py     # Schema generation
```

## Key Components

### WeaviateRepository<T> (Generic Base)
**File**: `repositories/weaviate_repository.py` (645 lines)

**Purpose**: Type-safe generic repository with CRUD operations

**Key Features**:
- Generic type parameter: `WeaviateRepository[Book]`, `WeaviateRepository[Paragraph]`
- Automatic Pydantic ↔ Weaviate conversion
- Vector similarity search
- Batch operations
- Filtering and pagination

**Methods**:
- `create(entity)` - Create single entity
- `get(id)` - Retrieve by ID
- `update(entity)` - Update existing
- `delete(id)` - Delete by ID
- `list(limit, offset)` - Paginated list
- `vector_search(query_vector, limit, filters)` - Similarity search
- `batch_create(entities)` - Bulk insert

### BookRepositoryManager
**File**: `repositories/book_repository.py` (286 lines)

**Purpose**: Aggregates specialized repositories for all entity types

**Repositories**:
- `BookRepository` - Books
- `ChapterRepository` - Chapters
- `ParagraphRepository` - Paragraphs with vector search
- `ChatSessionRepository` - Chat sessions
- `ChatMessageRepository` - Chat messages

**Usage**:
```python
from history_book.database.repositories import BookRepositoryManager

manager = BookRepositoryManager()
books = manager.book_repository.list()
results = manager.paragraph_repository.vector_search(query_vector, limit=5)
```

### Repository Interfaces
**File**: `interfaces/repository_interface.py`

**Purpose**: Abstract contracts for repositories

**Interfaces**:
- `RepositoryInterface[T]` - Base CRUD operations
- `VectorRepositoryInterface[T]` - Adds vector search capability

### Database Configuration
**File**: `config/database_config.py`

**Purpose**: Environment-based Weaviate configuration

**Environments**:
- `development` - Local Weaviate, verbose logging
- `test` - Isolated collections, temporary data
- `production` - Production Weaviate instance

**Environment Variables**:
```bash
WEAVIATE_URL=http://localhost:8080
ENVIRONMENT=development
```

### Weaviate Client Singleton
**File**: `server.py`

**Purpose**: Manages Weaviate client lifecycle

**Functions**:
- `get_weaviate_client()` - Returns configured client
- Singleton pattern ensures single connection
- Handles initialization and connection errors

### Schema Generation
**File**: `collections.py`

**Purpose**: Auto-generates Weaviate schemas from Pydantic models

**Features**:
- Pydantic field types → Weaviate data types
- Vector configuration
- Nested object handling

## Repository Pattern

### Why Use Repository Pattern?
- **Abstraction**: Business logic doesn't know about Weaviate
- **Testability**: Easy to mock repositories
- **Consistency**: Same interface across entity types
- **Type Safety**: Generic types catch errors at compile time

### Adding New Entity Type

**Step 1: Define Entity** (`data_models/entities.py`)
```python
from pydantic import BaseModel

class Author(BaseModel):
    id: str | None = None
    name: str
    birth_year: int
```

**Step 2: Create Repository** (`database/repositories/book_repository.py`)
```python
from history_book.database.repositories import WeaviateRepository
from history_book.data_models import Author

class AuthorRepository(WeaviateRepository[Author]):
    def __init__(self, client):
        super().__init__(client=client, model=Author, collection_name="Author")
```

**Step 3: Add to Manager**
```python
class BookRepositoryManager:
    def __init__(self, client=None):
        # ... existing repos ...
        self.author_repository = AuthorRepository(client)
```

**Step 4: Create Collection Schema**
```bash
# Schema auto-generated from Pydantic model
# Run setup script to create in Weaviate
poetry run python scripts/setup_development_config.py
```

## Vector Search

### How It Works
1. Text → Embedding (during ingestion)
2. Query → Embedding (during search)
3. Cosine similarity in Weaviate
4. Return top K results

### Configuring Search
```python
results = paragraph_repo.vector_search(
    query_vector=query_embedding,
    limit=10,
    filters={"book_id": "some-book-id"},
    distance_threshold=0.8  # Similarity cutoff
)
```

### Tuning Parameters
- `limit` - Max results to return
- `distance_threshold` - Minimum similarity score
- Use `/notebooks/investigate_vector_search_cutoff.ipynb` to tune

## Environment Management

### Development Environment
```bash
poetry run python scripts/setup_development_config.py
```
- Creates local collections
- Verbose logging
- Easier debugging

### Test Environment
```bash
poetry run python scripts/setup_test_config.py
```
- Isolated test collections
- Cleaned up after tests
- No pollution of dev data

### Switching Environments
```bash
poetry run python scripts/switch_environment.py
```

## Common Tasks

### Inspecting Database
```bash
poetry run python scripts/inspect_and_clear_database.py
```

### Managing Collections
```bash
poetry run python scripts/manage_collections.py
```

### Deleting Collections
```bash
poetry run python scripts/delete_collections.py
```

### Troubleshooting Connection Issues
1. Check Weaviate is running: `curl http://localhost:8080/v1/.well-known/ready`
2. Verify WEAVIATE_URL environment variable
3. Check `database_config.py` for correct environment
4. Review logs for connection errors

## Best Practices

### Repository Usage
- Always use repositories, never direct Weaviate client in services
- Use batch operations for multiple entities
- Handle exceptions gracefully
- Log important operations

### Schema Changes
- Modify Pydantic model first
- Regenerate schema with setup script
- May require collection recreation (data loss!)
- Plan migrations for production

### Performance
- Use batch operations for bulk inserts
- Index frequently queried fields
- Monitor vector search latency
- Tune similarity thresholds

## Related Files
- Service Layer: `/src/history_book/services/CLAUDE.md`
- Entity Definitions: `/src/history_book/data_models/entities.py`
- Setup Scripts: `/scripts/CLAUDE.md`
```

---

#### 3. `/src/history_book/api/CLAUDE.md`
**Priority**: HIGH - Integration point for frontend
**Size**: ~150 lines
**Status**: Not created

**Content Outline**:
```markdown
# CLAUDE.md - API Layer

## Overview
- FastAPI REST API layer
- HTTP endpoints for chat functionality
- Request/response validation with Pydantic
- OpenAPI documentation auto-generated

## Quick Commands
```bash
# Start API server
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# View API docs
open http://localhost:8000/docs

# Test endpoint
curl -X POST http://localhost:8000/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Session"}'
```

## Directory Structure
```
api/
├── main.py              # FastAPI app setup, CORS, middleware
├── routes/
│   └── chat.py          # Chat endpoints
└── models/
    └── api_models.py    # Request/response Pydantic models
```

## API Endpoints

### Session Management

**Create Session**
```
POST /chat/sessions
Request: {"name": "Session Name"}
Response: {
  "id": "uuid",
  "name": "Session Name",
  "created_at": "2025-11-07T12:00:00"
}
```

**List Sessions**
```
GET /chat/sessions
Response: [
  {"id": "uuid", "name": "Session 1", "created_at": "..."},
  ...
]
```

### Message Management

**Send Message**
```
POST /chat/sessions/{session_id}/messages
Request: {"content": "What were the first civilizations?"}
Response: {
  "id": "msg-uuid",
  "session_id": "session-uuid",
  "role": "assistant",
  "content": "The first civilizations were...",
  "citations": [
    {
      "paragraph_id": "para-uuid",
      "book_title": "Ancient History",
      "chapter_title": "Early Civilizations",
      "page_number": 42,
      "text": "Mesopotamia and Egypt..."
    }
  ],
  "created_at": "2025-11-07T12:00:00"
}
```

**Get Message History**
```
GET /chat/sessions/{session_id}/messages
Response: [
  {"role": "user", "content": "...", ...},
  {"role": "assistant", "content": "...", ...},
  ...
]
```

## Request/Response Models

### API Models (`models/api_models.py`)

**CreateSessionRequest**
```python
class CreateSessionRequest(BaseModel):
    name: str
```

**SendMessageRequest**
```python
class SendMessageRequest(BaseModel):
    content: str
```

**MessageResponse**
```python
class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    citations: list[Citation]
    created_at: datetime
```

**Citation**
```python
class Citation(BaseModel):
    paragraph_id: str
    book_title: str
    chapter_title: str
    page_number: int | None
    text: str
```

## FastAPI App Setup

### Main Application (`main.py`)

**CORS Configuration**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Router Registration**:
```python
app.include_router(chat_router, prefix="/chat", tags=["chat"])
```

## Dependency Injection

### Service Dependencies (`routes/chat.py`)

**Pattern**:
```python
def get_chat_service():
    return ChatService()

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    result = await chat_service.send_message(session_id, request.content)
    return convert_to_response(result)
```

**Benefits**:
- Easy testing (mock dependencies)
- Centralized service creation
- Clean route handlers

## Citation Generation

### How It Works

1. ChatService returns `ChatResult` with message + retrieved paragraphs
2. API layer formats paragraphs into `Citation` objects
3. Citations include book/chapter context for frontend display

**Optimization**:
- Retrieved paragraphs already include book/chapter data
- No additional database queries needed
- Efficient for large result sets

## Error Handling

### HTTP Status Codes
- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Session/message not found
- `500 Internal Server Error` - Server error

### Error Responses
```python
{
  "detail": "Error message"
}
```

## Testing

### Using FastAPI TestClient
```python
from fastapi.testclient import TestClient
from history_book.api.main import app

client = TestClient(app)

def test_create_session():
    response = client.post("/chat/sessions", json={"name": "Test"})
    assert response.status_code == 200
```

### Manual Testing
```bash
# Create session
curl -X POST http://localhost:8000/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Session"}'

# Send message (use session_id from previous response)
curl -X POST http://localhost:8000/chat/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "What were the first civilizations?"}'
```

### Test Files
- `test_api.py` - API endpoint tests

## OpenAPI Documentation

### Accessing Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Auto-Generated From**:
- Pydantic models for request/response validation
- Type hints in route handlers
- Docstrings in route functions

## Common Tasks

### Adding New Endpoint

1. **Define Models** (`models/api_models.py`):
```python
class NewRequest(BaseModel):
    field: str

class NewResponse(BaseModel):
    result: str
```

2. **Create Route** (`routes/chat.py` or new router):
```python
@router.post("/new-endpoint")
async def new_endpoint(
    request: NewRequest,
    service: SomeService = Depends(get_service)
):
    result = service.do_something(request.field)
    return NewResponse(result=result)
```

3. **Test**: Visit http://localhost:8000/docs and try the endpoint

### Modifying Existing Endpoint

1. Update Pydantic models if request/response changes
2. Modify route handler logic
3. Update tests in `test_api.py`
4. Check OpenAPI docs to ensure changes are reflected

### Changing CORS Settings

Edit `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://production.com"],
    ...
)
```

## Best Practices

- Use Pydantic models for all request/response types
- Keep route handlers thin (business logic in services)
- Use dependency injection for services
- Return appropriate HTTP status codes
- Add docstrings to routes for OpenAPI docs
- Validate inputs with Pydantic (automatic with FastAPI)
- Handle exceptions and return meaningful errors

## Related Files
- Service Layer: `/src/history_book/services/CLAUDE.md`
- Frontend Client: `/frontend/CLAUDE.md`
- API Tests: `test_api.py`
```

---

#### 4. `/frontend/CLAUDE.md`
**Priority**: HIGH - Separate tech stack
**Size**: ~300 lines
**Status**: Not created

**Content Outline**:
```markdown
# CLAUDE.md - Frontend (React TypeScript)

## Overview
- React 19 with TypeScript
- Material-UI component library
- Axios for API communication
- State management with React hooks

## Quick Commands
```bash
# Install dependencies
cd frontend && npm install

# Start development server
cd frontend && npm start
# Opens http://localhost:3000

# Build for production
cd frontend && npm run build

# Run tests
cd frontend && npm test

# Type checking
cd frontend && npx tsc --noEmit

# Linting
cd frontend && npm run lint
```

## Directory Structure
```
frontend/
├── src/
│   ├── components/      # React components
│   │   ├── MessageInput.tsx
│   │   ├── MessageList.tsx
│   │   └── SessionDropdown.tsx
│   ├── pages/          # Page components
│   │   └── ChatPage.tsx
│   ├── services/       # API client
│   │   └── api.ts
│   ├── types/          # TypeScript type definitions
│   │   └── index.ts
│   ├── App.tsx         # Root component
│   └── index.tsx       # Entry point
├── public/             # Static assets
└── package.json        # Dependencies
```

## Components

### MessageInput
**File**: `src/components/MessageInput.tsx`

**Purpose**: User input field with send button

**Props**:
```typescript
interface MessageInputProps {
  onSendMessage: (content: string) => void;
  disabled?: boolean;
}
```

**Features**:
- Text input with Material-UI TextField
- Send button with Icon
- Enter key to send
- Disabled state during message processing

### MessageList
**File**: `src/components/MessageList.tsx`

**Purpose**: Displays conversation history with citations

**Props**:
```typescript
interface MessageListProps {
  messages: Message[];
  loading?: boolean;
}
```

**Features**:
- Scrollable message container
- User vs Assistant message styling
- Citation display with expandable details
- Loading indicator

**Message Display**:
- User messages: Right-aligned, blue
- Assistant messages: Left-aligned, gray
- Citations: Expandable accordions with book/chapter/page info

### SessionDropdown
**File**: `src/components/SessionDropdown.tsx`

**Purpose**: Session management UI

**Props**:
```typescript
interface SessionDropdownProps {
  sessions: Session[];
  currentSessionId: string | null;
  onSessionSelect: (sessionId: string) => void;
  onNewSession: () => void;
}
```

**Features**:
- Dropdown to select existing sessions
- "New Session" button
- Session name display

### ChatPage
**File**: `src/pages/ChatPage.tsx`

**Purpose**: Main page composition and state management

**State**:
```typescript
const [sessions, setSessions] = useState<Session[]>([]);
const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
const [messages, setMessages] = useState<Message[]>([]);
const [loading, setLoading] = useState(false);
```

**Lifecycle**:
1. Load sessions on mount
2. Create/select session
3. Load message history
4. Send messages
5. Display responses

## API Client

### API Service (`services/api.ts`)

**Purpose**: Axios-based client for backend communication

**Configuration**:
```typescript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});
```

**Methods**:

**Session Management**:
```typescript
export const createSession = async (name: string): Promise<Session> => {
  const response = await apiClient.post('/chat/sessions', { name });
  return response.data;
};

export const getSessions = async (): Promise<Session[]> => {
  const response = await apiClient.get('/chat/sessions');
  return response.data;
};
```

**Message Management**:
```typescript
export const sendMessage = async (
  sessionId: string,
  content: string
): Promise<Message> => {
  const response = await apiClient.post(
    `/chat/sessions/${sessionId}/messages`,
    { content }
  );
  return response.data;
};

export const getMessages = async (sessionId: string): Promise<Message[]> => {
  const response = await apiClient.get(`/chat/sessions/${sessionId}/messages`);
  return response.data;
};
```

## Type Definitions

### Types (`types/index.ts`)

**Session**:
```typescript
export interface Session {
  id: string;
  name: string;
  created_at: string;
}
```

**Message**:
```typescript
export interface Message {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  created_at: string;
}
```

**Citation**:
```typescript
export interface Citation {
  paragraph_id: string;
  book_title: string;
  chapter_title: string;
  page_number: number | null;
  text: string;
}
```

## State Management

### React Hooks Pattern

**State in ChatPage**:
- `useState` for component state
- `useEffect` for side effects (loading data)
- Props for passing data/callbacks to children

**Example Flow**:
```typescript
// Load sessions on mount
useEffect(() => {
  const loadSessions = async () => {
    const sessionList = await getSessions();
    setSessions(sessionList);
  };
  loadSessions();
}, []);

// Send message
const handleSendMessage = async (content: string) => {
  setLoading(true);
  try {
    const message = await sendMessage(currentSessionId!, content);
    setMessages([...messages, message]);
  } catch (error) {
    console.error('Error sending message:', error);
  } finally {
    setLoading(false);
  }
};
```

## Styling

### Material-UI Theme

**Components Used**:
- `TextField` - Text input
- `Button` - Send button, new session button
- `Select` - Session dropdown
- `Paper` - Message containers
- `Accordion` - Citation expansion
- `CircularProgress` - Loading indicator

**Customization**:
- Theme configured in `App.tsx`
- Component-level styling with `sx` prop
- Responsive design with Material-UI breakpoints

## Development Workflow

### Local Development

1. **Start Backend**:
```bash
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload
```

2. **Start Frontend**:
```bash
cd frontend && npm start
```

3. **Access**: http://localhost:3000

### Environment Variables

Create `.env` in `frontend/`:
```
REACT_APP_API_URL=http://localhost:8000
```

### Hot Reload
- Frontend: Changes auto-reload
- Backend: Changes auto-reload with `--reload` flag

## Common Tasks

### Adding New Component

1. **Create Component** (`src/components/NewComponent.tsx`):
```typescript
import React from 'react';

interface NewComponentProps {
  data: string;
}

const NewComponent: React.FC<NewComponentProps> = ({ data }) => {
  return <div>{data}</div>;
};

export default NewComponent;
```

2. **Import and Use**:
```typescript
import NewComponent from './components/NewComponent';

<NewComponent data="Hello" />
```

### Modifying API Client

1. Add new method to `services/api.ts`
2. Define TypeScript types in `types/index.ts`
3. Use in components with async/await

### Updating Types

When backend API changes:
1. Update TypeScript interfaces in `types/index.ts`
2. Update API client methods if needed
3. Update components using those types
4. Run type checker: `npx tsc --noEmit`

### Adding New Page

1. Create page component in `src/pages/`
2. Add routing (if using React Router)
3. Update navigation

## Testing

### Jest + React Testing Library

**Running Tests**:
```bash
cd frontend && npm test
```

**Example Test**:
```typescript
import { render, screen } from '@testing-library/react';
import MessageInput from './MessageInput';

test('renders message input', () => {
  render(<MessageInput onSendMessage={jest.fn()} />);
  const inputElement = screen.getByPlaceholderText(/type a message/i);
  expect(inputElement).toBeInTheDocument();
});
```

## Build and Deployment

### Production Build
```bash
cd frontend && npm run build
```

**Output**: `build/` directory with optimized static files

### Serving Build
```bash
# Using serve package
npx serve -s build -p 3000

# Or configure web server (nginx, Apache) to serve static files
```

### Environment Variables
- Development: `.env.development`
- Production: `.env.production`

## Best Practices

- Use TypeScript for type safety
- Define interfaces for all props and state
- Use functional components with hooks (no class components)
- Keep components small and focused
- Extract reusable logic into custom hooks
- Handle loading and error states
- Use Material-UI components consistently
- Follow React naming conventions (PascalCase for components)

## Related Files
- Backend API: `/src/history_book/api/CLAUDE.md`
- API Tests: `test_full_integration.py`
```

---

### Tier 2: Valuable Documentation (Create Next)

#### 5. `/scripts/CLAUDE.md`
**Priority**: MEDIUM - Operational knowledge
**Size**: ~500 lines across 8+ scripts
**Status**: Not created

**Content Outline**:
```markdown
# CLAUDE.md - Utility Scripts

## Overview
Operational scripts for setup, data pipeline, and maintenance

## Scripts

### Data Pipeline
- `run_ingestion.py` - Process PDFs into vector database
- `run_evals.py` - Run evaluation framework

### Environment Setup
- `setup_development_config.py` - Initialize dev environment
- `setup_test_config.py` - Initialize test environment
- `switch_environment.py` - Switch configs

### Database Management
- `manage_collections.py` - Collection operations
- `delete_collections.py` - Clean up
- `inspect_and_clear_database.py` - Inspection

## When to Use Each Script
[Detailed guide for each script]

## Related Files
- Services: `/src/history_book/services/CLAUDE.md`
- Database: `/src/history_book/database/CLAUDE.md`
```

---

#### 6. `/src/history_book/llm/CLAUDE.md`
**Priority**: MEDIUM - LLM abstraction
**Size**: ~100 lines
**Status**: Has README.md (convert/extend)

**Content Outline**:
```markdown
# CLAUDE.md - LLM Configuration

## Overview
Provider-agnostic LLM configuration for OpenAI/Anthropic

## LLMConfig
- Environment-driven configuration
- Provider abstraction
- Model selection

## Switching Providers
[Guide to change from OpenAI to Anthropic]

## Integration
Used by RagService for LangChain

## Related Files
- RagService: `/src/history_book/services/CLAUDE.md`
```

---

#### 7. `/notebooks/CLAUDE.md`
**Priority**: MEDIUM - Data science workflow
**Status**: Not created

**Content Outline**:
```markdown
# CLAUDE.md - Analysis Notebooks

## Overview
Jupyter notebooks for exploration and analysis

## Key Notebooks
- `BuildEvalDataset.ipynb` - Create eval datasets
- `chat_service_demo.ipynb` - Interactive testing
- `investigate_vector_search_cutoff.ipynb` - Parameter tuning
- `paragraph_stats.ipynb` - Data analysis

## Workflow
When to use notebooks vs scripts

## Running Notebooks
`jupyter notebook notebooks/`

## Related Files
- Evaluation: `/src/history_book/evals/CLAUDE.md`
```

---

### Tier 3: Optional Documentation

#### 8. `/src/history_book/database/repositories/CLAUDE.md`
**Priority**: LOW - Deep dive (optional)
**Size**: ~900 lines
**Status**: Not created

**Note**: Only create if developers frequently extend repositories. Otherwise, coverage in `/src/history_book/database/CLAUDE.md` is sufficient.

---

## Documentation Themes

Every CLAUDE.md should include these sections:

### 1. Overview
- What this subsystem does
- Why it exists
- Key responsibilities

### 2. Quick Commands
```bash
# Commands to run/test this subsystem
```

### 3. Architecture/Structure
- Directory layout
- Key files and their purposes
- How components relate

### 4. Key Components
- Detailed explanation of main classes/functions
- Code examples
- Configuration options

### 5. Integration Points
- How this subsystem connects to others
- Data flow
- Dependencies

### 6. Common Tasks
- Step-by-step guides for typical modifications
- "How do I...?" answers
- Code snippets

### 7. Best Practices
- Patterns to follow
- Anti-patterns to avoid
- Performance considerations

### 8. Related Files
- Links to other CLAUDE.md files
- Cross-references

---

## Hierarchical Navigation

### Root CLAUDE.md (Exists)
High-level navigation hub pointing to:
- `/src/history_book/services/CLAUDE.md`
- `/src/history_book/database/CLAUDE.md`
- `/src/history_book/api/CLAUDE.md`
- `/src/history_book/evals/CLAUDE.md`
- `/frontend/CLAUDE.md`
- `/scripts/CLAUDE.md`
- `/notebooks/CLAUDE.md`
- `/src/history_book/llm/CLAUDE.md`

### Subsystem CLAUDE.md Files
Detailed component guides that:
- Link back to root CLAUDE.md
- Cross-reference other subsystem CLAUDE.md files
- Provide deep technical detail

---

## Implementation Order

### Phase 1: Core Backend (1-3 files)
1. `/src/history_book/services/CLAUDE.md` - HIGHEST PRIORITY
2. `/src/history_book/database/CLAUDE.md`
3. `/src/history_book/api/CLAUDE.md`

**Rationale**: Services are the heart of the application. Database is the foundation. API is the integration point. These three form the core backend architecture.

### Phase 2: Frontend & Operations (2-3 files)
4. `/frontend/CLAUDE.md`
5. `/scripts/CLAUDE.md`
6. `/notebooks/CLAUDE.md`

**Rationale**: Frontend is a separate codebase. Scripts and notebooks support development workflow.

### Phase 3: Advanced Topics (1-2 files)
7. `/src/history_book/llm/CLAUDE.md`
8. `/src/history_book/database/repositories/CLAUDE.md` (optional)

**Rationale**: Nice-to-have deep dives on specific topics.

---

## Next Steps

### To Continue This Work:

1. **Choose Priority Level**: Start with Phase 1 (services, database, api)

2. **Pick a File**: Select specific CLAUDE.md to create

3. **Reference This Plan**: Use the detailed content outlines above

4. **Follow Template**: Use the Documentation Themes section for structure

5. **Cross-Link**: Add references to other CLAUDE.md files

6. **Update Root**: Update root `/CLAUDE.md` to reference new files

### Commands to Create Files:

```bash
# Create services CLAUDE.md
# Use Write tool with content from this plan

# Create database CLAUDE.md
# Use Write tool with content from this plan

# Create api CLAUDE.md
# Use Write tool with content from this plan

# etc.
```

### Validation:

After creating each file:
1. ✅ Does it follow the Documentation Themes structure?
2. ✅ Does it include Quick Commands section?
3. ✅ Does it cross-reference other CLAUDE.md files?
4. ✅ Is it detailed enough to help developers?
5. ✅ Is it concise enough to be useful?

---

## Maintenance

### Keeping Documentation Current:

- Update CLAUDE.md when architecture changes
- Add new sections for new features
- Remove/archive sections for deprecated code
- Keep commands and code examples working
- Review quarterly for accuracy

### Signs Documentation Needs Update:

- Commands in CLAUDE.md don't work
- New developers ask questions already "documented"
- Code examples are outdated
- New subsystems exist without documentation
- Integration points have changed

---

## Context for Claude Code

When Claude Code reads these CLAUDE.md files, it will:
- Understand architecture and design patterns
- Know where to find specific functionality
- Follow established conventions
- Make changes consistent with existing code
- Understand integration points and dependencies
- Use appropriate testing strategies
- Follow best practices for each subsystem

This hierarchical documentation system creates a **knowledge graph** of the codebase that both humans and AI can navigate effectively.
