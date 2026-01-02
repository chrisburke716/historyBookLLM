# CLAUDE.md - API Layer

FastAPI REST API for the History Book RAG chat application. Provides HTTP endpoints for chat sessions and messaging.

## Quick Commands

```bash
# Start API server
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# View OpenAPI docs
open http://localhost:8000/docs

# Test health endpoint
curl http://localhost:8000/

# Run API tests
poetry run python test_api.py
```

## Structure

```
api/
├── main.py                    # FastAPI app setup, CORS
├── routes/
│   └── chat.py               # Chat endpoints
└── models/
    └── api_models.py         # Pydantic request/response models
```

## API Endpoints

### Base URL: `http://localhost:8000/api`

### Sessions

**Create Session**
```http
POST /api/chat/sessions
Content-Type: application/json

{
  "title": "Ancient History Chat"  // optional
}

Response: {
  "id": "uuid",
  "title": "Ancient History Chat",
  "created_at": "2025-11-07T12:00:00",
  "updated_at": "2025-11-07T12:00:00"
}
```

**List Sessions**
```http
GET /api/chat/sessions?limit=10

Response: {
  "sessions": [
    {
      "id": "uuid",
      "title": "Ancient History Chat",
      "created_at": "...",
      "updated_at": "..."
    },
    ...
  ]
}
```

### Messages

**Send Message**
```http
POST /api/chat/sessions/{session_id}/messages
Content-Type: application/json

{
  "content": "What were the first civilizations?",
  "enable_retrieval": true,  // optional, default true
  "max_context_paragraphs": 5  // optional, default 5
}

Response: {
  "message": {
    "id": "msg-uuid",
    "content": "The first civilizations...",
    "role": "assistant",
    "timestamp": "2025-11-07T12:01:00",
    "session_id": "session-uuid",
    "citations": ["Page 42", "Page 67", ...]
  }
}
```

**Get Session Messages**
```http
GET /api/chat/sessions/{session_id}/messages

Response: {
  "messages": [
    {
      "id": "msg-uuid-1",
      "content": "What were the first civilizations?",
      "role": "user",
      "timestamp": "...",
      "session_id": "session-uuid",
      "citations": null
    },
    {
      "id": "msg-uuid-2",
      "content": "The first civilizations...",
      "role": "assistant",
      "timestamp": "...",
      "session_id": "session-uuid",
      "citations": ["Page 42", "Page 67"]
    }
  ]
}
```

---

### Agent API Endpoints (`/api/agent/*`)

LangGraph-based chat API with checkpointing and graph visualization.

**Base URL**: `http://localhost:8000/api/agent`

#### Session Management

**Create Session**
```http
POST /api/agent/sessions
Content-Type: application/json

{
  "title": "My Agent Chat"  // optional
}

Response: {
  "id": "uuid",
  "title": "My Agent Chat",
  "created_at": "2025-11-09T...",
  "updated_at": "2025-11-09T..."
}
```

**List Sessions**
```http
GET /api/agent/sessions?limit=10

Response: {
  "sessions": [
    {
      "id": "uuid",
      "title": "My Agent Chat",
      "created_at": "...",
      "updated_at": "..."
    },
    ...
  ]
}
```

**Delete Session**
```http
DELETE /api/agent/sessions/{session_id}

Response: {
  "status": "deleted",
  "session_id": "uuid"
}
```

#### Messaging

**Send Message (Non-streaming)**
```http
POST /api/agent/sessions/{session_id}/messages
Content-Type: application/json

{
  "content": "What is the French Revolution?"
}

Response: {
  "message": {
    "id": "msg-uuid",
    "content": "The French Revolution was...",
    "role": "assistant",
    "timestamp": "2025-11-09T...",
    "session_id": "uuid",
    "citations": ["Page 42", "Page 67", ...],
    "metadata": {
      "num_retrieved_paragraphs": 40,
      "graph_execution": "simple_rag"
    }
  },
  "session": {
    "id": "uuid",
    "title": "French Revolution and Its Causes",
    "created_at": "2025-11-09T...",
    "updated_at": "2025-11-09T..."
  }
}
```

**Title Generation**: The `session` field in the response includes an auto-generated title after the first AI response. Titles are generated synchronously based on conversation content (max 100 chars). This eliminates the need for frontend polling - the title is immediately available in the response.

```

**Get Messages**
```http
GET /api/agent/sessions/{session_id}/messages

Response: {
  "messages": [
    {
      "id": "msg-uuid-1",
      "content": "What is the French Revolution?",
      "role": "user",
      "timestamp": "...",
      "session_id": "uuid",
      "citations": null,
      "metadata": {"graph_execution": "simple_rag"}
    },
    {
      "id": "msg-uuid-2",
      "content": "The French Revolution was...",
      "role": "assistant",
      "timestamp": "...",
      "session_id": "uuid",
      "citations": ["Page 42", ...],
      "metadata": {
        "num_retrieved_paragraphs": 40,
        "graph_execution": "simple_rag"
      }
    }
  ]
}
```

#### Graph Introspection

**Get Graph Visualization**
```http
GET /api/agent/sessions/{session_id}/graph

Response: {
  "mermaid": "graph TD\n    __start__ --> retrieve\n    ...",
  "nodes": ["retrieve", "generate"],
  "edges": [
    ["START", "retrieve"],
    ["retrieve", "generate"],
    ["generate", "END"]
  ]
}
```

The Mermaid diagram can be visualized at https://mermaid.live

#### Comparison with Chat API

| Feature | /api/chat/* (LCEL) | /api/agent/* (LangGraph) |
|---------|-------------------|--------------------------|
| Checkpointing | No | Yes (MemorySaver) |
| Multi-turn Context | Manual | Automatic |
| Graph Visualization | No | Yes (Mermaid) |
| Metadata | Basic | Enhanced (execution details) |
| Performance | 9.50s avg | 8.97s avg (faster) |
| Future Tools | No | Yes (extensible) |
| LangSmith Tracing | @traceable | Built-in graph structure |

#### When to Use

**Use `/api/agent/*` when**:
- Multi-turn conversations requiring context
- Need graph visualization for debugging
- Want better performance (5.6% faster)
- Planning to add tools or multi-step reasoning

**Use `/api/chat/*` when**:
- Simple one-off queries
- Existing integrations
- Don't need checkpointing

---

## Request/Response Models

**File**: `models/api_models.py`

### Request Models

```python
class SessionCreateRequest(BaseModel):
    title: str | None = None

class MessageRequest(BaseModel):
    content: str
    enable_retrieval: bool = True
    max_context_paragraphs: int = 5
```

### Response Models

```python
class SessionResponse(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

class MessageResponse(BaseModel):
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    session_id: str
    citations: list[str] | None = None  # e.g., ["Page 42", "Page 67"]

class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]

class MessageListResponse(BaseModel):
    messages: list[MessageResponse]

class ChatResponse(BaseModel):
    message: MessageResponse

class AgentChatResponse(BaseModel):
    message: AgentMessageResponse
    session: AgentSessionResponse  # Includes updated title
```

**Note**: Agent API (`/api/agent/*`) returns `AgentChatResponse` which includes both the message and the updated session (with auto-generated title). This allows the frontend to update the session title immediately without polling.

```

## App Configuration

**File**: `main.py`

### CORS Setup

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production**: Update `allow_origins` to include production frontend URL.

### Router Registration

```python
app.include_router(chat.router, prefix="/api")
```

All chat routes accessible at `/api/chat/*`.

## Dependency Injection

**File**: `routes/chat.py`

```python
def get_chat_service():
    """Dependency that provides ChatService instance."""
    return ChatService()

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: MessageRequest,
    chat_service: ChatService = Depends(get_chat_service)  # Injected
):
    result = await chat_service.send_message(...)
    return convert_message_to_response(result.message)
```

**Pattern**: FastAPI injects dependencies automatically via `Depends()`.

**Benefits**:
- Easy testing (mock service)
- Clean separation of concerns
- Centralized service instantiation

## Citation Optimization

**Implementation** (`routes/chat.py:131`):
```python
# Use retrieved paragraphs directly instead of database calls
response_message = convert_message_to_response(
    result.message,
    retrieved_paragraphs=result.retrieved_paragraphs  # From ChatResult
)
```

**Why**: `ChatService.send_message()` returns `ChatResult` with both message and retrieved paragraphs. API uses these directly to generate citations without additional DB queries.

## Error Handling

### HTTP Status Codes

- `200 OK` - Success
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Server error

### Example

```python
try:
    session = await chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # ... process request
except HTTPException:
    raise  # Re-raise HTTP exceptions
except Exception as e:
    logger.error(f"Failed to send message: {e}")
    raise HTTPException(status_code=500, detail="Failed to send message") from e
```

## OpenAPI Documentation

**Auto-generated** at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

**Generated from**:
- Pydantic models (request/response validation)
- Type hints in route handlers
- Docstrings in route functions

**App Metadata** (`main.py:11-15`):
```python
app = FastAPI(
    title="History Book Chat API",
    description="API for RAG-based chat with historical documents",
    version="0.1.0",
)
```

## Testing

### Manual Testing

```bash
# Create session
curl -X POST http://localhost:8000/api/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Send message (use session_id from above)
curl -X POST http://localhost:8000/api/chat/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "What were the first civilizations?"}'
```

### API Tests

```bash
poetry run python test_api.py
```

Uses FastAPI TestClient to test endpoints.

## Common Tasks

### Adding New Endpoint

1. **Define Models** (`models/api_models.py`):
```python
class NewRequest(BaseModel):
    field: str

class NewResponse(BaseModel):
    result: str
```

2. **Add Route** (`routes/chat.py` or new router):
```python
@router.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(
    request: NewRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    result = await chat_service.do_something(request.field)
    return NewResponse(result=result)
```

3. **Test**: Visit http://localhost:8000/docs and try endpoint.

### Modifying CORS

```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://production-domain.com"
    ],
    ...
)
```

### Adding Validation

```python
from pydantic import Field, validator

class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)
    enable_retrieval: bool = True

    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v
```

FastAPI automatically validates and returns 422 for invalid requests.

## Integration

**Upstream** (Frontend):
- React app at http://localhost:3000 calls these endpoints via Axios

**Downstream** (Services):
- Routes call `ChatService.send_message()`, `create_session()`, etc.
- Services handle business logic and database access

**Flow**:
```
Frontend (React) → API (FastAPI) → ChatService → RagService → Database
```

## Related Files

- Services: `/src/history_book/services/` - Business logic called by API
- Frontend: `/frontend/` - React app that consumes this API
- Tests: `test_api.py` - API endpoint tests
- Entity Models: `/src/history_book/data_models/entities.py` - Domain entities
