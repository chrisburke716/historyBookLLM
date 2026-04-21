# CLAUDE.md - API Layer

FastAPI REST API for the History Book RAG chat application.

## Quick Commands

```bash
# Start API server
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000

# View OpenAPI docs
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/
```

## Structure

```
api/
├── main.py                    # FastAPI app setup, CORS, router registration
├── routes/
│   ├── chat.py               # Chat endpoints (/api/chat/*)
│   ├── books.py              # Book browsing endpoints (/api/books/*)
│   ├── kg.py                 # KG Explorer endpoints (/api/kg/*)
│   └── kg_metrics.py         # KG metrics endpoints
└── models/
    ├── chat_models.py        # Chat request/response Pydantic models
    ├── api_models.py         # Book API models
    └── kg_models.py          # KG API models
```

## Chat API (`/api/chat/*`)

### Sessions

```http
POST /api/chat/sessions
{"title": "Optional title"}
→ SessionResponse {id, title, created_at, updated_at}

GET /api/chat/sessions?limit=10
→ SessionListResponse {sessions: [SessionResponse]}

DELETE /api/chat/sessions/{session_id}
→ {"status": "deleted", "session_id": "..."}
```

### Messages

```http
POST /api/chat/sessions/{id}/messages
{"content": "What caused WWI?"}
→ ChatResponse {message: MessageResponse, session: SessionResponse}

GET /api/chat/sessions/{id}/messages
→ MessageListResponse {messages: [MessageResponse]}

POST /api/chat/sessions/{id}/stream
{"content": "..."}
→ text/event-stream (token-by-token SSE)

GET /api/chat/sessions/{id}/graph
→ GraphVisualization {mermaid, nodes, edges}
```

`ChatResponse.session` includes the auto-generated title (updated synchronously after each response).

### Quick Test

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test"}' | jq -r '.id')

curl -X POST http://localhost:8000/api/chat/sessions/$SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Who was Julius Caesar?"}'
```

## Book API (`/api/books/*`)

```http
GET /api/books
GET /api/books/{book_index}/chapters
GET /api/books/{book_index}/chapters/{chapter_index}
```

## KG Explorer API (`/api/kg/*`)

```http
GET  /api/kg/graphs
GET  /api/kg/graphs/{graph_name}
GET  /api/kg/graphs/{graph_name}/subgraph?entity_id=<uuid>&hops=1|2|3
GET  /api/kg/entities/{entity_id}
POST /api/kg/search
```

## Models (`models/chat_models.py`)

```python
class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)

class MessageResponse(BaseModel):
    id, content, role, timestamp, session_id
    citations: list[str] | None   # e.g. ["Ch 5, p. 123"]
    metadata: dict                # num_retrieved_paragraphs, tool_iterations

class ChatResponse(BaseModel):
    message: MessageResponse
    session: SessionResponse      # includes updated title
```

## Dependency Injection

```python
def get_chat_service() -> ChatService:
    return ChatService()

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: MessageRequest,
    service: ChatService = Depends(get_chat_service),
):
    ...
```

## Error Handling

- `404` — session not found
- `500` — internal error (logged)
- HTTP exceptions re-raised; all others wrapped in 500

## Related Files

- Services: `services/chat_service.py`, `services/agents/`
- Frontend: `frontend/src/services/agentAPI.ts`
