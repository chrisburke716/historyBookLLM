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
│   ├── chat.py               # Session/history CRUD (/api/chat/sessions/*, /graph)
│   ├── agent.py              # AG-UI streaming endpoint (POST /api/chat/agent)
│   ├── books.py              # Book browsing endpoints (/api/books/*)
│   ├── kg.py                 # KG Explorer endpoints (/api/kg/*)
│   └── kg_metrics.py         # KG metrics endpoints
└── models/
    ├── chat_models.py        # Session + message Pydantic models
    ├── api_models.py         # Book API models
    └── kg_models.py          # KG API models
```

## Layer boundaries — routes are thin

Routes do HTTP/transport only and delegate to a service. They MUST NOT:
- access repositories directly or read state a service owns (caches, sessions, graphs, etc.)
- chain multiple service calls into a multi-step operation — that sequence belongs in one service method
- wrap/subclass infrastructure that adapts a service's internals (those wrappers live next to the service)
- duplicate logic that already exists on the service

Smell test: deleting a route file should lose URL plumbing only, not real logic.

For streaming endpoints: the service returns an `AsyncIterator[event]` and owns pre/post-side-effects in a `finally`; the route wraps the iterator with transport encoding and an error-frame fallback.

---

## Chat API (`/api/chat/*`)

Live chat goes through one streaming endpoint (`POST /api/chat/agent`); everything else is plain JSON session/history CRUD.

### Live chat (AG-UI streaming)

```http
POST /api/chat/agent
RunAgentInput {threadId, runId, messages, state, tools, context, forwardedProps}
→ text/event-stream (AG-UI events: RUN_STARTED, TEXT_MESSAGE_*, TOOL_CALL_*,
                     STATE_SNAPSHOT, RUN_FINISHED, RUN_ERROR, ...)
```

`thread_id` must match an existing session id. The route validates the session (400 if missing, 404 if unknown) and then delegates to `ChatService.send_message_agui`, wrapping the resulting `AsyncIterator[ProcessedEvents]` with `EventEncoder` for SSE framing. Errors mid-stream are emitted as `RUN_ERROR` events; the route doesn't break the SSE connection on failure.

Frontend consumers don't talk to this endpoint directly — CopilotKit's `HttpAgent` from `@ag-ui/client` does. See `frontend/CLAUDE.md`.

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

### History + graph

```http
GET /api/chat/sessions/{id}/messages
→ MessageListResponse {messages: [MessageResponse]}

GET /api/chat/sessions/{id}/graph
→ GraphVisualization {mermaid, nodes, edges}
```

`MessageResponse.citations` use the unified compact format `[B X, Ch Y, p. Z]`.

### Quick test (curl against the AG-UI endpoint)

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/chat/sessions \
  -H "Content-Type: application/json" -d '{"title": "Test"}' | jq -r '.id')

curl -N -X POST http://localhost:8000/api/chat/agent \
  -H "Content-Type: application/json" -H "Accept: text/event-stream" \
  -d "{
    \"threadId\": \"$SESSION_ID\",
    \"runId\": \"run-$(uuidgen)\",
    \"state\": {},
    \"messages\": [{\"id\": \"m1\", \"role\": \"user\", \"content\": \"Who was Charlemagne?\"}],
    \"tools\": [], \"context\": [], \"forwardedProps\": {}
  }"
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
class SessionCreateRequest(BaseModel):
    title: str | None = None

class MessageResponse(BaseModel):
    id, content, role, timestamp, session_id
    citations: list[str] | None   # e.g. ["[B3, Ch5, p.42]"]
    metadata: dict

class SessionResponse(BaseModel):
    id, title, created_at, updated_at

class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]

class MessageListResponse(BaseModel):
    messages: list[MessageResponse]

class GraphVisualization(BaseModel):
    mermaid, nodes, edges
```

The AG-UI endpoint uses `ag_ui.core.RunAgentInput` directly (not a model we define here).

## Dependency Injection

`ChatService` is a module-level singleton in `routes/chat.py` — this preserves the `MemorySaver` checkpointer across requests so conversation history survives between turns within a process. `routes/agent.py` imports the same `get_chat_service` from `chat.py` so both routers share the singleton.

```python
_chat_service: ChatService = ChatService()

def get_chat_service() -> ChatService:
    return _chat_service

# in agent.py:
@router.post("/agent")
async def agent_endpoint(
    input_data: RunAgentInput,
    request: Request,
    service: ChatService = Depends(get_chat_service),
):
    ...
```

## Error Handling

- `404` — session not found
- `500` — internal error (logged)
- HTTP exceptions re-raised; all others wrapped in 500

## Related Files

- Services: `services/chat_service.py`, `services/agents/` (includes the AG-UI wrapper)
- Frontend: `frontend/src/services/agentAPI.ts` (session CRUD), `frontend/src/components/CopilotProvider.tsx` (AG-UI HttpAgent)
