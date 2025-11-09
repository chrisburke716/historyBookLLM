# CLAUDE.md - Frontend (React TypeScript)

React 19 + TypeScript chat interface for the History Book RAG application. Uses Material-UI components and Axios for API communication.

## Quick Commands

```bash
# Install dependencies
cd frontend && npm install

# Start development server (http://localhost:3000)
cd frontend && npm start

# Build for production
cd frontend && npm run build

# Run tests
cd frontend && npm test

# Type checking
cd frontend && npx tsc --noEmit
```

## Tech Stack

- **React 19** with TypeScript
- **Material-UI (MUI)** v7 - Component library
- **Axios** - HTTP client
- **React Hooks** - State management
- **Jest + React Testing Library** - Testing

## Dual Backend Support

The frontend supports **two backend implementations** via a unified API abstraction:

1. **Agent API** (default) - LangGraph-based RAG (`/api/agent/*`)
   - Built on LangGraph for future extensibility (tools, planning, reflection)
   - Graph-based execution with checkpointing

2. **Chat API** (legacy) - LCEL-based RAG (`/api/chat/*`)
   - Original implementation using LangChain Expression Language
   - Simpler chain-based execution

### Switching Backends

Controlled by **`.env`** file in `frontend/` directory:

```bash
# Use Agent API (LangGraph) - default
REACT_APP_USE_AGENT_API=true

# Use Chat API (LCEL) - legacy
REACT_APP_USE_AGENT_API=false
```

The frontend code uses a **unified abstraction** (`api` from `services/api.ts`), so switching backends requires no code changes - just update the environment variable and restart `npm start`.

### Future Agent Features (Phase 7+)

When agent capabilities expand (tools, multi-step reasoning), the UI may expose:
- Graph visualization panels (Mermaid diagrams)
- Execution metadata displays (nodes executed, timing)
- Reasoning step viewers (planning, tool calls, reflection)
- Advanced settings (configure retrieval, enable tools)

These will be **optional/collapsible** features for power users. For now, the UI remains identical regardless of backend.

---

## Structure

```
frontend/src/
├── components/           # React components
│   ├── MessageInput.tsx      # User input field
│   ├── MessageList.tsx       # Message display
│   └── SessionDropdown.tsx   # Session selector
├── pages/
│   └── ChatPage.tsx          # Main chat page
├── services/
│   ├── api.ts               # Unified API abstraction (switches backends)
│   ├── agentAPI.ts          # Agent API client (LangGraph)
│   └── (ChatAPI in api.ts)  # Chat API client (LCEL)
├── hooks/
│   └── useChat.ts           # Chat state management hook
├── types/
│   ├── index.ts             # Shared TypeScript interfaces
│   └── agent.ts             # Agent-specific types (metadata, future features)
├── App.tsx                  # Root component
└── index.tsx                # Entry point
```

## Key Files

### API Client (`services/api.ts`)

The API client provides a **unified interface** that works with both backends (Chat and Agent).

**Unified API Export** (recommended):
```typescript
import { api } from '../services/api';

const session = await api.createSession({ title: 'New Chat' });
const response = await api.sendMessage(session.id, { content: 'Hello' });
```

The `api` export automatically switches between `ChatAPI` and `AgentAPI` based on `REACT_APP_USE_AGENT_API`.

**Individual API Classes**:

Both `ChatAPI` and `AgentAPI` implement the same interface:

```typescript
interface APIClient {
  createSession(request: SessionCreateRequest): Promise<SessionResponse>
  getSessions(limit: number): Promise<SessionListResponse>
  getSessionMessages(sessionId: string): Promise<MessageListResponse>
  sendMessage(sessionId: string, request: MessageRequest): Promise<ChatResponse>
  healthCheck(): Promise<{ message: string }>
}
```

- **ChatAPI**: Calls `/api/chat/*` endpoints (LCEL-based)
- **AgentAPI**: Calls `/api/agent/*` endpoints (LangGraph-based)

**Legacy access**:
```typescript
import { chatAPI, agentAPI } from '../services/api';

// Directly use specific backend (not recommended)
const response = await chatAPI.sendMessage(...);
const response = await agentAPI.sendMessage(...);
```

### Type Definitions (`types/index.ts`)

**Matches backend API models exactly**:

```typescript
// Request types
interface SessionCreateRequest {
  title?: string;
}

interface MessageRequest {
  content: string;
  enable_retrieval?: boolean;
  max_context_paragraphs?: number;
}

// Response types
interface SessionResponse {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
}

interface MessageResponse {
  id: string;
  content: string;
  role: string;  // "user" | "assistant"
  timestamp: string;
  session_id: string;
  citations?: string[];  // ["Page 42", ...]
  metadata?: AgentMetadata;  // Agent API only - graph execution details
}

// UI state
interface ChatState {
  currentSession: SessionResponse | null;
  sessions: SessionResponse[];
  messages: MessageResponse[];
  isLoading: boolean;
  error: string | null;
}
```

### Custom Hook (`hooks/useChat.ts`)

**Purpose**: Manages chat state and API interactions with React hooks.

**State Management**:
```typescript
const [state, setState] = useState<ChatState>({
  currentSession: null,
  sessions: [],
  messages: [],
  isLoading: false,
  error: null,
});
```

**Key Methods**:
- `loadSessions()` - Fetch session list
- `createSession(title)` - Create new session
- `selectSession(sessionId)` - Switch to session and load messages
- `sendMessage(content)` - Send message, get AI response
- `setError(error)` - Set error state
- `setLoading(loading)` - Set loading state

**Usage in Components**:
```typescript
import { useChat } from '../hooks/useChat';

const ChatPage = () => {
  const {
    state,
    loadSessions,
    createSession,
    selectSession,
    sendMessage
  } = useChat();

  // Component logic
};
```

## Components

### MessageInput

**Purpose**: Text input field with send button.

**Props**:
```typescript
interface MessageInputProps {
  onSendMessage: (content: string) => void;
  disabled?: boolean;
}
```

**Features**:
- Material-UI TextField
- Send button (disabled while loading)
- Enter key to send
- Auto-focus on load

### MessageList

**Purpose**: Displays conversation with citations.

**Props**:
```typescript
interface MessageListProps {
  messages: MessageResponse[];
  isLoading?: boolean;
}
```

**Features**:
- Scrollable container
- User messages (right-aligned, blue)
- Assistant messages (left-aligned, gray)
- Citations display below AI responses
- Loading indicator
- Auto-scroll to bottom on new messages

### SessionDropdown

**Purpose**: Session selector dropdown.

**Props**:
```typescript
interface SessionDropdownProps {
  sessions: SessionResponse[];
  currentSession: SessionResponse | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
}
```

**Features**:
- Material-UI Select component
- "New Chat" button
- Session list with titles

### ChatPage

**Purpose**: Main page - composes all components and manages state.

**Responsibilities**:
- Uses `useChat()` hook for state
- Renders SessionDropdown, MessageList, MessageInput
- Handles user interactions
- Coordinates API calls

**Lifecycle**:
1. Load sessions on mount
2. Auto-create session if none exists
3. User sends message → call `sendMessage()`
4. Display AI response with citations

## State Management Pattern

**Custom Hook Pattern** (not Redux/Context):

```typescript
// useChat.ts provides state + methods
const { state, loadSessions, sendMessage } = useChat();

// Components use state directly
{state.messages.map(msg => <Message key={msg.id} {...msg} />)}

// Components call methods
<MessageInput onSendMessage={(content) => sendMessage(content)} />
```

**Benefits**:
- Simple for small app
- No global state complexity
- Easy to test
- Type-safe with TypeScript

## Material-UI Components Used

- `TextField` - Text input
- `Button` - Send button, new session
- `Select`, `MenuItem` - Session dropdown
- `Paper`, `Box` - Layout containers
- `Typography` - Text styling
- `CircularProgress` - Loading spinner
- `Alert` - Error messages

## Environment Configuration

Frontend configuration is controlled by `.env` file in `frontend/` directory.

**Example `.env`**:
```bash
# Backend selection (default: Agent API)
REACT_APP_USE_AGENT_API=true

# Backend URL
REACT_APP_API_URL=http://localhost:8000
```

**Environment Variables**:
- `REACT_APP_USE_AGENT_API` - Set to `true` for Agent API (LangGraph), `false` for Chat API (LCEL)
- `REACT_APP_API_URL` - Backend base URL (default: `http://localhost:8000`)

**Note**: Changes to `.env` require restarting `npm start`.

## Development Workflow

### Local Development

1. **Start backend**:
```bash
PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload
```

2. **Start frontend**:
```bash
cd frontend && npm start
```

3. **Access**: http://localhost:3000

### Hot Reload

Both frontend and backend auto-reload on file changes.

## Common Tasks

### Adding New Component

```typescript
// src/components/NewComponent.tsx
import React from 'react';

interface NewComponentProps {
  data: string;
}

const NewComponent: React.FC<NewComponentProps> = ({ data }) => {
  return <div>{data}</div>;
};

export default NewComponent;
```

### Updating TypeScript Types

When backend API changes:
1. Update `types/index.ts` to match new API models
2. Run `npx tsc --noEmit` to check for errors
3. Fix type errors in components

### Adding API Method

```typescript
// services/api.ts
async newMethod(param: string): Promise<ResponseType> {
  const response = await this.api.get(`/api/new-endpoint/${param}`);
  return response.data;
}
```

### Adding Hook Method

```typescript
// hooks/useChat.ts
const newMethod = useCallback(async (param: string) => {
  try {
    setLoading(true);
    const result = await chatAPI.newMethod(param);
    setState(prev => ({ ...prev, newData: result }));
  } catch (error) {
    setError(`Failed: ${error}`);
  } finally {
    setLoading(false);
  }
}, [setLoading, setError]);
```

## Testing

### Run Tests

```bash
cd frontend && npm test
```

### Example Test

```typescript
import { render, screen } from '@testing-library/react';
import MessageInput from './MessageInput';

test('renders message input', () => {
  render(<MessageInput onSendMessage={jest.fn()} />);
  const inputElement = screen.getByPlaceholderText(/type a message/i);
  expect(inputElement).toBeInTheDocument();
});
```

## Build & Deployment

### Production Build

```bash
cd frontend && npm run build
```

**Output**: `build/` directory with optimized static files.

### Serve Build

```bash
# Local testing
npx serve -s build -p 3000

# Or configure nginx/Apache to serve static files
```

### Environment Variables

- Development: `.env.development`
- Production: `.env.production`

## Integration

**Upstream**: User browser at http://localhost:3000

**Downstream**: FastAPI backend at http://localhost:8000

**Flow**:
```
User → React UI → Axios → FastAPI → [ChatService | GraphChatService] → Database
                                      (LCEL)      (LangGraph)
```

**CORS**: Backend allows `http://localhost:3000` in development.

## Related Files

- **Backend APIs**:
  - `/src/history_book/api/routes/chat.py` - Chat API endpoints (LCEL)
  - `/src/history_book/api/routes/agent.py` - Agent API endpoints (LangGraph)
- **Agent Documentation**: `/src/history_book/services/agents/CLAUDE.md` - LangGraph implementation details
- **Root Documentation**: `/CLAUDE.md` - High-level architecture overview
- **Verification Scripts**:
  - `scripts/verify/verify_api.py` - API endpoint testing
  - `scripts/verify/verify_integration.py` - End-to-end tests with frontend + backend
