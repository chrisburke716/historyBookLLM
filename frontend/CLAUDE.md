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
│   ├── BookSelector.tsx      # Book and chapter dropdown selector
│   ├── ChapterView.tsx       # Chapter content display
│   ├── MessageInput.tsx      # User input field
│   ├── MessageList.tsx       # Message display
│   └── SessionDropdown.tsx   # Session selector
├── pages/
│   ├── BookPage.tsx          # Book browsing page
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
├── App.tsx                  # Root component with routing
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

---

## Book Browsing Interface

The application includes a dedicated book browsing interface that allows users to read book content directly in the browser. This improves the workflow by eliminating the need to reference physical books while chatting.

**Access**: `/book` or `/book/:bookIndex/:chapterIndex`

### Features

- **Cascading Selection**: Book → Chapter dropdown navigation
- **URL-based Deep Linking**: Navigate directly to specific chapters via URL parameters
- **Scroll Position Persistence**: Automatically saves and restores reading position per chapter
- **Page Number Display**: Shows page numbers in left margin for citation reference
- **Fixed Navigation**: Book/Chapter selectors remain visible while scrolling
- **Tab Navigation**: Switch between Chat and Book modes via navigation bar

### Components

#### BookSelector

**Purpose**: Cascading dropdown for book and chapter selection.

**Props**:
```typescript
interface BookSelectorProps {
  selectedBookIndex: number | null;
  selectedChapterIndex: number | null;
  onSelectionChange: (bookIndex: number | null, chapterIndex: number | null) => void;
  disabled?: boolean;
}
```

**Features**:
- Loads books on mount from `/api/books`
- Loads chapters when book selected from `/api/books/{bookIndex}/chapters`
- Material-UI Select components with loading states
- Cascading behavior: chapter dropdown disabled until book selected
- Calls `onSelectionChange` callback when selection changes

**File**: `src/components/BookSelector.tsx`

#### ChapterView

**Purpose**: Displays chapter content with paragraphs and page numbers.

**Props**:
```typescript
interface ChapterViewProps {
  chapterContent: ChapterContentResponse | null;
  isLoading: boolean;
}
```

**Features**:
- Material-UI Paper wrapper for visual elevation
- Chapter title with metadata (page range)
- Flexbox layout with page numbers in left margin (60px)
- Paragraph text with justified alignment
- Loading state with CircularProgress
- Empty state when no chapter selected

**Layout**:
```
┌─────────────────────────────────────┐
│  Chapter Title                      │
│  Pages 42-67                        │
│                                     │
│  42   First paragraph text...       │
│       continuing on same page.      │
│                                     │
│  43   Next paragraph starts here... │
│                                     │
└─────────────────────────────────────┘
```

**File**: `src/components/ChapterView.tsx`

#### BookPage

**Purpose**: Main container that orchestrates book browsing experience.

**Responsibilities**:
- Manages URL parameters for book/chapter selection
- Loads chapter content from `/api/books/{bookIndex}/chapters/{chapterIndex}`
- Handles scroll position persistence via localStorage
- Provides scrollable container with fixed navigation
- Updates URL when selection changes

**State**:
```typescript
const [selectedBookIndex, setSelectedBookIndex] = useState<number | null>(null);
const [selectedChapterIndex, setSelectedChapterIndex] = useState<number | null>(null);
const [chapterContent, setChapterContent] = useState<ChapterContentResponse | null>(null);
const [isLoadingContent, setIsLoadingContent] = useState(false);
const [error, setError] = useState<string | null>(null);
```

**Layout Architecture**:
```typescript
<Container
  maxWidth="lg"
  sx={{
    height: 'calc(100vh - 64px)',  // Full height minus AppBar (64px)
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden'  // Prevent outer scroll
  }}
>
  {/* Fixed at top - doesn't scroll */}
  <BookSelector ... />

  {/* Scrollable container */}
  <Box
    ref={scrollContainerRef}
    sx={{
      flex: 1,
      overflow: 'auto',  // Enable scrolling
      mt: 2
    }}
  >
    <ChapterView ... />
  </Box>
</Container>
```

**File**: `src/pages/BookPage.tsx`

### URL Routing

**Routes** (defined in `App.tsx`):
```typescript
<Routes>
  <Route path="/book" element={<BookPage />} />
  <Route path="/book/:bookIndex/:chapterIndex" element={<BookPage />} />
</Routes>
```

**URL Parameters**:
- `bookIndex` - Integer index of selected book
- `chapterIndex` - Integer index of selected chapter within book

**Examples**:
- `/book` - No selection (shows empty state)
- `/book/3/4` - Book 3, Chapter 4

**Navigation Behavior**:
- Selecting book/chapter updates URL via `navigate()`
- URL changes trigger content loading via `useEffect`
- Browser back/forward buttons work naturally
- Deep linking: Share URLs to specific chapters

### Scroll Position Persistence

**Implementation**: Uses `localStorage` with automatic cleanup.

**Storage Format**:
```typescript
// localStorage key format
const key = `book-scroll-${bookIndex}-${chapterIndex}`;

// Stored value (JSON)
{
  position: number,      // Scroll position in pixels
  timestamp: number      // Date.now() for cleanup
}
```

**Lifecycle**:

1. **Save** (debounced 300ms on scroll):
```typescript
const saveScrollPosition = debounce(() => {
  const position = scrollContainerRef.current.scrollTop;
  const data = { position, timestamp: Date.now() };
  localStorage.setItem(key, JSON.stringify(data));
  cleanupOldScrollPositions();
}, 300);
```

2. **Restore** (when chapter loads):
```typescript
useEffect(() => {
  const savedData = localStorage.getItem(key);
  if (savedData) {
    const { position } = JSON.parse(savedData);
    scrollContainerRef.current?.scrollTo({
      top: position,
      behavior: 'smooth'
    });
  }
}, [chapterContent, selectedBookIndex, selectedChapterIndex]);
```

3. **Cleanup** (automatic, max 10 positions):
```typescript
const MAX_SAVED_POSITIONS = 10;

function cleanupOldScrollPositions() {
  // Get all scroll position keys with timestamps
  const scrollKeys = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith('book-scroll-')) {
      const data = JSON.parse(localStorage.getItem(key));
      scrollKeys.push({ key, timestamp: data.timestamp });
    }
  }

  // Remove oldest positions beyond MAX
  if (scrollKeys.length > MAX_SAVED_POSITIONS) {
    scrollKeys
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(0, scrollKeys.length - MAX_SAVED_POSITIONS)
      .forEach(({ key }) => localStorage.removeItem(key));
  }
}
```

**Benefits**:
- Maintains reading position across sessions
- Automatic cleanup prevents localStorage bloat
- Smooth scroll restoration for better UX
- Independent position per chapter

### Backend API Integration

**Book API Endpoints** (see `/src/history_book/api/routes/books.py`):

```typescript
// Get all books
GET /api/books
→ BookListResponse { books: BookResponse[] }

// Get chapters for a book
GET /api/books/{book_index}/chapters
→ ChapterListResponse { chapters: ChapterResponse[] }

// Get chapter content with paragraphs
GET /api/books/{book_index}/chapters/{chapter_index}
→ ChapterContentResponse {
    chapter: ChapterResponse,
    paragraphs: ParagraphResponse[]
  }
```

**Type Definitions** (`types/index.ts`):
```typescript
interface BookResponse {
  id: string;
  title: string;
  book_index: number;
  start_page: number;
  end_page: number;
}

interface ChapterResponse {
  id: string;
  title: string;
  chapter_index: number;
  book_index: number;
  start_page: number;
  end_page: number;
}

interface ParagraphResponse {
  text: string;
  page: number;
  paragraph_index: number;
}

interface ChapterContentResponse {
  chapter: ChapterResponse;
  paragraphs: ParagraphResponse[];
}
```

### Usage Workflow

1. **Navigate to Book Page**: Click "Book" tab in navigation bar
2. **Select Book**: Choose from dropdown (e.g., "Book 3: Medieval History")
3. **Select Chapter**: Choose from dropdown (e.g., "Chapter 4: The Crusades")
4. **Read Content**: Scroll through chapter paragraphs
5. **Navigate Away**: Switch to Chat tab or close browser
6. **Return**: Come back to same book/chapter → scroll position restored

### Development Notes

**Performance Optimizations**:
- Debounced scroll events (300ms) to reduce localStorage writes
- Cached repository manager on backend (via `@lru_cache`) to avoid connection overhead
- Delayed scroll restoration (100ms) to ensure content is fully rendered

**Layout Strategy**:
- Navigation bar fixed at top (doesn't scroll)
- Container height calculated as `calc(100vh - 64px)` where 64px is AppBar height
- Flexbox for layout control (vertical stacking)
- Scroll container uses `overflow: auto` for native browser scrolling

**Future Enhancements**:
- Click citations in chat to jump to book location
- Highlight text to auto-populate chat question
- Search within chapter
- Bookmark specific paragraphs

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
  - `/src/history_book/api/routes/books.py` - Book browsing API endpoints
- **Agent Documentation**: `/src/history_book/services/agents/CLAUDE.md` - LangGraph implementation details
- **Root Documentation**: `/CLAUDE.md` - High-level architecture overview
- **Verification Scripts**:
  - `scripts/verify/verify_api.py` - API endpoint testing
  - `scripts/verify/verify_integration.py` - End-to-end tests with frontend + backend
