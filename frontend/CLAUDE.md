# CLAUDE.md - Frontend (React TypeScript)

React 19 + TypeScript UI for the History Book RAG application. Three pages: Chat, Book browsing, KG Explorer.

## Quick Commands

```bash
# Install dependencies
cd frontend && npm install

# Dev server (http://localhost:3000)
cd frontend && npm start

# Production build
cd frontend && npm run build

# Type check
cd frontend && npx tsc --noEmit
```

## Tech Stack

- **React 19** with TypeScript (5.x; bumped above CRA's 4.x default because CopilotKit's `.d.cts` files use modern TS syntax)
- **Material-UI v7** — component library and theming
- **CopilotKit** (`@copilotkit/react-core`) + `@ag-ui/client` — chat hooks + AG-UI transport
- **react-markdown** + **remark-gfm** — assistant message rendering
- **TanStack Query** — server state for chat session list/history + KG data
- **Redux Toolkit + react-redux** — UI state for KG Explorer
- **Axios** — REST surface (sessions, history, books, KG); live chat goes through CopilotKit's HttpAgent, not Axios
- **react-force-graph-2d** — KG canvas

---

## Structure

```
frontend/src/
├── components/
│   ├── CopilotProvider.tsx        # Wraps chat page with <CopilotKit> + HttpAgent
│   ├── ChatThreadController.tsx   # Hydrates history; fires onRunEnd on agent completion
│   ├── MessageList.tsx            # Renders conversation as turns (user + tool calls + answer)
│   ├── MessageInput.tsx           # Driven by useCopilotChatInternal.sendMessage
│   ├── ToolCallCard.tsx           # Compact MUI chip for a single tool call
│   ├── SessionDropdown.tsx        # Session selector
│   ├── BookSelector.tsx, ChapterView.tsx
│   └── kg/
│       ├── KGTopBar.tsx
│       ├── ForceGraphPanel.tsx
│       └── EntityPanel.tsx
├── pages/
│   ├── ChatPage.tsx
│   ├── BookPage.tsx
│   └── KGPage.tsx
├── services/
│   ├── api.ts                     # Re-exports for backwards compat
│   ├── agentAPI.ts                # REST client: sessions + history (no chat send)
│   └── kgAPI.ts                   # KG API client
├── store/                         # Redux (KG Explorer only)
├── hooks/
│   ├── useChat.ts                 # Session list + history + currentSession
│   └── useKGQueries.ts            # KG TanStack hooks
├── types/index.ts                 # Interfaces matching backend Pydantic models
├── App.tsx                        # Routes + provider stack
└── index.tsx
```

---

## Chat surface

Custom MUI components driven by CopilotKit hooks. No prebuilt `<CopilotChat>` — design consistency with the rest of the MUI app outweighs the implementation savings.

### Component graph

```
ChatPage
  ├── SessionDropdown                  ← useChat (sessions, switchToSession, createSession)
  └── CopilotProvider                  ← scopes <CopilotKit> to the chat page only
        ├── ChatThreadController       ← setMessages on mount; fires onRunEnd
        ├── MessageList                ← useCopilotChatInternal + useCoAgent
        └── MessageInput               ← useCopilotChatInternal.sendMessage
```

### Hooks in play

- **`useCopilotChatInternal()`** — gives `messages`, `sendMessage`, `setMessages`. We use the "internal" variant because the OSS-public `useCopilotChat` strips `messages` and `sendMessage` (those are reserved for Cloud users). `Internal` is exported and stable; pin the CopilotKit version.
- **`useCoAgent({ name: 'rag', initialState })`** — gives `state` (our `retrieved_paragraphs`), `running` (true→false transition signals run-end). `running` drives both the input's disabled state and `ChatThreadController`'s session-refresh trigger.
- **`HttpAgent` from `@ag-ui/client`** — POSTs `RunAgentInput` to `/api/chat/agent` and parses the SSE stream. Constructed once per session in `CopilotProvider` with `threadId={currentSession.id}` so the agent's `threadId` round-trips correctly to the backend.

### Turn-based rendering

`MessageList` doesn't iterate the raw AG-UI messages array. It folds them into `Turn[]` via `groupIntoTurns(messages)`: every `UserMessage` starts a new turn; subsequent `AssistantMessage`s contribute text and `toolCalls`; `ToolMessage`s match into the turn's tool list by `toolCallId`. This makes the visual structure stable regardless of how AG-UI splits/consolidates assistant messages across mid-run vs end-of-run.

Per turn the renderer emits:
- User bubble (right, user avatar)
- Tool-call chip stack (left, no avatar — chips clearly bot output by styling alone)
- Either the assistant text bubble (left, bot avatar, markdown via `react-markdown` + `remark-gfm`, citation chips from `state.retrieved_paragraphs` on the latest turn) **or** a "Thinking…" indicator (same position, only shown on the latest turn while `running` and `assistantText` is empty)

### Citations

Live citations come from `useCoAgent`'s `state.retrieved_paragraphs` and render under the latest turn's assistant bubble using the unified compact format `[B X, Ch Y, p. Z]` (same format the backend uses for chip labels, tool excerpts fed to the LLM, and inline-citation prompt instructions). Historical chips on previously-completed turns are a known gap — addressing that would require attaching citations to each message rather than just the latest one (AG-UI's message types don't have a metadata slot).

### Provider config (CopilotProvider)

- `runtimeUrl` is set to a stub URL even though we use `selfManagedAgents` — CopilotKit's prop validator requires `runtimeUrl` / `publicApiKey` / `publicLicenseKey` to be set. With `selfManagedAgents`, the named agent takes precedence at runtime, so the stub URL is a dead-letter for validation only.
- `<CopilotKit key={threadId}>` — re-keying on session switch unmounts and remounts the subtree, giving `ChatThreadController` a fresh `hydrated` ref so it hydrates exactly once per session.

### Session state (useChat)

TanStack Query owns server state — same pattern as the KG page. Local `useState` for `currentSession` (UI choice) and `dismissed` (most recently dismissed Error ref for snackbar UX).

```ts
const {
  currentSession,          // useState
  sessions,                // useQuery(['sessions'])
  historicalMessages,      // useQuery(['session-messages', id])
  sessionsLoaded,          // sessionsQuery.isFetched
  isLoading,               // createMutation.isPending || switchMutation.isPending
  error,                   // derived; null if user already dismissed this Error ref
  onRunEnd,                // invalidates sessions + active session's messages
  createSession,           // useMutation
  switchToSession,         // useMutation (pre-fetches messages, then setCurrentSession)
  clearError,
} = useChat();
```

`onRunEnd` is wired to `ChatThreadController`. Its messages-invalidation matters because a finished turn means Weaviate now has new persisted messages; re-entering the session later should re-fetch fresh history rather than serve a stale cache.

### Error UX

Snackbar shows a derived error message — first non-null among the four query/mutation errors. The `dismissed` state tracks the user's last dismissal *by Error reference*; a new failure produces a new Error reference and the snackbar re-appears. This sidesteps the "useEffect that copies state into state" antipattern from React's *You Might Not Need an Effect* guide.

---

## REST surface (`services/agentAPI.ts`)

Used only for the non-streaming parts: create/list/delete session, fetch persisted history. Live chat goes through CopilotKit's HttpAgent.

```ts
agentAPI.createSession(request)
agentAPI.getSessions(limit)
agentAPI.getSessionMessages(sessionId)
agentAPI.healthCheck()
```

All hit `/api/chat/*`. Base URL configurable via `REACT_APP_API_URL` (default `http://localhost:8000`).

---

## Types (`types/index.ts`)

Mirrors the backend Pydantic models in `src/history_book/api/models/chat_models.py`. Keep them in sync:

```ts
SessionCreateRequest, SessionResponse, SessionListResponse
MessageResponse, MessageListResponse
BookResponse, ChapterResponse, ParagraphResponse, ...
```

Chat-send types (`MessageRequest`, `ChatResponse`, `ChatState`) were removed when the synchronous chat endpoint was retired in favor of AG-UI.

---

## KG Explorer (`/kg`)

Interactive force-directed graph visualization of the knowledge graph. (Unchanged by the chat overhaul.)

### Architecture

- **`KGPage.tsx`** — owns `displayGraph` memo (occurrence filter + recursive leaf trim); passes filtered data to `ForceGraphPanel` and raw graph stats to `EntityPanel`.
- **`ForceGraphPanel.tsx`** — wraps `react-force-graph-2d`. Custom canvas node rendering with focus highlighting; first-order neighbors of focused entity opaque, others dimmed.
- **`EntityPanel.tsx`** — unfocused → graph stats; focused → entity name, type, aliases, descriptions, relationship list with direction icons, clickable neighbors, source citations.
- **`KGTopBar.tsx`** — graph selector (volume/book/chapter grouped + sorted, titles from `/api/books`), entity search, N-hop/Full toggle, hop count, occurrence threshold slider (1–4), leaf-trim toggle.

### Redux state (`graphSlice`)

| Field | Default | Purpose |
|-------|---------|---------|
| `focusEntityId` | `null` | Selected node; drives subgraph fetch and panel |
| `graphName` | `'volume_full'` | Active graph scope |
| `displayMode` | `'full'` | `'full'` = whole graph, `'nhop'` = ego subgraph |
| `hopCount` | `2` | Ego radius for N-hop mode |
| `occurrenceThreshold` | `2` | Filter nodes with count < threshold |
| `trimLeaves` | `false` | Recursively remove degree-≤1 nodes after threshold filter |
| `searchResults` | `[]` | Current search dropdown results |

`clearFocus` resets `displayMode` → `'full'`. `setGraphName` clears focus.

### Implementation notes

- **Link mutation**: `react-force-graph-2d` mutates `link.source`/`link.target` from string IDs to node objects during simulation. Use `linkNodeId(endpoint)` to resolve IDs from links — direct string comparison fails post-simulation.
- **Leaf trimming**: `trimLeavesRecursive` in `KGPage` uses degree `<= 1` (catches isolated nodes too). Focused entity is always protected from removal.
- **Graph dropdown filtering**: Book-type graphs filtered with `/^book\d+$/` to exclude partial merge artifacts.

---

## Book Browsing (`/book`)

Read chapter content directly in the browser. Eliminates needing the physical book during chat. (Unchanged by the chat overhaul.)

- **Cascading selection**: Book → Chapter dropdown
- **URL-based deep linking**: `/book/:bookIndex/:chapterIndex`
- **Scroll position persistence** via `localStorage` (debounced 300ms; max 10 saved positions, oldest evicted)
- **Page numbers** in left margin for citation reference

### Components

- **`BookSelector`** — cascading dropdowns (book then chapter); loads from `/api/books` and `/api/books/{idx}/chapters`.
- **`ChapterView`** — paragraph layout with 60px left-margin page numbers; loaded from `/api/books/{idx}/chapters/{idx}`.
- **`BookPage`** — owns URL params, content load, scroll-position effect.

---

## Provider stack (App.tsx)

```tsx
<Provider store={store}>
  <QueryClientProvider client={queryClient}>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>...routes...</BrowserRouter>
    </ThemeProvider>
  </QueryClientProvider>
</Provider>
```

CopilotKit is scoped to the chat page only via `<CopilotProvider>` inside `ChatPage`, not lifted to `App.tsx`.

---

## Environment

`.env` in `frontend/`:

```bash
REACT_APP_API_URL=http://localhost:8000
```

Changes require restarting `npm start`.

---

## Dev workflow

1. Start backend: `PYTHONPATH=src poetry run uvicorn src.history_book.api.main:app --reload --port 8000`
2. Start frontend: `cd frontend && npm start`
3. Open http://localhost:3000

Both auto-reload on file changes.

CopilotKit's transitive deps produce ~50 source-map warnings on every build (`@0no-co/graphql.web`, `@a2ui/web_core`, `rehype-harden`, `@urql/core` ship bundles whose `//# sourceMappingURL` points at `.ts` sources not published to npm). Harmless; ignore. Suppressible via `GENERATE_SOURCEMAP=false` in `.env` if they get noisy, at the cost of debugging into our own source.

---

## Build

```bash
cd frontend && npm run build
```

Outputs `build/` with optimized static files. Serve with any static server (`npx serve -s build` for local testing).

---

## Related backend files

- `src/history_book/api/routes/agent.py` — AG-UI streaming endpoint (the one HttpAgent talks to)
- `src/history_book/api/routes/chat.py` — session CRUD + history fetch
- `src/history_book/api/routes/books.py` — book browsing
- `src/history_book/api/routes/kg.py` — KG explorer
- `src/history_book/services/agents/CLAUDE.md` — LangGraph agent + AG-UI wrapper
- Root `CLAUDE.md` — high-level architecture
