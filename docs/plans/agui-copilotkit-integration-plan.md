# AG-UI + CopilotKit Integration Plan

A handoff document for Claude Code. This focuses on requirements and decisions, not file-by-file implementation — Claude Code will figure out where things go in the existing codebase.

---

## Context

Existing application:

- **Backend:** Python + FastAPI, with a LangGraph v1.x agent (human input → AI output with streamed tokens, plus intermediate tool call messages)
- **Frontend:** React + MUI + Redux, part of a larger app (knowledge graph explorer is one page; the chat will be another surface within the same app)
- **Goal:** Wire the existing LangGraph agent up to a React chat UI using the AG-UI protocol with CopilotKit, prioritizing correct display of streamed messages.

The user is strong on Python/backend, learning React. Bias toward backend-side complexity and a thin, predictable frontend.

---

## Architectural decisions (already made — do not relitigate)

1. **Protocol:** AG-UI (event-driven streaming protocol for agent ↔ UI communication)
2. **Adapter library on backend:** CopilotKit's `LangGraphAGUIAgent` to wrap the existing LangGraph graph and emit AG-UI events
3. **Transport:** Server-Sent Events over a single FastAPI endpoint
4. **Frontend integration:** CopilotKit's React hooks/components on top of MUI; AG-UI events drive rendering
5. **State:** Reuse the existing Redux store for app-level concerns; let CopilotKit own conversational state. Do not duplicate streaming message state into Redux — that's been a documented source of bugs.

---

## High-level requirements

### Backend

**R1. Wrap the existing LangGraph graph with `LangGraphAGUIAgent` without modifying graph internals.**
The existing agent's nodes, state schema, and edges should not need to change. The wrapper goes around the compiled graph.

**R2. Expose a single streaming AG-UI endpoint on the existing FastAPI app.**

- Standard pattern: `POST /agent` (or similar), accepting `RunAgentInput`, returning a `StreamingResponse` of SSE-encoded AG-UI events
- Use CopilotKit's `EventEncoder` with the request's `Accept` header so content-type negotiation works correctly
- Mount on the existing FastAPI app — don't create a separate service
- Confirm CORS is configured for the React frontend's origin

**R3. Use a checkpointer with `thread_id`.**

- If one isn't already wired up, add one (Memory for dev, Postgres for production-readiness later)
- `thread_id` should be passable from the frontend so conversations are resumable
- This is also what enables HITL/interrupts later if needed

**R4. Surface intermediate progress from inside nodes.**

- For any node doing meaningful work (tool calls, retrieval, computation), use LangGraph's `get_stream_writer()` to emit structured progress events
- These should flow through AG-UI as state/custom events that the frontend can render as "thinking" or "tool running" indicators
- Don't pollute the `messages` state with progress — it goes through the side channel

**R5. Typed error frames.**

- Wrap the event generator in try/except
- On exception, yield a structured AG-UI error event before closing the stream
- Never let an unhandled exception just kill the SSE connection — the frontend will show a generic failure with no actionable info
- Errors should include: error code, user-facing message, optional retry hint

**R6. Authentication and rate limiting.**

- Match the existing pattern (shared passcode + `slowapi` setup)
- Apply to the new agent endpoint
- If passcode auth is a header, make sure it's part of the CORS allowlist

### Frontend

**R7. Install and configure CopilotKit.**

- Add `@copilotkit/react-core` and `@copilotkit/react-ui` (or just `react-core` if building custom UI on MUI)
- Wrap the relevant subtree in `<CopilotKit>` provider with the runtime URL pointing at the FastAPI endpoint
- The provider should be scoped — don't wrap the whole app if chat is one page; wrap just the chat page/route

**R8. Build the chat UI surface.**

Decide between two paths and pick one:

- **(a) Use CopilotKit's prebuilt `<CopilotChat>` / `<CopilotSidebar>` components** — fastest path, themeable but may not match MUI perfectly
- **(b) Build custom MUI components** using `useCoAgent` / `useCoAgentStateRender` / `useCopilotAction` hooks — more work, fits the existing MUI design language

Recommend (a) for v1 unless design constraints rule it out. The user is learning React; prebuilt components are the right starting point. They can be swapped for custom MUI components later without changing the backend.

**R9. Render tool calls as first-class UI elements.**

- Each tool call should render as a distinct visual element inside the assistant message stream (collapsible card, chip, or inline block — match the existing app's design language)
- Three visible states: *calling* (spinner + tool name), *running* (streamed arguments visible), *done* (result summary, expandable)
- AG-UI streams tool call arguments incrementally — the UI should reflect this, not wait for the call to complete

**R10. Throttle token rendering.**

- Token-by-token re-renders at full speed are expensive and can cause jank, especially with markdown rendering
- Throttle text-part renders to ~30–60ms
- CopilotKit may handle this; if not, add it at the rendering layer

**R11. Error display.**

- Map AG-UI error events to user-visible inline error states inside the message stream — not toasts, not modal dialogs
- The user should see *which* turn failed and have a retry affordance

**R12. Thread management (v1 scope decision).**

- Decide explicitly: does v1 support multiple threads (sidebar with history), or single-thread-per-session?
- Recommend single-thread-per-session for v1 to limit scope. Multi-thread can be added later — the `thread_id` plumbing already supports it.

---

## Out of scope for v1 (explicit non-goals)

State these clearly so Claude Code doesn't go building them:

- Multi-thread sidebar / conversation history UI
- Human-in-the-loop interrupts (the checkpointer enables this; UI for it is later)
- Generative UI / A2UI components (rendering agent-generated UI widgets)
- Voice / multimodal input
- Message editing or branching
- Persistent conversation storage beyond what the checkpointer gives you (e.g., no separate "conversations" table yet)

---

## Suggested implementation order

A sensible sequence — Claude Code can adjust:

1. **Backend wrapper + endpoint, no frontend yet.** Verify with `curl` that the SSE stream produces well-formed AG-UI events for a simple agent run.
2. **Checkpointer wired in,** verify `thread_id` round-trips work.
3. **Minimal CopilotKit frontend** using prebuilt `<CopilotChat>` against the new endpoint. Goal: get one full happy-path turn rendering correctly, including a tool call.
4. **Progress events from `get_stream_writer()`** flowing through to UI as visible "thinking" indicators.
5. **Error frame handling** end-to-end — force an exception in a node, confirm it renders as an inline error.
6. **Polish:** throttling, MUI theming, auth integration, CORS, rate limiting.

Each step should be independently verifiable before moving to the next. Don't stack all of this into one giant change.

---

## Questions to resolve before starting

Things to ask the user (or confirm by reading the codebase) rather than guess:

- Where does the existing FastAPI app live? Is the agent already mounted as a router, or is it called inline somewhere?
- What's the current state schema of the LangGraph graph? (Specifically: is there a `messages` key using `add_messages`, or a custom structure?)
- What's the existing auth pattern's exact shape (header name, middleware vs dependency)?
- What MUI theme tokens / design system constraints exist that the chat UI needs to respect?
- Is the chat its own route, or embedded as a panel within an existing page?

---

## Reference resources

- AG-UI protocol: https://docs.ag-ui.com
- CopilotKit + LangGraph: https://docs.copilotkit.ai
- LangGraph streaming docs: https://docs.langchain.com/oss/python/langgraph/streaming
- Reference repo pattern: https://www.lifewithdata.org/blog/agui-fastapi-langgraph (the `LangGraphAGUIAgent` + FastAPI endpoint pattern is exactly what R1/R2 describe)
