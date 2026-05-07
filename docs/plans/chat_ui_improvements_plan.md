# Chat UI Improvements

## Context

The chat UI has four rough edges that hurt the experience:

1. **No streaming.** Backend has a working SSE endpoint at `/api/chat/sessions/{id}/stream` (chat.py:168, chat_service.py:211) that emits raw token chunks, but the frontend never calls it — `useChat.ts` waits for the full response before displaying anything. Long answers feel unresponsive.
2. **Tool calls are invisible.** The agent calls KG/book-search tools via LangGraph, but tool events are silently swallowed — `send_message_stream` accumulates `retrieved_paragraphs` from the `updates` stream (chat_service.py:242–244) without ever forwarding them. Users can't see what the agent is doing during the multi-second pause.
3. **Markdown doesn't render.** `MessageList.tsx:30-34` only handles `**bold**` via regex; `#` headers, lists, tables, code, etc. show as raw text. No markdown library is installed.
4. **Citations are inconsistent and incomplete.** Chips show `f"Page {p.page}"` (chat.py:50) — no book or chapter context. Tool excerpt headers use `[Chapter X, Page Y]` (prompts.py:83). Inline-citation prompt instruction uses `[Ch X, p. Y]` (prompts.py:12, 17, 25). Three different formats, none include book number.

Outcome: streamed responses with live tool-progress indicators, properly rendered markdown, and one consistent compact citation format `[B{book}, Ch{chapter}, p.{page}]` used everywhere.

We will execute one step at a time; the user will manually test between steps.

---

## Step 1 — Streaming protocol redesign + frontend hookup

Switch the SSE wire format from raw tokens to typed JSON events so it can carry tool events later. Hook the frontend up to the streaming endpoint.

### Backend

**`src/history_book/services/chat_service.py`** — `send_message_stream` (line 211):
- Yield typed event dicts (or pre-serialized JSON strings) instead of raw token text. Event types this step:
  - `{"type": "token", "text": "..."}` — incremental text
  - `{"type": "done", "citations": [...], "session": {...}}` — terminal event with final citations + updated session (so the frontend can update the title in one shot)
  - `{"type": "error", "message": "..."}`
- Keep tool-event emission stubbed/noop for this step — it's added in Step 2. (Still parse the `updates` stream so `retrieved_paragraphs` accumulates.)
- Build the citations list inside the streamer once retrieval is done (using the new compact format from Step 4 — but for Step 1 keep the existing `f"Page {p.page}"` so changes are isolated).

**`src/history_book/api/routes/chat.py`** — `stream_message` (line 168):
- Replace `yield f"data: {chunk}\n\n"` with `yield f"data: {json.dumps(event)}\n\n"`.
- Forward errors as a `{"type": "error", ...}` event.

### Frontend

**`frontend/src/services/agentAPI.ts`**:
- Add `sendMessageStream(sessionId, content, handlers)` using `fetch` + `ReadableStream` (EventSource doesn't support POST bodies cleanly).
- Parse SSE frames (`data: ...\n\n`), `JSON.parse` payloads, dispatch to `handlers.onToken`, `handlers.onDone`, `handlers.onError`.

**`frontend/src/hooks/useChat.ts`** — `sendMessage` (lines 107–166):
- Switch to streaming path. Append optimistic user message as today. Create an in-progress assistant message in state with empty content; on each `token` event append text; on `done` finalize message (set citations, replace temp id, update session title).
- Keep the non-streaming path unused (or delete after Step 1 verified).

**`frontend/src/components/MessageList.tsx`**:
- Render the in-progress assistant message identically to a normal one — content just grows. No special UI yet.

### Test

- Send a message, see tokens arrive incrementally.
- Confirm citation chips still appear after `done`.
- Confirm session title updates after first message.
- Confirm error path (kill backend mid-stream) surfaces an error in the UI.

---

## Step 2 — Tool messages (live, Perplexity-style)

Surface each tool call as a small pill while the agent runs. Live-only (not persisted).

### Backend

**`src/history_book/services/chat_service.py`** — `send_message_stream`:
- In addition to `messages` mode, watch the agent's `updates` mode for two things:
  - `agent` node updates carrying an `AIMessage` with `tool_calls` → emit `{"type": "tool_start", "id": tool_call_id, "label": "...", "args_summary": "..."}` per tool call
  - `tools` node updates carrying `ToolMessage`s → emit `{"type": "tool_end", "id": tool_call_id, "summary": "..."}` per result
- Tool-call-id matches start↔end.

**New helper module** (e.g. `src/history_book/services/agents/tool_labels.py`):
- Map tool name + args → human-readable label and result summary. Keyed by tool name to keep `chat_service` clean.
  - `search_book(query)` → label `Searching the book for "{query}"`; summary `{n} passages`
  - `search_entities(query, ...)` → label `Searching the knowledge graph for "{query}"`; summary `{n} entities`
  - `get_entity_detail(entity_id)` → label `Looking up entity {short_id_or_name}`; summary `{n} relationships`
  - `get_paragraphs(ids)` → label `Retrieving {n} source paragraph(s)`; summary `{n} paragraphs`
  - `query_relationships_by_period(start, end)` → label `Querying relationships from {start}–{end}`; summary `{n} relationships`
- Pure functions; tested directly.

### Frontend

**`frontend/src/components/ToolStep.tsx`** (new):
- Small pill row: icon + label, spinner while running, checkmark + summary on completion. MUI `Chip` or `Box` with subtle background.

**`frontend/src/components/MessageList.tsx`**:
- Render an array of tool steps above the in-progress assistant message bubble (or stacked just above the answer text). Multiple steps per turn render as a list.

**`frontend/src/hooks/useChat.ts`**:
- Track `toolSteps: Map<id, {label, status, summary}>` on the in-progress message.
- Handle `tool_start` (add) and `tool_end` (mark complete + summary).
- Live-only: discard tool steps when the message finalizes (don't persist them on the saved `MessageResponse`). They will not reappear on reload.

### Test

- Ask a question that triggers KG tools (e.g., "Tell me about Charlemagne").
- Each tool step appears with its label, spins, then resolves with summary.
- Final answer streams underneath.
- Reload the page → tool steps are gone, only the persisted user/assistant messages remain.

---

## Step 3 — Markdown rendering

### Frontend

- `cd frontend && npm install react-markdown remark-gfm`
- **`frontend/src/components/MessageList.tsx`**:
  - Replace the regex `formatMessage` (lines 30-34) with `<ReactMarkdown remarkPlugins={[remarkGfm]}>`.
  - Remove `whiteSpace: 'pre-wrap'` from the assistant `Typography` (markdown owns whitespace now); keep `wordBreak: 'break-word'`.
  - Provide MUI-styled `components` overrides for `h1/h2/h3`, `ul/ol`, `code`/`pre`, `blockquote`, `a` so they render correctly inside the bubble.
  - Apply markdown rendering only to assistant messages; user messages stay plain text (safer against accidental injection and matches typical chat UI).
- No syntax highlighting — overkill for history content.

### Test

- Provoke a response with `#` headers, lists, **bold**, `inline code`, a blockquote.
- Confirm rendering looks reasonable inside both the in-progress and finalized message bubbles.
- Confirm user messages still render as plain text.

---

## Step 4 — Compact citation format, unified across the stack

Target format: `[B3, Ch5, p.42]`. Used for chip labels, tool excerpt headers, and inline-citation prompt instructions.

### Backend

1. **`src/history_book/api/routes/chat.py:50`** — citation chip format:
   ```python
   citations = [
       f"[B{p.book_index}, Ch{p.chapter_index}, p.{p.page}]"
       for p in retrieved_paragraphs
   ]
   ```
   (Also update the equivalent line in `chat_service.py` if Step 1 moved citation construction into the streamer.)

2. **`src/history_book/services/agents/prompts.py:83`** — `format_excerpts_for_llm`:
   ```python
   parts.append(
       f"[B{para.book_index}, Ch{para.chapter_index}, p.{para.page}]\n{para.text}"
   )
   ```

3. **`src/history_book/services/agents/prompts.py`** — instruction text at lines 12, 17, 25: change every `[Ch X, p. Y]` → `[B X, Ch Y, p. Z]` so the agent emits the same compact format inline.

### Test

- Send a fact-based question.
- Inline citations in the streamed answer use `[B X, Ch Y, p. Z]`.
- Citation chips below the message use the matching format.
- Tool excerpt headers seen by the agent (verifiable via LangSmith trace or logs) use the same format.

---

## Critical files

**Backend**
- `src/history_book/services/chat_service.py` — `send_message_stream` (Steps 1, 2, 4)
- `src/history_book/api/routes/chat.py` — SSE wire format + citation construction (Steps 1, 4)
- `src/history_book/services/agents/prompts.py` — tool-output header + inline-citation instructions (Step 4)
- `src/history_book/services/agents/tool_labels.py` *(new)* — per-tool label/summary functions (Step 2)

**Frontend**
- `frontend/src/services/agentAPI.ts` — add `sendMessageStream` (Step 1)
- `frontend/src/hooks/useChat.ts` — switch to streaming, track tool steps (Steps 1, 2)
- `frontend/src/components/MessageList.tsx` — render in-progress message, tool steps, markdown (Steps 1, 2, 3)
- `frontend/src/components/ToolStep.tsx` *(new)* — tool pill component (Step 2)
- `frontend/package.json` — add `react-markdown`, `remark-gfm` (Step 3)

## Order rationale

Step 1 first because Step 2 rides on the typed-event protocol. Steps 3 and 4 are cosmetic and independent of each other; doing them last keeps the streaming/tool work as a clean diff. Each step is a clean commit boundary.

## End-to-end verification (after all four steps)

1. Start backend + frontend (`PYTHONPATH=src poetry run uvicorn ...` + `cd frontend && npm start`).
2. Open http://localhost:3000/chat, create a session, ask: "What can you tell me about Charlemagne and his relationship to the church?"
3. Observe:
   - Tool pills appear in order ("Searching the knowledge graph for 'Charlemagne'" → "Looking up entity …" → "Retrieving N source paragraphs"), each spinning then resolving.
   - Answer text streams in token-by-token below the pills.
   - Markdown headers/lists/bold render correctly.
   - Inline citations look like `[B3, Ch5, p.42]`.
   - Citation chips below match.
4. Reload the page → user/assistant messages persist; tool pills are gone (live-only).
5. Verify a LangSmith trace shows tool excerpt headers in the new format.
