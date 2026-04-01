# Knowledge Graph Explorer — Frontend Architecture Spec

> **Handoff note for Claude Code:** This document captures architecture decisions and requirements agreed upon during planning. Implementation details (exact component file structure, hook implementations, MUI component choices, Redux toolkit boilerplate, etc.) are intentionally left to you. The goal is to preserve the *why* behind each decision so you can implement confidently without needing to re-litigate choices. Where a decision is flagged as "deferred," that means it was explicitly not decided during planning and is yours to make.

---

## 1. Project context

- **Backend:** Python / FastAPI, knowledge graph stored in Weaviate
- **Frontend:** React, MUI component library, Redux for state management
- **New addition:** A single standalone page — the Knowledge Graph Explorer — added alongside existing pages in the project
- **Graph library:** `react-force-graph-2d` (already selected; Cytoscape.js is the fallback if issues arise)

The knowledge graph is hierarchical: separate pre-built graphs exist for each chapter, each book, and the full volume. The frontend does not filter a single graph — it loads a different graph object depending on scope selection.

---

## 2. Page layout

```
┌─────────────────────────────────────────────────────┐
│                     Top bar                         │
│  [Search________________] [Book▾] [Chapter▾] [Controls] │
├──────────────────────────────┬──────────────────────┤
│                              │                      │
│        Graph panel           │   Entity panel       │
│     (react-force-graph)      │  (detail / stats)    │
│                              │                      │
│                              │                      │
└──────────────────────────────┴──────────────────────┘
```

- Graph panel and entity panel are **side by side with a draggable resize handle**
- No mobile/responsive requirement at this stage
- The top bar contains search, scope selectors (book + chapter dropdowns), N-hop toggle, hop count selector, and back/forward history buttons

---

## 3. State management

**Redux** handles all UI state. **TanStack Query** handles all server state (fetching, caching, refetching). They are complementary: Redux owns what the user has *chosen*, TanStack Query owns what the server has *returned*.

### Redux slice: `graphSlice`

| Field | Type | Notes |
|---|---|---|
| `focusEntityId` | `string \| null` | `null` = no focus |
| `scope` | `Scope` | See DTO section |
| `displayMode` | `'full' \| 'nhop'` | Full graph vs N-hop subgraph |
| `hopCount` | `1 \| 2 \| 3` | Default: **2** |
| `focusHistory` | `string[]` | Array of entity UUIDs, for back/forward |
| `historyIndex` | `number` | Current position in focusHistory |
| `searchResults` | `SearchResult[]` | Populated on search submit, cleared on new search |
| `colorMap` | `Record<EntityType, string>` | Entity type → hex color, derived once from graph load |

### TanStack Query hooks (keyed off Redux state)

| Hook | Query key | Fires when |
|---|---|---|
| `useGraphQuery(scope)` | `['graph', scope]` | Scope changes |
| `useSubgraphQuery(entityId, hops, scope)` | `['subgraph', entityId, hops, scope]` | `focusEntityId` changes + `displayMode === 'nhop'` |
| `useEntityQuery(entityId)` | `['entity', entityId]` | `focusEntityId` changes (non-null) |
| `useSearchQuery(query, filters, scope)` | n/a — triggered manually | User submits search (enter key) |
| `useBooksQuery()` | `['books']` | Once on page mount |

`useSearchQuery` is triggered imperatively (on enter), not reactively, so search results live in Redux rather than TanStack Query cache. The query is fired manually and its result dispatched to `searchResults`.

---

## 4. Focus entity behavior

"Focus" is the central interaction concept. It means: *this entity is selected, the entity panel shows its details, and if N-hop mode is on, the graph shows its subgraph.*

### Setting focus
Focus can be set from three places:
1. **Click a node** in the graph panel → `dispatch(setFocus(entityId))`
2. **Click a relationship** in the entity panel → `dispatch(setFocus(otherEntityId))` — this is a *full focus*: updates graph + entity panel + re-centers graph view
3. **Select a search result** → `dispatch(setFocus(entityId))` + `dispatch(setDisplayMode('nhop'))`

### Clearing focus
- Click the canvas background (graph panel)
- Click the "unfocus" button in the entity panel (only visible when focused)

**Clearing focus always resets to full-graph mode.** N-hop mode without a focus entity is meaningless, so `clearFocus()` should set `displayMode` back to `'full'` as part of the same action.

### Focus + re-center
When focus is set (from any source), the graph panel must call `graphRef.current.centerAt(node.x, node.y, durationMs)` reactively. Implement this as a `useEffect` watching `focusEntityId` that reads the current node position from the simulation and calls `centerAt`.

### Focus history
`focusHistory` tracks entity UUIDs only (not scope changes). Back/forward buttons step through `historyIndex`. When a new focus is set mid-history (i.e. `historyIndex < focusHistory.length - 1`), truncate forward history — standard browser-history semantics. This feature is **lower priority** and can be scaffolded without full implementation initially.

---

## 5. Scope change behavior

On scope change (book or chapter dropdown):
1. `dispatch(setScope(newScope))` + `dispatch(clearFocus())` — these should fire together (single thunk or combined action)
2. TanStack Query detects the new scope key and refetches the graph
3. Entity panel drops to the stats/unfocused view
4. `colorMap` should be recomputed from the new graph's entity types

---

## 6. Graph panel

### Library
`react-force-graph-2d`. The component accepts `graphData={{ nodes, links }}` directly matching the API response shape (see DTOs). The library mutates node objects in place during simulation, adding `x`, `y`, `vx`, `vy` — this is expected and useful (positions can be read back for future position-persistence feature).

### Node rendering
- **Size:** derived from `occurrence_count`. Apply sqrt or log scaling with a ceiling. Exact formula deferred — implement as a pure utility function `nodeSize(occurrenceCount: number): number` so it's easy to tune.
- **Color:** looked up from Redux `colorMap` by `node.entity_type`
- **Label:** `node.label` rendered as canvas text on or below the node

### Edge rendering
- Directed arrows (the library supports this natively)
- Hover → show tooltip with `link.relationship_type` and `link.description`
- Whether to render relationship type text on edges is deferred — start with nothing, add later

### Node interactions
- **Hover:** show tooltip with `node.aliases` (if any)
- **Click:** `dispatch(setFocus(node.id))`
- **Click canvas background:** `dispatch(clearFocus())`
- Standard pan, drag-node, and zoom are built into the library

### Display mode switching
When `displayMode` switches to `'nhop'` (and `focusEntityId` is set), `useSubgraphQuery` fires and the graph panel receives a new `graphData` object. When it switches to `'full'`, `useGraphQuery` data is used instead. The component just consumes whichever data is active — the switching logic lives in a selector or the component's data-binding.

### Position persistence (future / nice-to-have)
Not required at launch. When implemented: maintain a `Map<nodeId, {x, y}>` that captures positions from the simulation before a graph reload. On reload, seed matching nodes with their previous `x`/`y` before passing to the component. This prevents the full graph from re-randomizing on every N-hop rebuild. Stub the position cache store now so it's easy to wire up later.

---

## 7. Entity panel

### Unfocused state
Show graph stats: node count, edge count (both available on `GraphResponse`). Additional stats (degree distribution, most-connected entities, etc.) are **deferred** — stub this as a placeholder component.

### Focused state
Display an entity card with:
- Name (entity label) + entity type badge
- Aliases list
- Description
- Relationships list — each item shows: direction (incoming/outgoing), relationship type, description, and the other entity's label as a clickable link → triggers full focus on that entity

### Unfocus button
Visible only when `focusEntityId` is non-null. Clicking dispatches `clearFocus()` (which also resets to full-graph mode).

---

## 8. Top bar components

### Search
- Text input — search fires on **enter key**, not on keystroke
- Results display as a dropdown below the input while the input is focused and `searchResults` is non-empty
- Results persist in Redux until a new search is submitted
- Selecting a result: `dispatch(setFocus(result.id))` + `dispatch(setDisplayMode('nhop'))`
- Display 5–10 results; no pagination initially
- Each result shows: entity label, entity type badge, aliases (abbreviated)
- Dropdown filters: entity type (multi-select), relationship type (multi-select) — these are passed as `SearchFilters` to the search API

### Scope selectors
- Dropdown 1: Book (options: "Full volume" + each book title from `BooksResponse`)
- Dropdown 2: Chapter (disabled/grayed out when "Full volume" is selected; shows chapters for the selected book)
- On change: fires the combined scope-change action described in §5

### Controls
- N-hop toggle (on/off)
- Hop count selector (1 / 2 / 3) — grayed out when N-hop is off; default **2**
- Back / Forward buttons (history navigation) — lower priority

---

## 9. DTOs

These are the agreed Pydantic models for the backend. Frontend TypeScript types should mirror these exactly.

### Shared

```python
class EntityType(str, Enum):
    # Values to match whatever entity types exist in the KG schema
    pass

class ScopeType(str, Enum):
    VOLUME = "volume"
    BOOK = "book"
    CHAPTER = "chapter"

class Scope(BaseModel):
    type: ScopeType
    book_id: str | None = None      # None if type == VOLUME
    chapter_id: str | None = None   # None unless type == CHAPTER
```

### Graph response (used for both full graph and subgraph)

```python
class GraphNode(BaseModel):
    id: str                    # Weaviate UUID — stable key for react-force-graph
    label: str                 # entity text, rendered on node
    entity_type: EntityType    # drives colorMap lookup
    occurrence_count: int      # frontend computes display size (sqrt/log scaling)
    aliases: list[str] = []    # shown on hover tooltip

class GraphLink(BaseModel):
    source: str                # entity UUID (react-force-graph resolves to node object)
    target: str                # entity UUID
    relationship_type: str
    description: str = ""

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]
    scope: Scope
    node_count: int            # for stats panel
    edge_count: int
```

`GraphResponse` is the return type for **both** the full graph and subgraph endpoints. The frontend component does not need to know which it received — it just renders `{ nodes, links }`.

### Entity detail (loaded on focus)

```python
class RelationshipSummary(BaseModel):
    relationship_id: str
    relationship_type: str
    description: str
    direction: Literal["outgoing", "incoming"]
    other_entity_id: str
    other_entity_label: str    # denormalized to avoid N+1 lookups in the entity panel

class EntityDetail(BaseModel):
    id: str
    label: str
    entity_type: EntityType
    aliases: list[str]
    description: str
    occurrence_count: int
    relationships: list[RelationshipSummary]
```

### Search

```python
class SearchFilters(BaseModel):
    entity_types: list[EntityType] = []
    relationship_types: list[str] = []

class SearchResult(BaseModel):
    id: str
    label: str
    entity_type: EntityType
    aliases: list[str]
    score: float               # hybrid search score

class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str                 # echoed back for cache-key matching
```

### Scope / navigation

```python
class ChapterMeta(BaseModel):
    id: str
    title: str
    number: int

class BookMeta(BaseModel):
    id: str
    title: str
    chapters: list[ChapterMeta]

class BooksResponse(BaseModel):
    books: list[BookMeta]
```

---

## 10. API contract

All endpoints under `/api/v1/`. Scope params are repeated on the subgraph endpoint because subgraphs are always scope-bounded (a 2-hop subgraph in chapter 3 differs from the same entity's subgraph in the full volume).

```
GET  /api/v1/graph
     ?scope_type=volume|book|chapter
     ?book_id=<uuid>          (required if scope_type=book or chapter)
     ?chapter_id=<uuid>       (required if scope_type=chapter)
     → GraphResponse

GET  /api/v1/graph/subgraph
     ?entity_id=<uuid>
     ?hops=1|2|3
     ?scope_type=volume|book|chapter
     ?book_id=<uuid>
     ?chapter_id=<uuid>
     → GraphResponse

GET  /api/v1/entities/{entity_id}
     → EntityDetail

POST /api/v1/search
     body: { query: str, filters: SearchFilters, scope: Scope, limit: int = 10 }
     → SearchResponse

GET  /api/v1/books
     → BooksResponse
```

---

## 11. Key design decisions (summary)

| Decision | Choice | Rationale |
|---|---|---|
| State management | Redux (UI) + TanStack Query (server) | Complementary, not competing. Redux owns user choices; TQ owns fetched data and caching. |
| Graph library | `react-force-graph-2d` | Most mature 2D force graph lib for React; good hooks for custom rendering and simulation access. |
| Graph building | Backend-only | All graph assembly, N-hop traversal, and preprocessing done in Python/NetworkX. Frontend receives ready-to-render `{ nodes, links }`. |
| Subgraph response shape | Same as full graph (`GraphResponse`) | Component doesn't need to know — just renders what it receives. |
| `occurrence_count` scaling | Frontend utility function | Scaling (sqrt/log/ceiling) is a display concern — easy to tune without touching the backend. |
| `other_entity_label` on `RelationshipSummary` | Denormalized | Avoids N+1 requests just to render relationship labels in the entity panel. |
| `source`/`target` on `GraphLink` | UUIDs (strings) | react-force-graph resolves string IDs to node objects internally. Keeps JSON clean. |
| Scope change | Clears focus, loads new graph | Graphs are pre-built per scope; this is a graph load, not a filter operation. |
| Clear focus | Also resets to full-graph mode | N-hop without a focus entity is meaningless — always revert together. |
| Search results | Redux (not TQ) | Triggered imperatively (enter key), not reactively. |
| Position persistence | Deferred (nice-to-have) | Stub the position cache now; wire it up in a follow-up. |
| Graph stats in entity panel | Stub for now | Additional stats (degree, centrality, etc.) will be designed separately. |
| Mobile/responsive | Not in scope | Desktop-only for now. |

---

## 12. Out of scope / deferred

- **Graph stats content:** The unfocused entity panel will show node count and edge count. Richer stats (degree distribution, most-connected nodes, centrality measures, etc.) are a separate workstream.
- **Node position persistence across graph rebuilds:** Architecture is stubbed (position cache store), implementation deferred.
- **Relationship type text on edges:** Start with no labels; add later if desired.
- **Back/forward history:** Scaffold the Redux state and buttons; full implementation is lower priority.
- **Visual customization of force simulation:** Start with react-force-graph defaults; iterate later.
- **Node size scaling formula:** Implement as a standalone utility function so it's easy to tune. Exact formula (sqrt vs log, ceiling value) is a runtime decision.
