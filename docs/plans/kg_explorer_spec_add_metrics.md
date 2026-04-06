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
┌──────────────────────────────────────────────────────────────────┐
│                           Top bar                                │
│  [Search________] [Book▾] [Chapter▾] [Size▾] [Color▾] [Controls]│
├──────────────────────────────────┬───────────────────────────────┤
│                                  │                               │
│          Graph panel             │       Entity panel            │
│       (react-force-graph)        │   (detail / stats / metrics)  │
│                                  │                               │
│                                  │                               │
└──────────────────────────────────┴───────────────────────────────┘
```

- Graph panel and entity panel are **side by side with a draggable resize handle**
- No mobile/responsive requirement at this stage
- The top bar contains: search, scope selectors (book + chapter), node size metric selector, node color metric selector, N-hop toggle, hop count selector, and back/forward history buttons

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

### Redux slice: `metricsSlice`

| Field | Type | Notes |
|---|---|---|
| `nodeSizeMetric` | `NodeSizeMetric` | Currently selected size metric; default `'occurrence_count'` |
| `nodeColorMetric` | `NodeColorMetric \| NodePairMetric` | Currently selected color metric; default `'entity_type'` |
| `nodeSizeParams` | `Record<string, number>` | Parameter overrides for current size metric (e.g. `{ damping: 0.85 }`) |
| `nodeColorParams` | `Record<string, number>` | Parameter overrides for current color metric (e.g. `{ k: 4 }`) |

Metric name values are string enums defined to match backend metric identifiers exactly (see §10).

### TanStack Query hooks (keyed off Redux state)

| Hook | Query key | Fires when |
|---|---|---|
| `useGraphQuery(scope)` | `['graph', scope]` | Scope changes |
| `useSubgraphQuery(entityId, hops, scope)` | `['subgraph', entityId, hops, scope]` | `focusEntityId` changes + `displayMode === 'nhop'` |
| `useEntityQuery(entityId)` | `['entity', entityId]` | `focusEntityId` changes (non-null) |
| `useSearchQuery(query, filters, scope)` | n/a — triggered manually | User submits search (enter key) |
| `useBooksQuery()` | `['books']` | Once on page mount |
| `useGraphMetricsQuery(scope)` | `['metrics', 'graph', scope]` | Scope changes |
| `useNodeMetricQuery(scope, metric, params)` | `['metrics', 'node', scope, metric, params]` | Size or color metric selection changes |
| `useNodePairMetricQuery(scope, focusId, metric, params)` | `['metrics', 'node-pair', scope, focusId, metric, params]` | Focus changes + node-pair color metric selected |

`useSearchQuery` is triggered imperatively (on enter), not reactively, so search results live in Redux rather than TanStack Query cache.

`useNodeMetricQuery` and `useGraphMetricsQuery` use `refetchInterval` polling when the response status is `202 Accepted` (metric still computing on backend). Polling stops when a `200` with data is received.

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

**Clearing focus silently deactivates node-pair color metrics.** If a node-pair metric (e.g. cosine similarity) is active when focus is cleared, the graph falls back to the entity_type default coloring. The `metricsSlice` selection is preserved — it reactivates automatically when a new focus is set.

### Focus + re-center
When focus is set (from any source), the graph panel must call `graphRef.current.centerAt(node.x, node.y, durationMs)` reactively. Implement this as a `useEffect` watching `focusEntityId` that reads the current node position from the simulation and calls `centerAt`.

### Focus history
`focusHistory` tracks entity UUIDs only (not scope changes). Back/forward buttons step through `historyIndex`. When a new focus is set mid-history (i.e. `historyIndex < focusHistory.length - 1`), truncate forward history — standard browser-history semantics. This feature is **lower priority** and can be scaffolded without full implementation initially.

---

## 5. Scope change behavior

On scope change (book or chapter dropdown):
1. `dispatch(setScope(newScope))` + `dispatch(clearFocus())` — these should fire together (single thunk or combined action)
2. TanStack Query detects the new scope key and refetches the graph and graph-level metrics
3. Entity panel drops to the stats/unfocused view
4. `colorMap` should be recomputed from the new graph's entity types
5. All cached node metric query results for the old scope remain in TanStack Query cache; results for the new scope will be fetched fresh on demand

---

## 6. Graph panel

### Library
`react-force-graph-2d`. The component accepts `graphData={{ nodes, links }}` directly matching the API response shape. The library mutates node objects in place during simulation, adding `x`, `y`, `vx`, `vy` — this is expected and useful.

### Node rendering — size

Node size is controlled by `nodeSizeMetric` from Redux. The pipeline:

1. If `nodeSizeMetric === 'occurrence_count'`: use `node.occurrence_count` directly (original behavior)
2. Otherwise: look up the node's value from the active `NodeMetricResponse.values` dict by UUID, then normalize using the response's `norm_min` / `norm_max` bounds

All size computation lives in a pure utility: `nodeDisplaySize(rawValue, normMin, normMax): number`. The backend always returns `norm_min` and `norm_max` so the frontend never has to compute them across the node set.

**Loading fallback:** while a metric is still loading (TanStack Query `isLoading`, or `status: 'computing'` poll not yet resolved), fall back to `occurrence_count` sizing. No broken or blank state.

### Node rendering — color

Node color is controlled by `nodeColorMetric`. Three cases:

1. **`'entity_type'` (default):** `colorMap[node.entity_type]` — original behavior
2. **Node-level color metric:** look up value from `NodeMetricResponse.values` by UUID, map through a color scale using `norm_min`/`norm_max`
3. **Node-pair color metric (only active when `focusEntityId` is non-null):** look up value from `NodePairMetricResponse.values`, map through a color scale

For **community membership** (categorical): assign distinct hues from a fixed palette by community ID. Palette assignment must be deterministic — sort community IDs numerically, assign palette colors in order — so colors don't shuffle on re-render.

For **continuous metrics**: use a color scale appropriate to the metric. The exact scale choices are deferred — implement as a utility `metricToColor(value, normMin, normMax, scaleType): string` so the scale can be swapped without touching rendering logic.

**Loading fallback:** while a metric is loading, fall back to `'entity_type'` coloring.

**Metric values always come from the full-scope graph, regardless of display mode.** When the graph panel is showing an N-hop subgraph, it looks up each displayed node's metric value from the full-graph metric response by UUID. The frontend never needs to request or hold the full graph data structure itself for this — only the `values` dict from the metric response.

### Edge rendering
- Directed arrows (the library supports this natively)
- Hover → show tooltip with `link.relationship_type` and `link.description`
- First-order edges from focus node are visually highlighted (already implemented)
- Relationship type text on edges: deferred

### Node interactions
- **Hover:** show tooltip with `node.aliases` (if any), plus current metric value if a non-default metric is active
- **Click:** `dispatch(setFocus(node.id))`
- **Click canvas background:** `dispatch(clearFocus())`
- Standard pan, drag-node, zoom built into the library
- Leaf node trimming toggle already implemented

### Position persistence (future / nice-to-have)
Not required at launch. Stub the position cache (`Map<nodeId, {x, y}>`) so it's easy to wire up later.

---

## 7. Entity panel

### Unfocused state

Two sections:

**Graph-level metrics** (from `useGraphMetricsQuery`): show all fields from `GraphMetricsResponse`. Show a loading skeleton while computing. This is the primary content of the unfocused panel — it replaces the "graph stats stub" from the previous spec version.

**Node/edge counts**: `node_count` and `edge_count` from `GraphResponse`, always available immediately.

### Focused state
Entity card:
- Name + entity type badge
- Aliases
- Description
- Relationships list — direction, type, description, other entity label as a clickable link → full focus

### Unfocus button
Visible only when `focusEntityId` is non-null. Dispatches `clearFocus()`.

---

## 8. Top bar components

### Search
- Fires on **enter key**
- Dropdown of results while input is focused and `searchResults` is non-empty
- Results persist in Redux until new search
- Selecting a result: `dispatch(setFocus(result.id))` + `dispatch(setDisplayMode('nhop'))`
- 5–10 results, no pagination initially
- Filters: entity type (multi-select), relationship type (multi-select)

### Scope selectors
- Dropdown 1: Book ("Full volume" + book titles)
- Dropdown 2: Chapter (disabled when "Full volume" selected)
- On change: combined scope-change + clear-focus action (§5)

### Node size metric selector
Options:
- Occurrence count *(default)*
- Degree centrality
- Betweenness centrality
- PageRank *(parametric: damping factor, default 0.85)*
- Closeness centrality
- K-core number

### Node color metric selector
Two groups:

**Always available:**
- Entity type *(default)*
- Community — Leiden
- Community — Girvan-Newman
- Community — Label Propagation
- Community — Spectral *(parametric: k, number of clusters)*
- Local clustering coefficient
- K-core number

**Focus-relative** (grayed out with tooltip "select a focus node first" when `focusEntityId` is null):
- Cosine similarity
- Jaccard similarity
- Adamic-Adar index
- Common neighbor count
- Shortest path length
- Resistance distance

When focus is cleared while a focus-relative metric is selected, the color display falls back to `'entity_type'`. The dropdown selection is preserved in Redux and reactivates on next focus.

### Parameter inputs
Shown inline in the top bar when a parametric metric is active (small number input adjacent to the metric dropdown).

- **Apply triggers:** enter key OR blur (tab away) — consistent with search UX
- Each `(scope, metric, params)` combination is a separate TanStack Query cache entry and a separate server-side cache entry
- Show a loading indicator on the metric selector while the new param combination is computing

### Controls
- N-hop toggle
- Hop count (1/2/3), grayed out when N-hop off, default **2**
- Back/Forward buttons (lower priority)

---

## 9. Metric computation architecture

### Core principle
All metrics are computed on the **full scope graph**, never on N-hop subgraphs. Metrics are computed **lazily on first request** for a given `(scope, metric, params)` combination, then cached server-side indefinitely (until server restart).

### Server-side cache
In-memory on the FastAPI process. Cache key:
```
(scope_type, book_id, chapter_id, metric_name, **sorted_params)
```
Each param combination is a separate cache entry. No Redis required.

### Async response pattern
For slow metrics (betweenness centrality, some community detection methods):

- Result in cache → `200` with data immediately
- Computation in progress → `202 Accepted` with `{ status: "computing", metric, params }`
- Not yet started → start background task, return `202` immediately

Frontend polls with TanStack Query `refetchInterval` while status is `202`; stops on `200`.

Fast metrics (degree, k-core, Jaccard, Adamic-Adar, common neighbors, shortest path) always return `200` synchronously.

### Graph-level metrics trigger
The first `/graph` call for a scope triggers a background task to compute `GraphMetricsResponse`. The graph endpoint returns immediately; `/metrics/graph` will return `202` briefly then `200` when ready.

---

## 10. DTOs

Frontend TypeScript types should mirror these Pydantic models exactly.

### Shared

```python
class EntityType(str, Enum):
    # Values to match KG schema entity types
    pass

class ScopeType(str, Enum):
    VOLUME = "volume"
    BOOK = "book"
    CHAPTER = "chapter"

class Scope(BaseModel):
    type: ScopeType
    book_id: str | None = None
    chapter_id: str | None = None
```

### Graph response

```python
class GraphNode(BaseModel):
    id: str
    label: str
    entity_type: EntityType
    occurrence_count: int
    aliases: list[str] = []

class GraphLink(BaseModel):
    source: str          # entity UUID
    target: str          # entity UUID
    relationship_type: str
    description: str = ""

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]
    scope: Scope
    node_count: int
    edge_count: int
```

### Entity detail

```python
class RelationshipSummary(BaseModel):
    relationship_id: str
    relationship_type: str
    description: str
    direction: Literal["outgoing", "incoming"]
    other_entity_id: str
    other_entity_label: str    # denormalized

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
    score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
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

### Graph-level metrics

```python
class GraphMetricsResponse(BaseModel):
    scope: Scope
    density: float
    giant_component_ratio: float
    num_connected_components: int
    avg_shortest_path_length: float | None  # None if too large to compute
    diameter: int | None                    # None if too large to compute
    global_clustering_coefficient: float
    num_communities: int                    # from default community detection method
    articulation_point_count: int
    status: Literal["ready", "computing"]
```

### Node-level metrics

```python
class NodeMetricResponse(BaseModel):
    scope: Scope
    metric: str
    params: dict[str, float] = {}
    values: dict[str, float]       # entity UUID → metric value
    norm_min: float
    norm_max: float
    status: Literal["ready", "computing"]

class CommunityMetricResponse(BaseModel):
    """For categorical community membership metrics."""
    scope: Scope
    metric: str
    params: dict[str, float] = {}
    values: dict[str, int]         # entity UUID → community ID
    num_communities: int
    status: Literal["ready", "computing"]
```

### Node-pair metrics

```python
class NodePairMetricResponse(BaseModel):
    scope: Scope
    focus_entity_id: str
    metric: str
    params: dict[str, float] = {}
    values: dict[str, float]       # entity UUID → value relative to focus node
    norm_min: float
    norm_max: float
    # No status field — all node-pair metrics return synchronously
```

### Metric name enums

```python
class NodeSizeMetric(str, Enum):
    OCCURRENCE_COUNT = "occurrence_count"
    DEGREE_CENTRALITY = "degree_centrality"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    PAGERANK = "pagerank"                       # params: damping (default 0.85)
    CLOSENESS_CENTRALITY = "closeness_centrality"
    KCORE_NUMBER = "kcore_number"

class NodeColorMetric(str, Enum):
    ENTITY_TYPE = "entity_type"
    COMMUNITY_LEIDEN = "community_leiden"
    COMMUNITY_GIRVAN_NEWMAN = "community_girvan_newman"
    COMMUNITY_LABEL_PROPAGATION = "community_label_propagation"
    COMMUNITY_SPECTRAL = "community_spectral"   # params: k (number of clusters)
    LOCAL_CLUSTERING_COEFFICIENT = "local_clustering_coefficient"
    KCORE_NUMBER = "kcore_number"

class NodePairMetric(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"     # backend fetches Weaviate vectors; never shipped to frontend
    JACCARD_SIMILARITY = "jaccard_similarity"
    ADAMIC_ADAR = "adamic_adar"
    COMMON_NEIGHBOR_COUNT = "common_neighbor_count"
    SHORTEST_PATH_LENGTH = "shortest_path_length"
    RESISTANCE_DISTANCE = "resistance_distance"
```

---

## 11. API contract

All endpoints under `/api/v1/`.

### Existing endpoints (unchanged)

```
GET  /api/v1/graph
     ?scope_type=volume|book|chapter  &book_id=<uuid>  &chapter_id=<uuid>
     → GraphResponse
     Side effect: triggers background computation of GraphMetricsResponse for this scope

GET  /api/v1/graph/subgraph
     ?entity_id=<uuid>  &hops=1|2|3
     ?scope_type=...  &book_id=<uuid>  &chapter_id=<uuid>
     → GraphResponse

GET  /api/v1/entities/{entity_id}
     → EntityDetail

POST /api/v1/search
     body: { query: str, filters: SearchFilters, scope: Scope, limit: int = 10 }
     → SearchResponse

GET  /api/v1/books
     → BooksResponse
```

### New metric endpoints

```
GET  /api/v1/metrics/graph
     ?scope_type=...  &book_id=<uuid>  &chapter_id=<uuid>
     → GraphMetricsResponse
     202 if background task not yet complete

GET  /api/v1/metrics/node
     ?scope_type=...  &book_id=<uuid>  &chapter_id=<uuid>
     ?metric=<NodeSizeMetric | NodeColorMetric>
     ?damping=0.85        (PageRank only)
     ?k=4                 (spectral clustering only)
     → NodeMetricResponse | CommunityMetricResponse
     202 if not yet cached; cache key: (scope, metric, **params)

GET  /api/v1/metrics/node-pair
     ?scope_type=...  &book_id=<uuid>  &chapter_id=<uuid>
     ?focus_entity_id=<uuid>
     ?metric=<NodePairMetric>
     → NodePairMetricResponse
     Always 200 synchronously
```

---

## 12. Key design decisions (summary)

| Decision | Choice | Rationale |
|---|---|---|
| State management | Redux (UI) + TanStack Query (server) | Complementary. Redux owns user choices; TQ owns fetched data and caching. |
| Graph library | `react-force-graph-2d` | Most mature 2D force graph lib for React. |
| Graph building | Backend-only | All assembly, N-hop traversal, preprocessing in Python/NetworkX. |
| Subgraph response shape | Same as full graph (`GraphResponse`) | Component doesn't need to know which it received. |
| `occurrence_count` scaling | Frontend utility function | Display concern — easy to tune without touching backend. |
| `other_entity_label` on `RelationshipSummary` | Denormalized | Avoids N+1 requests in entity panel. |
| `source`/`target` on `GraphLink` | UUIDs (strings) | react-force-graph resolves internally. |
| Scope change | Clears focus, loads new graph | Graphs are pre-built per scope; this is a load, not a filter. |
| Clear focus | Also resets to full-graph mode | N-hop without focus is meaningless. |
| Search results | Redux (not TQ) | Triggered imperatively (enter key), not reactively. |
| Position persistence | Deferred | Stub the cache now; wire up later. |
| Mobile/responsive | Not in scope | Desktop-only. |
| Metric computation scope | Always full scope graph | N-hop subgraph is a display artifact; metrics on it would be misleading. |
| Metric values in N-hop mode | UUID lookup from full-graph metric response | Transparent regardless of display mode; no full graph data needed on frontend. |
| Metric fetch strategy | Independent per-metric requests | Size and color fetched separately; changing one doesn't invalidate the other. |
| Server-side metric cache | In-memory, keyed by `(scope, metric, params)` | Each param combination is a separate entry. No Redis needed. |
| Metric compute timing | Lazy on first request; graph-level as background side effect of `/graph` | No eager startup precompute. |
| Slow metric UX | Progressive enhancement with 202 polling | Graph renders immediately; size/color updates when ready. Falls back to defaults while loading. |
| Cosine similarity | Backend only | Weaviate vectors never shipped to frontend. |
| Parametric metric apply | Enter key or blur | Consistent with search UX; explicit, no debounce needed. |
| Focus-relative metrics when focus cleared | Silent fallback to entity_type; selection preserved | Reactivates automatically on next focus set. |
| Community color palette | Deterministic by sorted community ID | Prevents color shuffling on re-render. |

---

## 13. Out of scope / deferred

- **Node position persistence:** Stub the position cache (`Map<nodeId, {x, y}>`); implement later.
- **Relationship type text on edges:** Start with nothing; add later.
- **Back/forward history:** Scaffold Redux state and buttons; full implementation lower priority.
- **Force simulation customization:** Start with react-force-graph defaults; iterate later.
- **Node size scaling formula:** Implement as `nodeDisplaySize(rawValue, normMin, normMax)` utility; exact formula (sqrt vs log, ceiling) is a runtime tuning decision.
- **Color scale choices for continuous metrics:** Implement `metricToColor(value, normMin, normMax, scaleType)` as a swappable utility; specific scale choices deferred.
- **Richer graph stats in entity panel:** Node/edge counts shown now; degree distribution, most-connected nodes, etc. are a separate workstream.
- **Resistance distance at launch:** Computationally expensive (matrix inversion on full graph). Include in the API contract but Claude Code should assess feasibility against actual graph sizes before implementing.
- **Metric value display in entity panel for node-pair metrics:** Whether to show focused node's self-similarity as a reference point is deferred.
