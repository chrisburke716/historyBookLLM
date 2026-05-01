# Agent System

LangGraph-based RAG agent using modern v1 primitives.

## Architecture

```
START → agent_node → tools_node → agent_node → ... → END
                   ↘ END (no tool calls, or iteration cap reached)
```

### Key Files

| File | Purpose |
|------|---------|
| `rag_agent.py` | `build_rag_agent(enabled_tools=None)` factory — compiles the `StateGraph` with a configurable tool set |
| `context.py` | `AgentContext` dataclass — runtime config passed via `context=` |
| `state.py` | `AgentState(MessagesState)` + `add_paragraphs` reducer |
| `prompts.py` | Prompt assembly — `build_system_prompt(enabled_tool_names)` + cross-tool notes |
| `tools/registry.py` | `TOOL_REGISTRY` + `resolve_tools(enabled)` |
| `tools/_format.py` | Shared formatters: year ranges, relationship summaries |
| `tools/book_search.py` | `search_book` — vector text search |
| `tools/search_entities.py` | `search_entities` — KG entity hybrid search |
| `tools/entity_detail.py` | `get_entity_detail` — entity + 1-hop relationships, prominence-sorted |
| `tools/entity_neighborhood.py` | `get_entity_neighborhood` — N-hop ego subgraph |
| `tools/get_paragraphs.py` | `get_paragraphs` — batch paragraph lookup by ID |
| `tools/relationships_by_period.py` | `query_relationships_by_period` — temporal slice |

### AgentContext

```python
@dataclass
class AgentContext:
    llm_config: LLMConfig
    repository_manager: BookRepositoryManager
    kg_service: KGService
    volume_graph_name: str | None = None     # resolved at ChatService init
    max_tool_iterations: int = 3
    tool_max_results: int = 40
    tool_min_similarity: float = 0.4
```

`volume_graph_name` is resolved once in `ChatService._resolve_volume_graph_name()`:
prefer `graph_type == "volume"`; fall back to the largest `book` graph; warn and
set `None` if neither exists.

### AgentState

```python
class AgentState(MessagesState):
    retrieved_paragraphs: Annotated[list[Paragraph], add_paragraphs]
```

`add_paragraphs` deduplicates by `(book_index, chapter_index, paragraph_index)`.
`search_book` and `get_paragraphs` populate this so citations track across turns.

### Tool registry & enabling subsets

```python
from history_book.services.agents.tools import TOOL_REGISTRY, resolve_tools

resolve_tools(None)                                   # all tools
resolve_tools(["search_book", "search_entities"])     # subset
```

`build_rag_agent(enabled_tools=...)` and `ChatService(enabled_tools=...)` both
forward the same arg. Useful for A/B comparing a baseline (text-only) against
the full KG-aware toolset in evals.

### Prompt assembly

`prompts.build_system_prompt(enabled_tool_names: set[str])` returns a prompt
composed of:
1. **Preamble** — assistant role, attribution, "tool docstrings explain when to use each."
2. **Tool selection notes** (optional) — only the entries in `CROSS_TOOL_NOTES`
   whose key set is fully enabled (e.g., the `search_book ↔ search_entities`
   note only appears if both are bound). Per-tool guidance lives in docstrings.
3. **Workflow** — search → review → synthesize.
4. **Rules** — citation format, no training-data leakage, IDK fallback.

`FINAL_ITERATION_SUFFIX` is appended when the iteration cap is reached.

### Tool pattern (`Command` return)

```python
@tool
def search_book(query: str, runtime: ToolRuntime[AgentContext]) -> Command:
    results = runtime.context.repository_manager.paragraphs.similarity_search_by_text(...)
    paragraphs = [p for p, _ in results]
    return Command(update={
        "messages": [ToolMessage(content=formatted, tool_call_id=runtime.tool_call_id)],
        "retrieved_paragraphs": paragraphs,
    })
```

### Iteration cap

Derived from message history on each agent turn:
```python
iterations = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
is_final = iterations >= ctx.max_tool_iterations
```

When `is_final`, the LLM is invoked without bound tools → forced final answer.

### Streaming

```python
async for mode, data in agent.astream(inputs, context=ctx, config=cfg,
                                       stream_mode=["updates", "messages"]):
    if mode == "messages":
        token, _meta = data
        if token.content: yield token.content
    elif mode == "updates" and "tools" in data:
        retrieved.extend(data["tools"].get("retrieved_paragraphs", []))
```

## KG tool playbook

The expected agent flow for entity-grounded questions:

1. `search_entities("Charlemagne")` → list of candidate entity IDs
2. `get_entity_detail(<id>)` → descriptions, relationships, `source_paragraph_ids`
3. (optional) `get_entity_neighborhood(<id>, hops=2)` → cluster context
4. `get_paragraphs([...source_paragraph_ids])` → original passages for citations

For era-scoped questions:
- `query_relationships_by_period(start_year, end_year)` → time-window slice;
  pick interesting entities and follow up with `get_entity_detail`.

All KG tools are pinned to `ctx.volume_graph_name` (resolved once per service).

## Future: create_agent Migration

The graph uses the same state schema and tool shape as `create_agent`:
```python
from langgraph.prebuilt import create_agent
graph = create_agent(
    model=llm,
    tools=resolve_tools(enabled_tools),
    state_schema=AgentState,
    context_schema=AgentContext,
)
```
