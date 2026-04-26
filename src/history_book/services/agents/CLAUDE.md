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
| `rag_agent.py` | `build_rag_agent()` factory — compiles the `StateGraph` |
| `context.py` | `AgentContext` dataclass — runtime config passed via `context=` |
| `state.py` | `AgentState(MessagesState)` + `add_paragraphs` reducer |
| `prompts.py` | `AGENT_SYSTEM_PROMPT`, `FINAL_ITERATION_SUFFIX`, `format_excerpts_for_llm` |
| `tools/book_search.py` | `search_book` tool — `ToolRuntime[AgentContext]` + `Command` return |

### AgentContext

```python
@dataclass
class AgentContext:
    llm_config: LLMConfig
    repository_manager: BookRepositoryManager
    max_tool_iterations: int = 3
    tool_max_results: int = 40
    tool_min_similarity: float = 0.4
```

Passed at invoke time: `agent.ainvoke(inputs, context=AgentContext(...), config=...)`.
No re-instantiation inside tools — `runtime.context` provides access.

### AgentState

```python
class AgentState(MessagesState):
    retrieved_paragraphs: Annotated[list[Paragraph], add_paragraphs]
```

`MessagesState` provides `messages: Annotated[list[BaseMessage], add_messages]`.
`add_paragraphs` deduplicates by `(book_index, chapter_index, paragraph_index)`.

### Tool Pattern

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

Tools return `Command` — no JSON parsing needed, paragraphs flow into state via reducer.

### Iteration Cap

Derived from message history on each agent turn:
```python
iterations = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
is_final = iterations >= ctx.max_tool_iterations
```

When `is_final`, LLM invoked without bound tools → forced final answer.

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

Single pass — no second `.ainvoke()` call to recover paragraphs.

## Future Tools

All fit the `ToolRuntime[AgentContext]` shape. Add `kg_service: KGService` to `AgentContext` when needed:
- `get_paragraph(id)` — paragraph by ID lookup
- `search_entities(query)` — KG entity search
- `get_entity_detail(id)` — entity with relationships

## Future: create_agent Migration

The graph uses the same state schema and tool shape as `create_agent`. Migration is:
```python
from langgraph.prebuilt import create_agent
graph = create_agent(
    model=llm,
    tools=TOOLS,
    state_schema=AgentState,
    context_schema=AgentContext,
    # middleware=[ToolCallLimitMiddleware(max_calls=3), ...]
)
```
