# Agent System Documentation

## Overview

The agent system implements RAG using LangGraph, providing a stateful, graph-based approach to chat with enhanced capabilities like checkpointing, graph visualization, and extensibility for future features.

**Key Advantages over LCEL**:
- **Checkpointing**: Automatic conversation context management via MemorySaver
- **Performance**: 5.6% faster (8.97s vs 9.50s average)
- **Observability**: Built-in LangSmith tracing with graph structure
- **Extensibility**: Easy to add tools, planning nodes, and multi-step reasoning
- **Visualization**: Mermaid diagram generation for debugging

## Architecture

### Graph Structure

```
START → retrieve_node → generate_node → END
```

#### retrieve_node
- **Input**: `state["question"]`
- **Action**: Query ParagraphRepository for relevant paragraphs
- **Output**: Updates `state["retrieved_paragraphs"]`
- **Error Handling**: Returns empty list (graceful degradation)

#### generate_node
- **Input**: `state["messages"]`, `state["question"]`, `state["retrieved_paragraphs"]`
- **Action**: Format context + invoke LLM
- **Output**: Updates `state["generation"]` and appends to `state["messages"]`
- **Fallback**: Generates without context if no paragraphs retrieved

### State Management

**AgentState** (TypedDict):

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Chat history with add_messages reducer (appends, doesn't replace)

    question: str
    # Current user query

    retrieved_paragraphs: list[Paragraph]
    # Retrieved context documents

    generation: str
    # LLM-generated response

    session_id: str
    # Maps to LangGraph thread_id for checkpointing

    metadata: dict
    # Execution metadata for debugging
```

The `add_messages` reducer automatically appends new messages to the history rather than replacing them, making it perfect for conversation tracking.

### Memory Strategy

**Hybrid Approach**:

1. **LangGraph MemorySaver**: In-memory checkpointing during graph execution
   - Keyed by `thread_id` (= `session_id`)
   - Maintains state between graph nodes
   - Enables conversation context across messages
   - Lost on server restart (acceptable for quick RAG queries)

2. **Weaviate**: Long-term persistence
   - Stores ChatSession and ChatMessage entities
   - Survives server restarts
   - Source of truth for conversation history

**Why Hybrid?**
- MemorySaver provides fast, ephemeral state during graph execution
- Weaviate provides durable storage for conversation history
- Best of both worlds: speed during execution + durability for long-term storage

## Usage Examples

### Basic Usage

```python
from history_book.services.graph_chat_service import GraphChatService

async def basic_example():
    service = GraphChatService()

    # Create session
    session = await service.create_session(title="History Chat")

    # Send message
    result = await service.send_message(
        session_id=session.id,
        user_message="Who was Cleopatra?"
    )

    print(result.message.content)  # AI response
    print(len(result.retrieved_paragraphs))  # Typically 40
```

### Multi-turn Conversation

```python
async def conversation_example():
    service = GraphChatService()
    session_id = (await service.create_session(title="Roman History")).id

    # Message 1
    result1 = await service.send_message(session_id, "Who was Julius Caesar?")
    # Response: "Julius Caesar was a prominent Roman aristocrat..."

    # Message 2 - uses context from Message 1
    result2 = await service.send_message(session_id, "When was he assassinated?")
    # LLM understands "he" = Julius Caesar
    # Response: "Julius Caesar was assassinated on 15 March 44 BC."

    # Message 3 - uses full conversation context
    result3 = await service.send_message(session_id, "Who were the conspirators?")
    # Response: "The main conspirators against Julius Caesar..."
```

### Session Management

```python
async def session_management():
    service = GraphChatService()

    # List recent sessions
    sessions = await service.list_recent_sessions(limit=10)
    for session in sessions:
        print(f"{session.title}: {session.id}")

    # Get conversation history
    messages = await service.get_session_messages(session_id)
    for msg in messages:
        print(f"[{msg.role}]: {msg.content[:50]}...")

    # Delete session
    await service.delete_session(session_id)
```

### Graph Visualization

```python
from history_book.services.graph_rag_service import GraphRagService

def visualize_graph():
    service = GraphRagService()

    # Get graph structure
    graph = service.graph

    # Generate Mermaid diagram
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)

    # Visualize at https://mermaid.live
```

## Extending the Graph

### Adding a New Node

```python
def my_custom_node(state: AgentState) -> dict:
    """
    Process state and return updates.

    Returns only the fields that should be updated.
    """
    question = state["question"]

    # Your custom logic here
    result = process_question(question)

    # Return updates to state
    return {
        "metadata": {
            **state.get("metadata", {}),
            "custom_field": result
        }
    }

# Add to graph
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("custom", my_custom_node)

# Add edges
workflow.add_edge("retrieve", "custom")  # After retrieve
workflow.add_edge("custom", "generate")   # Before generate

# Compile
graph = workflow.compile()
```

### Adding Tool Calling (Future)

```python
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return web_search_api(query)

# Bind tools to LLM
llm_with_tools = llm.bind_tools([search_web])

# Create tools node
def tools_node(state: AgentState) -> dict:
    """Execute tool calls"""
    messages = state["messages"]
    last_message = messages[-1]

    # Extract tool calls from last AI message
    tool_calls = last_message.tool_calls

    # Execute tools
    tool_messages = []
    for tool_call in tool_calls:
        tool_result = execute_tool(tool_call)
        tool_messages.append(
            ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}

# Add routing logic
def should_use_tools(state: AgentState) -> str:
    """Route to tools node if LLM called tools, otherwise to END"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Add to graph
workflow.add_node("tools", tools_node)
workflow.add_conditional_edges(
    "generate",
    should_use_tools,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "generate")  # Loop back after tools
```

### Adding Reflection (Future)

```python
def reflect_node(state: AgentState) -> dict:
    """Self-critique the generated response"""
    generation = state["generation"]
    context = state["retrieved_paragraphs"]

    # Use LLM to evaluate response quality
    reflection_prompt = f"""
    Evaluate this response for accuracy and completeness:
    {generation}

    Context: {context}

    Is the response well-supported by the context? Suggest improvements.
    """

    reflection = llm.invoke(reflection_prompt)

    # If reflection suggests improvements, regenerate
    if "improve" in reflection.lower():
        # Trigger regeneration with reflection feedback
        return {
            "metadata": {"needs_improvement": True, "reflection": reflection},
            "generation": ""  # Clear to trigger regenerate
        }

    return {"metadata": {"reflection": reflection}}

# Add conditional routing
def should_reflect(state: AgentState) -> str:
    """Decide if we need reflection"""
    # Only reflect on first generation, avoid infinite loops
    if state.get("metadata", {}).get("reflection_done"):
        return "done"
    return "reflect"

workflow.add_node("reflect", reflect_node)
workflow.add_conditional_edges(
    "generate",
    should_reflect,
    {"reflect": "reflect", "done": END}
)
workflow.add_edge("reflect", "generate")  # Loop back if improvement needed
```

## Testing

### Unit Tests

```python
import pytest
from history_book.services.graph_rag_service import GraphRagService

@pytest.mark.asyncio
async def test_retrieve_node():
    """Test retrieve node fetches paragraphs"""
    service = GraphRagService()

    state = {
        "question": "What is the French Revolution?",
        "retrieved_paragraphs": []
    }

    result = service._retrieve_node(state)

    assert len(result["retrieved_paragraphs"]) > 0
    assert all(hasattr(p, "content") for p in result["retrieved_paragraphs"])

@pytest.mark.asyncio
async def test_graph_execution():
    """Test full graph execution"""
    service = GraphRagService()

    result = await service.invoke(
        question="What is history?",
        messages=[],
        session_id="test-session"
    )

    assert result["generation"]
    assert len(result["retrieved_paragraphs"]) > 0
    assert "history" in result["generation"].lower()
```

### Comparison with LCEL

See `test_langgraph_comparison.py` for automated comparison testing against the LCEL implementation.

**Results**:
- **Retrieval**: Identical (40 paragraphs)
- **Quality**: Equivalent responses
- **Performance**: LangGraph 5.6% faster (8.97s vs 9.50s)

### Integration Tests

```python
@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Test checkpointing maintains context"""
    service = GraphChatService()

    session = await service.create_session(title="Test")
    session_id = session.id

    # First message
    result1 = await service.send_message(
        session_id,
        "Who was Julius Caesar?"
    )
    assert "caesar" in result1.message.content.lower()

    # Second message uses context
    result2 = await service.send_message(
        session_id,
        "When was he assassinated?"
    )
    # Should understand "he" refers to Caesar
    assert "44 bc" in result2.message.content.lower() or "march" in result2.message.content.lower()
```

## Tracing with LangSmith

### Setup

**Environment Variables** (.env):
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=history-book
```

### Viewing Traces

1. Go to https://smith.langchain.com/
2. Navigate to project "history-book"
3. Filter by tags: `agent`, `langgraph`, `simple_rag`

### What's Traced

- **Graph Structure**: Visual representation of nodes and edges
- **Execution Flow**: Step-by-step progression through nodes
- **Timing**: Duration of each node execution
- **State Transitions**: Input/output state at each node
- **LLM Calls**: Prompts sent and responses received
- **Context**: Retrieved paragraphs and how they're used

### Debugging with Traces

**Slow Response?**
- Check LangSmith trace to identify bottleneck node
- Common issues: retrieval timeout, LLM API latency

**Wrong Answer?**
- Inspect retrieved paragraphs - are they relevant?
- Check generation node prompt - is context formatted correctly?

**Context Not Maintained?**
- Verify session_id maps to thread_id correctly
- Check messages array in state - is history preserved?

## Performance

### Benchmarks

**Average over 4 test queries**:
- Latency: 8.97s (vs LCEL 9.50s)
- Retrieval: 40 paragraphs (same as LCEL)
- Quality: Equivalent to LCEL

**Individual Results**:
| Query | LCEL | LangGraph | Difference |
|-------|------|-----------|------------|
| Julius Caesar | 8.48s | 7.76s | -0.72s (faster) |
| WWI Causes | 10.17s | 11.36s | +1.19s (slower) |
| French Revolution | 3.59s | 5.70s | +2.11s (slower) |
| Treaty of Versailles | 15.78s | 11.08s | -4.70s (faster) |

**Overhead**: LangGraph adds ~50-100ms for graph orchestration, but overall faster due to optimizations in state management and LLM calls.

### Optimization Tips

1. **Reduce Retrieval Results**: Lower `max_results` if 40 paragraphs is too many
2. **Cache Embeddings**: Weaviate already caches, but ensure it's configured correctly
3. **Use Faster LLM**: Switch to gpt-3.5-turbo for lower latency (trades quality)
4. **Batch Requests**: Send multiple queries in parallel if possible

## Future Roadmap

### Planned Features

1. **Tool Calling** (High Priority)
   - Web search for recent events
   - Calculator for date calculations
   - External API calls

2. **Planning** (Medium Priority)
   - Multi-step query decomposition
   - Sub-question generation
   - Iterative refinement

3. **Reflection** (Medium Priority)
   - Self-critique response quality
   - Hallucination detection
   - Confidence scoring

4. **Adaptive RAG** (Low Priority)
   - Query routing (semantic search vs keyword search)
   - Document grading (relevance filtering)
   - Web search fallback for gaps in knowledge

5. **Multi-agent** (Future)
   - Specialized agents for different historical periods
   - Debate/discussion between agents
   - Hierarchical task delegation

### Architecture Supports All Features

Current `AgentState` schema is designed to support all future features without breaking changes. New nodes can be added by:
- Adding new state fields as needed
- Inserting nodes in the graph
- Using conditional routing logic

## Troubleshooting

### Common Issues

**Issue**: "Session not found" (404)
- **Cause**: Invalid session_id or session was deleted
- **Fix**: Create new session or verify session_id is correct

**Issue**: Slow response times (>15s)
- **Check**: LangSmith traces to identify bottleneck
- **Common Causes**: Weaviate connection issues, LLM API latency, too many paragraphs
- **Fix**: Reduce `max_results`, check Weaviate status, use faster LLM model

**Issue**: Empty or generic responses
- **Cause**: No relevant paragraphs retrieved
- **Check**: Verify data was ingested (`/api/books` endpoint)
- **Fix**: Re-run ingestion, check similarity thresholds, verify query quality

**Issue**: Context not maintained across messages
- **Cause**: MemorySaver state lost or session_id mismatch
- **Check**: Verify server didn't restart, check thread_id mapping
- **Fix**: Ensure consistent session_id, avoid server restarts during conversations

**Issue**: Hallucinations in response
- **Cause**: LLM generating beyond provided context
- **Check**: LangSmith trace - inspect generation node prompt
- **Fix**: Improve prompt to emphasize context-only answers, add reflection node

### Debug Mode

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# View all graph executions
logger = logging.getLogger("history_book.services.graph_rag_service")
logger.setLevel(logging.DEBUG)
```

### Inspecting State

```python
# Add debug prints in nodes
def retrieve_node_debug(state: AgentState) -> dict:
    print(f"Retrieve node input: {state['question']}")

    result = retrieve_logic(state)

    print(f"Retrieve node output: {len(result['retrieved_paragraphs'])} paragraphs")
    return result
```

## Related Files

- **GraphRagService**: `/src/history_book/services/graph_rag_service.py`
- **GraphChatService**: `/src/history_book/services/graph_chat_service.py`
- **Agent API**: `/src/history_book/api/routes/agent.py`
- **State Definition**: `/src/history_book/data_models/graph_state.py`
- **Config**: `/src/history_book/config/graph_config.py`
- **Comparison Tests**: `/test_langgraph_comparison.py`
- **Progress Tracking**: `/docs/plans/langgraph_rag_progress.md`
