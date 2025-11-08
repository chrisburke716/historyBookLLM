# LangGraph RAG Implementation Plan

**Created**: 2025-11-08
**Status**: Planning Phase
**Goal**: Re-implement RAG chat using LangGraph alongside existing LCEL implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Design Decisions](#design-decisions)
4. [Technical Architecture](#technical-architecture)
5. [Implementation Phases](#implementation-phases)
6. [LangGraph Patterns & Best Practices](#langgraph-patterns--best-practices)
7. [Files to Create/Modify](#files-to-createmodify)
8. [Testing Strategy](#testing-strategy)
9. [Future Extensibility](#future-extensibility)
10. [References](#references)

---

## Executive Summary

### Project Goals

1. **Re-implement** the existing RAG chat functionality using LangGraph
2. **Add, not replace** - Keep existing `/api/chat/*` endpoints unchanged
3. **Position for future** - Enable tool calling and multi-step reasoning workflows
4. **Learn LangGraph** - Intentional exploration of graph-based agent architecture

### Key Outcomes

- New `/api/agent/*` API namespace for LangGraph-based chat
- `GraphRagService` and `GraphChatService` classes
- Hybrid memory strategy: LangGraph MemorySaver + Weaviate persistence
- Simple RAG graph: retrieve → generate (foundation for future complexity)
- Full feature parity: streaming, tracing, multi-provider LLM support

### Non-Goals

- NOT replacing existing chat functionality
- NOT migrating frontend initially (API-only implementation)
- NOT implementing advanced agentic features yet (tool calling, multi-agent, etc.)
- NOT using complex persistence (PostgreSQL checkpointers, etc.)

---

## Current Architecture Analysis

### Overview

The current system implements a RAG chat application with:
- **Backend**: Python FastAPI with clean architecture pattern
- **RAG Implementation**: LangChain LCEL chains in `RagService`
- **Session Management**: `ChatService` orchestrating sessions, messages, and RAG
- **Storage**: Weaviate vector database for documents, sessions, and messages
- **LLM Support**: OpenAI and Anthropic via `LLMConfig` abstraction

### Current Data Flow

```
User Message → ChatService.send_message()
              ↓
              → Prepare & save user message to Weaviate
              → Load chat history from Weaviate
              → ChatService calls RagService.generate_response()
              ↓
              RagService:
              → Retrieve context (ParagraphRepository.similarity_search)
              → Format context with citations
              → Build LCEL chain (PromptTemplate | ChatModel | Parser)
              → Execute chain with query + history + context
              → Return RAGResult(response, source_paragraphs)
              ↓
              → ChatService saves AI message to Weaviate
              → Return ChatResult(message, retrieved_paragraphs)
```

### RagService Details

**File**: `src/history_book/services/rag_service.py`

#### LCEL Chain Architecture

Two separate chains based on whether context is available:

```python
# RAG Chain (with context)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{context}\n\nQuestion: {query}"),
])
rag_chain = rag_prompt | chat_model | StrOutputParser()

# Simple Chain (without context)
simple_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}"),
])
simple_chain = simple_prompt | chat_model | StrOutputParser()
```

#### Retrieval Strategy

**Method**: `_retrieve_context(query, min_results=5, max_results=40, similarity_cutoff=0.4)`

**Flow**:
1. Call `ParagraphRepository.similarity_search_by_text(query, max_results, similarity_cutoff)`
2. Uses Weaviate's `near_text` query (vector similarity search)
3. If results < min_results, retry without threshold to ensure minimum context
4. Return list of Paragraph entities with similarity scores

**Weaviate Integration**:
- Auto-generates embeddings using OpenAI's embedding model
- Cosine similarity search in vector space
- Returns paragraphs sorted by relevance

#### Context Formatting

**Format**:
```
Context from historical documents:

[Source 1, Page 42]: paragraph text here

[Source 2, Page 67]: another paragraph text

Please answer the question based on the context provided above.
```

**Truncation**:
- Max length: 100,000 characters
- Truncates at sentence boundaries (looks for last period in final 20%)
- Adds "..." if truncated

#### LLM Configuration

**Supported Providers**: OpenAI, Anthropic

**Model Creation**:
```python
def _create_chat_model(llm_config: LLMConfig):
    provider_class = import_provider_class(llm_config.provider)
    return provider_class(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        # ... other config parameters
    )
```

#### Output Format

**Returns**: `RAGResult` NamedTuple
```python
RAGResult(
    response: str,              # LLM response text
    source_paragraphs: list[Paragraph]  # Retrieved paragraphs used
)
```

#### Streaming Support

**Method**: `stream_response()`
- Same retrieval logic
- Returns tuple: `(async_generator, source_paragraphs)`
- Yields chunks as they arrive from LLM
- Does NOT save to database (handled by ChatService)

### ChatService Details

**File**: `src/history_book/services/chat_service.py`

#### Core Responsibilities

1. Chat session CRUD operations
2. Message persistence to Weaviate
3. History retrieval and formatting
4. RAG orchestration via RagService
5. LangSmith tracing (`@traceable` decorator)

#### Configuration Parameters

```python
def __init__(
    self,
    config: WeaviateConfig | None = None,
    llm_config: LLMConfig | None = None,
    min_context_results: int = 5,
    max_context_results: int = 40,
    context_similarity_cutoff: float = 0.4,
    retrieval_strategy: str = "similarity_search",
)
```

#### send_message() Flow

**Steps**:

1. **Prepare** (`_prepare_message_and_history`):
   - Create `ChatMessage` entity (role=USER)
   - Save to Weaviate via `chat_messages.create()`
   - Retrieve chat history via `get_session_messages()`
   - Append new user message to history

2. **Generate**:
   - Call `rag_service.generate_response(query, messages, ...)`
   - Receive `RAGResult` with response + source paragraphs

3. **Save** (`_save_ai_message_and_update_session`):
   - Create `ChatMessage` entity (role=ASSISTANT)
   - Include `retrieved_paragraphs` field (list of IDs)
   - Save to Weaviate
   - Update session's `updated_at` timestamp

4. **Return**:
   - `ChatResult(message=ChatMessage, retrieved_paragraphs=list[Paragraph])`

#### Streaming Flow

**Method**: `send_message_stream()`

**Differences**:
1. Calls `rag_service.stream_response()` instead
2. Wraps stream in async generator
3. Accumulates full response during streaming
4. Saves complete message after streaming finishes
5. Returns: `(stream_generator, retrieved_paragraphs)`

#### Chat History Management

**Retrieval**:
- `get_session_messages()` fetches all messages for a session
- Uses `ChatMessageRepository.find_by_session(session_id)`
- Returns messages sorted by timestamp (oldest first)

**Formatting** (in RagService):
- Calls `format_messages_for_llm()` utility
- Limits to most recent `max_conversation_length` messages (default 20)
- Converts to LangChain format:
  - USER → `HumanMessage`
  - ASSISTANT → `AIMessage`

### API Endpoints

**File**: `src/history_book/api/routes/chat.py`

#### Current Endpoints

1. **POST `/api/chat/sessions`** - Create session
   - Request: `SessionCreateRequest(title: str | None)`
   - Response: `SessionResponse`

2. **GET `/api/chat/sessions?limit=10`** - List recent sessions
   - Response: `SessionListResponse(sessions: list[SessionResponse])`

3. **GET `/api/chat/sessions/{session_id}/messages`** - Get history
   - Response: `MessageListResponse(messages: list[MessageResponse])`

4. **POST `/api/chat/sessions/{session_id}/messages`** - Send message
   - Request: `MessageRequest(content, enable_retrieval, max_context_paragraphs)`
   - Response: `ChatResponse(message: MessageResponse)`

#### Dependency Injection Pattern

```python
def get_chat_service():
    return ChatService()

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: MessageRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    result = await chat_service.send_message(...)
    return convert_message_to_response(result.message, result.retrieved_paragraphs)
```

### Data Models

**File**: `src/history_book/data_models/entities.py`

#### Core Entities

```python
class ChatSession(BaseModel):
    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class ChatMessage(BaseModel):
    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: MessageRole  # USER | ASSISTANT | SYSTEM
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    retrieved_paragraphs: list[str] | None = None  # IDs only

    vectorize_fields: ClassVar[list[str]] = ["content"]

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Paragraph(BaseModel):
    id: str | None
    text: str
    embedding: list[float] | None = None
    page: int
    paragraph_index: int
    book_index: int
    chapter_index: int

    vectorize_fields: ClassVar[list[str]] = ["text"]
```

#### Service DTOs

```python
class RAGResult(NamedTuple):
    response: str
    source_paragraphs: list[Paragraph]

@dataclass
class ChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
```

---

## Design Decisions

### Decision Matrix

| Decision Area | Choice | Rationale |
|--------------|--------|-----------|
| **Graph Architecture Pattern** | Simple RAG (retrieve → generate) | Foundation for future enhancement; recreate current functionality first |
| **Memory Strategy** | Hybrid: MemorySaver + Weaviate | LangGraph state management during execution, long-term persistence in Weaviate |
| **Service Architecture** | New service layer, shared repos | Clean separation, reuse existing data layer, follows "add not replace" goal |
| **API Namespace** | `/api/agent/*` | Signals agentic capabilities, completely separate from existing chat |
| **API Design** | Completely independent | Optimal design for LangGraph features without backwards compatibility constraints |
| **LLM Support** | Existing LLMConfig abstraction | Support all providers from day one using proven pattern |
| **Checkpointer** | MemorySaver only | Simple, appropriate for quick RAG executions; no need for PostgreSQL complexity |
| **Streaming** | Native LangGraph streaming | Better UX, built-in support with `stream_mode="messages"` |
| **Tracing** | Continue using LangSmith | Automatic tracing with better graph visualization |
| **Future Capabilities** | Design for tool calling + multi-step reasoning | State schema accommodates future features without breaking changes |

### Future Agent Capabilities (Planned)

Based on user requirements, the architecture should support:

1. **Tool calling** - Search, calculations, API calls, external data sources
2. **Multi-step reasoning** - Planning, self-correction, iterative refinement
3. **NOT planned initially**: Multi-agent collaboration, human-in-the-loop

### Why MemorySaver (Not PostgreSQL)?

**Context**: This is a personal project with simple, fast RAG workflows.

**MemorySaver is sufficient because**:
- RAG executions complete in 1-2 seconds (no long-running workflows to recover)
- No human-in-the-loop pausing mid-execution needed
- Single server instance (personal project)
- Long-term persistence handled by Weaviate (sessions + messages)
- In-graph state only needs to live during execution

**When you'd need PostgresSaver**:
- Multi-hour workflows that could be interrupted
- Human-in-the-loop approvals mid-execution
- Multiple server instances requiring shared state
- Complex branching workflows with rollback needs

**Decision**: Start with MemorySaver, upgrade to SqliteSaver or PostgresSaver only if future features require it.

---

## Technical Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                               │
│  /api/agent/*                    /api/chat/* (existing)     │
│  - AgentRouter (new)             - ChatRouter (unchanged)   │
└─────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│  Graph Services (NEW)       │  │  Existing Services          │
│  - GraphChatService         │  │  - ChatService              │
│  - GraphRagService          │  │  - RagService               │
└─────────────────────────────┘  └─────────────────────────────┘
                    │                           │
                    └───────────┬───────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │  Shared Components      │
                    │  - ParagraphRepository  │
                    │  - ChatSession Repos    │
                    │  - LLMConfig            │
                    │  - Weaviate Connection  │
                    └─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Weaviate Database     │
                    │  - Paragraphs (vectors) │
                    │  - Chat Sessions        │
                    │  - Chat Messages        │
                    └─────────────────────────┘
```

### LangGraph Architecture

#### State Schema

**File**: `src/history_book/data_models/graph_state.py`

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from history_book.data_models.entities import Paragraph

class AgentState(TypedDict):
    """
    State schema for the RAG agent graph.

    The state is passed between nodes and updated at each step.
    Uses Annotated with add_messages reducer for automatic message list management.
    """
    # Messages with automatic list management (reducer pattern)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Current user query
    question: str

    # Retrieved context documents
    retrieved_paragraphs: list[Paragraph]

    # Generated response
    generation: str

    # Session identifier (maps to Weaviate session_id and LangGraph thread_id)
    session_id: str

    # Execution metadata for debugging/tracing
    metadata: dict

    # Future fields for extensibility (not used initially):
    # tool_calls: list[dict]  # For tool calling feature
    # tool_results: list[dict]  # Results from tool executions
    # reasoning_steps: list[dict]  # For multi-step reasoning tracking
```

**Reducer Pattern**: The `add_messages` reducer automatically appends new messages to the list rather than replacing it, which is essential for maintaining chat history.

#### Graph Structure

**Simple RAG Graph**:

```
START → retrieve_node → generate_node → END
```

**Node Descriptions**:

1. **retrieve_node**:
   - Input: `state["question"]`
   - Action: Query ParagraphRepository.similarity_search_by_text()
   - Output: Updates `state["retrieved_paragraphs"]`
   - Error handling: Returns empty list on failure (graceful degradation)

2. **generate_node**:
   - Input: `state["messages"]`, `state["question"]`, `state["retrieved_paragraphs"]`
   - Action: Format context + invoke LLM via existing LLMConfig
   - Output: Updates `state["generation"]` and appends AI message to `state["messages"]`
   - Fallback: If no paragraphs retrieved, generates without context

### GraphRagService Design

**File**: `src/history_book/services/graph_rag_service.py`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from history_book.data_models.graph_state import AgentState
from history_book.llm.config import LLMConfig
from history_book.database.repositories.paragraph_repository import ParagraphRepository

class GraphRagService:
    """
    LangGraph-based RAG service for agentic chat functionality.

    Implements a simple RAG graph: retrieve → generate
    Designed to be extended with tool calling and multi-step reasoning.
    """

    def __init__(
        self,
        paragraph_repository: ParagraphRepository,
        llm_config: LLMConfig,
        min_context_results: int = 5,
        max_context_results: int = 40,
        context_similarity_cutoff: float = 0.4,
    ):
        self.paragraph_repo = paragraph_repository
        self.llm_config = llm_config
        self.min_results = min_context_results
        self.max_results = max_context_results
        self.similarity_cutoff = context_similarity_cutoff

        # Create LLM using existing config (supports OpenAI, Anthropic, etc.)
        self.llm = self._create_chat_model()

        # Build and compile graph
        self.graph = self._create_graph()

    def _create_chat_model(self):
        """Reuse existing LLM creation logic from LLMConfig"""
        # Enable streaming for token-by-token output
        # This is the same pattern as RagService
        pass

    def _create_graph(self) -> CompiledGraph:
        """Build and compile the RAG graph"""
        # Initialize workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        # Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile with checkpointer
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    def _retrieve_node(self, state: AgentState) -> dict:
        """
        Retrieval node: Fetch relevant paragraphs from vector database

        Mirrors the current RagService._retrieve_context() logic
        """
        question = state["question"]

        try:
            # Use existing repository method
            results = self.paragraph_repo.similarity_search_by_text(
                query_text=question,
                limit=self.max_results,
                threshold=self.similarity_cutoff,
            )

            # Extract paragraphs from (paragraph, score) tuples
            paragraphs = [p for p, score in results]

            # Ensure minimum results (retry without threshold if needed)
            if len(paragraphs) < self.min_results:
                results = self.paragraph_repo.similarity_search_by_text(
                    query_text=question,
                    limit=self.min_results,
                    threshold=None,  # No threshold
                )
                paragraphs = [p for p, score in results]

            return {"retrieved_paragraphs": paragraphs}

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {"retrieved_paragraphs": []}  # Graceful degradation

    def _generate_node(self, state: AgentState) -> dict:
        """
        Generation node: Create response using LLM + context

        Mirrors current RagService chain logic
        """
        question = state["question"]
        messages = state["messages"]
        paragraphs = state["retrieved_paragraphs"]

        # Format context (same as current implementation)
        if paragraphs:
            context = self._format_context(paragraphs)
            prompt = f"{context}\n\nQuestion: {question}"
        else:
            # No context available - answer without retrieval
            prompt = question

        # Build message list for LLM
        message_history = list(messages)
        message_history.append(HumanMessage(content=prompt))

        # Invoke LLM
        response = self.llm.invoke(message_history)

        return {
            "generation": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    def _format_context(self, paragraphs: list[Paragraph]) -> str:
        """Format paragraphs for LLM context (same as current RagService)"""
        # Reuse existing formatting logic from llm/utils.py
        pass

    def invoke(
        self,
        question: str,
        messages: list[BaseMessage],
        session_id: str,
        config: dict | None = None,
    ) -> AgentState:
        """
        Synchronous graph execution

        Args:
            question: User query
            messages: Chat history (LangChain format)
            session_id: Session identifier (used as thread_id)
            config: Optional RunnableConfig for tracing, etc.

        Returns:
            Final state after graph execution
        """
        initial_state = {
            "messages": messages,
            "question": question,
            "retrieved_paragraphs": [],
            "generation": "",
            "session_id": session_id,
            "metadata": {},
        }

        # Execute graph with thread_id for checkpointing
        config = config or {}
        config["configurable"] = {"thread_id": session_id}

        result = self.graph.invoke(initial_state, config=config)
        return result

    def stream(
        self,
        question: str,
        messages: list[BaseMessage],
        session_id: str,
        config: dict | None = None,
    ):
        """
        Streaming graph execution (token-by-token)

        Uses stream_mode="messages" to get LLM token chunks
        """
        initial_state = {
            "messages": messages,
            "question": question,
            "retrieved_paragraphs": [],
            "generation": "",
            "session_id": session_id,
            "metadata": {},
        }

        config = config or {}
        config["configurable"] = {"thread_id": session_id}

        # Stream with messages mode to get token chunks
        return self.graph.stream(
            initial_state,
            config=config,
            stream_mode="messages",  # Token-by-token streaming
        )
```

**Key Design Patterns**:
- Reuses existing `ParagraphRepository` (no changes to data layer)
- Reuses existing `LLMConfig` (multi-provider support)
- Same retrieval logic as current `RagService`
- Same context formatting as current implementation
- Graceful degradation on retrieval failure

### GraphChatService Design

**File**: `src/history_book/services/graph_chat_service.py`

```python
from history_book.services.graph_rag_service import GraphRagService
from history_book.database.repositories.chat_repository import ChatSessionRepository, ChatMessageRepository
from history_book.data_models.entities import ChatSession, ChatMessage, MessageRole
from history_book.llm.utils import format_messages_for_llm
from langchain_core.messages import HumanMessage, AIMessage

class GraphChatService:
    """
    Chat session orchestration for LangGraph-based RAG.

    Hybrid memory strategy:
    - LangGraph MemorySaver: In-graph state during execution
    - Weaviate: Long-term persistence of sessions and messages
    """

    def __init__(
        self,
        graph_rag_service: GraphRagService,
        chat_sessions_repo: ChatSessionRepository,
        chat_messages_repo: ChatMessageRepository,
    ):
        self.graph_rag = graph_rag_service
        self.sessions_repo = chat_sessions_repo
        self.messages_repo = chat_messages_repo

    async def send_message(
        self,
        session_id: str,
        content: str,
    ) -> tuple[ChatMessage, list[Paragraph]]:
        """
        Send a message and get response using LangGraph

        Flow:
        1. Save user message to Weaviate
        2. Load chat history from Weaviate
        3. Execute graph with history + new message
        4. Save AI response to Weaviate
        5. Return AI message + source paragraphs
        """
        # 1. Save user message
        user_message = ChatMessage(
            content=content,
            role=MessageRole.USER,
            session_id=session_id,
        )
        await self.messages_repo.create(user_message)

        # 2. Load chat history
        history = await self.get_session_messages(session_id)

        # 3. Convert to LangChain message format
        lc_messages = self._convert_to_langchain_messages(history)

        # 4. Execute graph
        result_state = self.graph_rag.invoke(
            question=content,
            messages=lc_messages,
            session_id=session_id,
        )

        # 5. Save AI response
        ai_message = ChatMessage(
            content=result_state["generation"],
            role=MessageRole.ASSISTANT,
            session_id=session_id,
            retrieved_paragraphs=[p.id for p in result_state["retrieved_paragraphs"]],
        )
        await self.messages_repo.create(ai_message)

        # 6. Update session timestamp
        await self.sessions_repo.update(session_id, {
            "updated_at": datetime.now(UTC)
        })

        return ai_message, result_state["retrieved_paragraphs"]

    async def send_message_stream(
        self,
        session_id: str,
        content: str,
    ):
        """
        Streaming version of send_message

        Yields token chunks, then saves complete message at end
        """
        # 1. Save user message (same as non-streaming)
        user_message = ChatMessage(
            content=content,
            role=MessageRole.USER,
            session_id=session_id,
        )
        await self.messages_repo.create(user_message)

        # 2. Load history and convert
        history = await self.get_session_messages(session_id)
        lc_messages = self._convert_to_langchain_messages(history)

        # 3. Stream graph execution
        full_response = ""
        retrieved_paragraphs = []

        async for chunk in self.graph_rag.stream(
            question=content,
            messages=lc_messages,
            session_id=session_id,
        ):
            # Yield token chunks to client
            if isinstance(chunk, dict) and "generation" in chunk:
                full_response += chunk["generation"]
                yield chunk["generation"]

            # Capture final state
            if isinstance(chunk, dict) and "retrieved_paragraphs" in chunk:
                retrieved_paragraphs = chunk["retrieved_paragraphs"]

        # 4. Save complete response
        ai_message = ChatMessage(
            content=full_response,
            role=MessageRole.ASSISTANT,
            session_id=session_id,
            retrieved_paragraphs=[p.id for p in retrieved_paragraphs],
        )
        await self.messages_repo.create(ai_message)

        # 5. Update session
        await self.sessions_repo.update(session_id, {
            "updated_at": datetime.now(UTC)
        })

    async def get_session_messages(self, session_id: str) -> list[ChatMessage]:
        """Load chat history from Weaviate"""
        return await self.messages_repo.find_by_session(session_id)

    def _convert_to_langchain_messages(
        self,
        messages: list[ChatMessage]
    ) -> list[BaseMessage]:
        """
        Convert ChatMessage entities to LangChain message format

        Reuses existing format_messages_for_llm() utility
        """
        # Same logic as current ChatService
        pass

    # Session management methods (reuse existing patterns)
    async def create_session(self, title: str | None = None) -> str:
        """Create new chat session"""
        pass

    async def delete_session(self, session_id: str):
        """Delete session and all messages"""
        pass

    async def list_sessions(self, limit: int = 10) -> list[ChatSession]:
        """List recent sessions"""
        pass
```

**Hybrid Memory Strategy**:

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph MemorySaver                     │
│  - In-memory state during graph execution                   │
│  - Keyed by thread_id (= session_id)                        │
│  - Enables human-in-the-loop, resumable workflows           │
│  - Lost on server restart (OK for quick RAG executions)     │
└─────────────────────────────────────────────────────────────┘
                            ▼
                   Graph Execution
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Weaviate Persistence                      │
│  - Long-term storage of sessions and messages               │
│  - Survives server restarts                                 │
│  - Enables cross-session analytics and search               │
│  - Source of truth for chat history                         │
└─────────────────────────────────────────────────────────────┘
```

### API Layer Design

**File**: `src/history_book/api/routes/agent.py`

#### Endpoints

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from history_book.services.graph_chat_service import GraphChatService
from history_book.api.models.agent_models import (
    AgentMessageRequest,
    AgentMessageResponse,
    AgentSessionCreate,
    GraphVisualization,
)

router = APIRouter(prefix="/api/agent", tags=["agent"])

def get_graph_chat_service() -> GraphChatService:
    """Dependency injection for GraphChatService"""
    # Initialize service with dependencies
    pass

# Session management
@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: AgentSessionCreate,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """Create a new agent chat session"""
    session_id = await service.create_session(request.title)
    return SessionResponse(session_id=session_id, ...)

@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    limit: int = 10,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """List recent agent sessions"""
    sessions = await service.list_sessions(limit)
    return [SessionResponse(...) for s in sessions]

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """Delete a session and all its messages"""
    await service.delete_session(session_id)
    return {"status": "deleted"}

# Message handling
@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """Get all messages for a session"""
    messages = await service.get_session_messages(session_id)
    return {"messages": [AgentMessageResponse(...) for m in messages]}

@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: AgentMessageRequest,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    Send a message to the agent (non-streaming)

    Returns complete response with graph execution metadata
    """
    ai_message, paragraphs = await service.send_message(
        session_id=session_id,
        content=request.content,
    )

    return AgentMessageResponse(
        id=ai_message.id,
        content=ai_message.content,
        role=ai_message.role,
        timestamp=ai_message.timestamp,
        citations=[f"Page {p.page}" for p in paragraphs],
        metadata={
            "num_retrieved_paragraphs": len(paragraphs),
            "graph_execution": "simple_rag",
        },
    )

@router.post("/sessions/{session_id}/stream")
async def stream_message(
    session_id: str,
    request: AgentMessageRequest,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    Send a message with streaming response (Server-Sent Events)
    """
    async def event_generator():
        async for chunk in service.send_message_stream(
            session_id=session_id,
            content=request.content,
        ):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

# Graph introspection and debugging
@router.get("/sessions/{session_id}/graph")
async def get_graph_visualization(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    Get graph structure visualization for debugging

    Returns Mermaid diagram text
    """
    graph = service.graph_rag.graph
    mermaid = graph.get_graph().draw_mermaid()

    return GraphVisualization(
        mermaid=mermaid,
        nodes=["retrieve", "generate"],
        edges=[("START", "retrieve"), ("retrieve", "generate"), ("generate", "END")],
    )

@router.get("/sessions/{session_id}/checkpoints")
async def list_checkpoints(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    List graph execution checkpoints for a session

    Useful for debugging and understanding execution history
    """
    # Access MemorySaver to list checkpoints for thread_id
    # This is advanced - may not implement initially
    pass
```

#### Request/Response Models

**File**: `src/history_book/api/models/agent_models.py`

```python
from pydantic import BaseModel, Field
from datetime import datetime
from history_book.data_models.entities import MessageRole

class AgentSessionCreate(BaseModel):
    """Request to create a new agent session"""
    title: str | None = None

class AgentMessageRequest(BaseModel):
    """Request to send a message to the agent"""
    content: str = Field(..., min_length=1, max_length=10000)

    # Optional retrieval configuration overrides
    max_context_paragraphs: int | None = Field(default=None, ge=1, le=100)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

class AgentMessageResponse(BaseModel):
    """Agent message with graph execution metadata"""
    id: str
    content: str
    role: MessageRole
    timestamp: datetime
    citations: list[str]  # e.g., ["Page 42", "Page 67"]

    # Graph execution metadata (new compared to chat API)
    metadata: dict = Field(default_factory=dict)
    # Example metadata:
    # {
    #     "num_retrieved_paragraphs": 5,
    #     "graph_execution": "simple_rag",
    #     "nodes_executed": ["retrieve", "generate"],
    #     "execution_time_ms": 1234,
    # }

class GraphVisualization(BaseModel):
    """Graph structure for debugging"""
    mermaid: str  # Mermaid diagram syntax
    nodes: list[str]
    edges: list[tuple[str, str]]
```

**Key Differences from Chat API**:
1. **Independent design**: Not constrained by existing `/api/chat/*` format
2. **Enhanced metadata**: Includes graph execution details
3. **Separate streaming endpoint**: `/stream` rather than query parameter
4. **Graph introspection**: `/graph` and `/checkpoints` endpoints for debugging
5. **Future-ready**: Easy to add tool calls, reasoning steps, etc. to metadata

---

## Implementation Phases

### Phase 1: Core Graph Implementation (Foundational)

**Goal**: Recreate existing RAG functionality with LangGraph

#### 1.1 Dependencies & Configuration

**Tasks**:
- [ ] Add LangGraph to `pyproject.toml`: `langgraph = "^1.0"`
- [ ] Add checkpointer package: `langgraph-checkpoint = "^2.0"`
- [ ] Update dependencies: `poetry lock && poetry install`
- [ ] Create configuration file: `src/history_book/config/graph_config.py`

**graph_config.py**:
```python
from pydantic_settings import BaseSettings

class GraphConfig(BaseSettings):
    """Configuration for LangGraph-based agent"""

    # Checkpointer settings
    checkpointer_type: str = "memory"  # "memory" | "sqlite" (future)

    # Feature flags
    enable_streaming: bool = True
    enable_tracing: bool = True
    enable_graph_visualization: bool = True

    # Retrieval configuration
    max_retrieval_results: int = 40
    min_retrieval_results: int = 5
    similarity_threshold: float = 0.4

    # LLM configuration (reuses existing LLMConfig)
    # Future: tool calling configuration
    # enabled_tools: list[str] = []
```

#### 1.2 State Schema

**Tasks**:
- [ ] Create `src/history_book/data_models/graph_state.py`
- [ ] Define `AgentState` TypedDict with all fields
- [ ] Add comprehensive docstrings
- [ ] Include future fields as comments

**Reference**: See [State Schema](#state-schema) section above for complete code

#### 1.3 GraphRagService

**Tasks**:
- [ ] Create `src/history_book/services/graph_rag_service.py`
- [ ] Implement `_create_graph()` method
- [ ] Implement `_retrieve_node()` - reuse ParagraphRepository
- [ ] Implement `_generate_node()` - reuse LLMConfig
- [ ] Implement `_format_context()` - reuse existing utility
- [ ] Implement `invoke()` method
- [ ] Implement `stream()` method with `stream_mode="messages"`
- [ ] Add error handling and logging
- [ ] Add docstrings

**Reference**: See [GraphRagService Design](#graphragservice-design) section above

**Testing**:
```python
# test_graph_rag_service.py
def test_retrieve_node():
    """Test retrieval node returns paragraphs"""

def test_generate_node():
    """Test generation node creates response"""

def test_invoke_full_graph():
    """Test complete graph execution"""

def test_streaming():
    """Test token streaming works"""
```

#### 1.4 GraphChatService

**Tasks**:
- [ ] Create `src/history_book/services/graph_chat_service.py`
- [ ] Implement session management (reuse existing repos)
- [ ] Implement `send_message()` - orchestrate graph + persistence
- [ ] Implement `send_message_stream()` - streaming variant
- [ ] Implement `_convert_to_langchain_messages()` helper
- [ ] Add error handling for missing sessions
- [ ] Add docstrings

**Reference**: See [GraphChatService Design](#graphchatservice-design) section above

**Testing**:
```python
# test_graph_chat_service.py
def test_send_message_saves_to_weaviate():
    """Ensure messages persisted correctly"""

def test_chat_history_loaded():
    """Ensure history passed to graph"""

def test_streaming_saves_complete_message():
    """Ensure streaming accumulates and saves"""
```

### Phase 2: API Layer (User-Facing)

**Goal**: Expose LangGraph functionality via REST API

#### 2.1 API Models

**Tasks**:
- [ ] Create `src/history_book/api/models/agent_models.py`
- [ ] Define request models: `AgentMessageRequest`, `AgentSessionCreate`
- [ ] Define response models: `AgentMessageResponse`, `GraphVisualization`
- [ ] Add validation rules (Pydantic)
- [ ] Add docstrings and examples

**Reference**: See [Request/Response Models](#requestresponse-models) section above

#### 2.2 Agent Router

**Tasks**:
- [ ] Create `src/history_book/api/routes/agent.py`
- [ ] Implement dependency injection for `GraphChatService`
- [ ] Implement session endpoints (POST, GET, DELETE)
- [ ] Implement message endpoints (GET, POST)
- [ ] Implement streaming endpoint (POST /stream with SSE)
- [ ] Implement graph visualization endpoint (GET /graph)
- [ ] Add proper error handling (404, 500, etc.)
- [ ] Add docstrings for OpenAPI docs

**Reference**: See [API Layer Design](#api-layer-design) section above

#### 2.3 Register Router

**Tasks**:
- [ ] Modify `src/history_book/api/main.py`
- [ ] Import and register agent router
- [ ] Ensure proper middleware (CORS, etc.)
- [ ] Test OpenAPI docs at `/docs`

```python
# main.py
from history_book.api.routes import agent

app.include_router(agent.router)
```

#### 2.4 API Testing

**Tasks**:
- [ ] Create `tests/test_agent_api.py`
- [ ] Test session creation
- [ ] Test message sending (non-streaming)
- [ ] Test streaming endpoint
- [ ] Test error cases (invalid session, etc.)
- [ ] Test graph visualization endpoint

**Example test**:
```python
def test_send_message_to_agent(client):
    # Create session
    response = client.post("/api/agent/sessions", json={"title": "Test"})
    session_id = response.json()["session_id"]

    # Send message
    response = client.post(
        f"/api/agent/sessions/{session_id}/messages",
        json={"content": "What is RAG?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "citations" in data
    assert "metadata" in data
```

### Phase 3: LangGraph Features (Enhanced)

**Goal**: Enable LangGraph-specific capabilities

#### 3.1 Streaming Support

**Tasks**:
- [ ] Ensure LLM initialized with `streaming=True`
- [ ] Test `stream_mode="messages"` returns token chunks
- [ ] Implement SSE (Server-Sent Events) properly
- [ ] Test with frontend or curl
- [ ] Handle streaming errors gracefully

**Testing streaming with curl**:
```bash
curl -N http://localhost:8000/api/agent/sessions/{id}/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "What is history?"}'
```

#### 3.2 LangSmith Tracing

**Tasks**:
- [ ] Verify existing `LANGCHAIN_*` env vars work
- [ ] Add "agent" tags to trace graph executions
- [ ] Test traces appear in LangSmith UI
- [ ] Compare LCEL vs LangGraph traces

**Environment variables** (should already be set):
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=history_book
```

**Add tags in GraphRagService**:
```python
def invoke(self, ..., config: dict | None = None):
    config = config or {}
    config.setdefault("tags", []).append("agent")
    config.setdefault("tags", []).append("langgraph")

    return self.graph.invoke(initial_state, config=config)
```

#### 3.3 Checkpointing with MemorySaver

**Tasks**:
- [ ] Ensure `MemorySaver()` used in graph compilation
- [ ] Map `session_id` → `thread_id` consistently
- [ ] Test state persists across multiple messages in same session
- [ ] Verify state isolated between different sessions

**Testing**:
```python
def test_checkpointing_preserves_state():
    # Send message 1
    result1 = service.send_message(session_id, "Hello")

    # Send message 2 - should have access to message 1 via checkpointing
    result2 = service.send_message(session_id, "Continue our conversation")

    # Verify message 1 context available in graph execution
```

#### 3.4 Graph Visualization

**Tasks**:
- [ ] Implement `GET /api/agent/sessions/{id}/graph`
- [ ] Use `graph.get_graph().draw_mermaid()` to generate diagram
- [ ] Return Mermaid syntax in response
- [ ] Test diagram renders correctly (can use online Mermaid editor)

**Example output**:
```mermaid
graph TD
    __start__[Start] --> retrieve
    retrieve --> generate
    generate --> __end__[End]
```

**Future enhancement**: Return PNG image using `.draw_png()` (requires graphviz)

### Phase 4: Testing & Validation (Quality)

**Goal**: Ensure parity and quality

#### 4.1 Comparison Testing

**Tasks**:
- [ ] Create test suite comparing LCEL vs LangGraph outputs
- [ ] Test with same queries, measure response similarity
- [ ] Compare retrieval results (should be identical)
- [ ] Compare execution times
- [ ] Document any differences

**Example**:
```python
def test_lcel_vs_langgraph_parity():
    # Same query to both systems
    query = "What caused World War I?"

    # LCEL (existing)
    lcel_result = rag_service.generate_response(query, [])

    # LangGraph (new)
    graph_result = graph_rag_service.invoke(query, [], session_id)

    # Compare
    assert len(lcel_result.source_paragraphs) == len(graph_result["retrieved_paragraphs"])
    # Response content may differ slightly but should be semantically similar
```

#### 4.2 Integration Testing

**Tasks**:
- [ ] Create `tests/test_graph_integration.py`
- [ ] Test end-to-end: API → Service → Graph → Database
- [ ] Test multi-turn conversations
- [ ] Test error scenarios (DB down, LLM error, etc.)
- [ ] Test streaming end-to-end

#### 4.3 Performance Testing

**Tasks**:
- [ ] Measure latency: LCEL vs LangGraph
- [ ] Identify any performance regressions
- [ ] Profile memory usage (MemorySaver)
- [ ] Test concurrent requests

**Expected**: LangGraph may add small overhead (~50-100ms) due to graph orchestration, but should be negligible for 1-2 second RAG responses.

#### 4.4 Manual Testing

**Tasks**:
- [ ] Test with Postman or similar
- [ ] Test streaming in browser or curl
- [ ] View traces in LangSmith
- [ ] Test graph visualization endpoint
- [ ] Verify Weaviate data integrity

### Phase 5: Documentation & Cleanup (Polish)

**Goal**: Document and prepare for future work

#### 5.1 Code Documentation

**Tasks**:
- [ ] Add comprehensive docstrings to all classes/methods
- [ ] Add type hints everywhere
- [ ] Add inline comments for complex logic
- [ ] Update existing CLAUDE.md files

#### 5.2 API Documentation

**Tasks**:
- [ ] Ensure OpenAPI docs at `/docs` are complete
- [ ] Add examples to endpoint descriptions
- [ ] Document request/response schemas
- [ ] Add error response documentation

#### 5.3 Create CLAUDE.md for Agent Subsystem

**File**: `src/history_book/services/agents/CLAUDE.md` (or similar)

**Content**:
- Overview of LangGraph implementation
- Architecture decisions
- How to extend the graph (add nodes, tools, etc.)
- Comparison with LCEL approach
- When to use agent vs chat API

#### 5.4 Update Root CLAUDE.md

**Tasks**:
- [ ] Add section on agent functionality
- [ ] Update architecture diagram
- [ ] Add new development commands if needed
- [ ] Link to agent subsystem CLAUDE.md

---

## LangGraph Patterns & Best Practices

### Core Concepts (2025 API)

#### State Management

**TypedDict with Reducers**:
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Simple field - replaces on update
    question: str

    # List with reducer - appends on update
    messages: Annotated[list, add_messages]

    # Dict - replaces on update
    metadata: dict
```

**Built-in Reducers**:
- `add_messages` - Appends to message list (LangChain messages)
- Custom reducers possible for complex state updates

#### Graph Construction

**Pattern**:
```python
from langgraph.graph import StateGraph, START, END

# 1. Define state schema
class MyState(TypedDict):
    field1: str
    field2: list

# 2. Create workflow
workflow = StateGraph(MyState)

# 3. Add nodes (functions that receive state)
def my_node(state: MyState) -> dict:
    # Process state
    return {"field1": "updated value"}

workflow.add_node("node_name", my_node)

# 4. Add edges (control flow)
workflow.add_edge(START, "node_name")
workflow.add_edge("node_name", END)

# 5. Compile
graph = workflow.compile()
```

#### Conditional Edges

**For future routing logic**:
```python
def should_continue(state: MyState) -> str:
    """Router function - decides next node"""
    if state["some_condition"]:
        return "node_a"
    else:
        return "node_b"

workflow.add_conditional_edges(
    "current_node",
    should_continue,
    {
        "node_a": "node_a",
        "node_b": "node_b",
    }
)
```

### Streaming Patterns

#### Token Streaming

**Setup**:
```python
# LLM must have streaming enabled
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    streaming=True,  # Required!
)
```

**Usage**:
```python
# Stream with messages mode
for chunk in graph.stream(initial_state, stream_mode="messages"):
    # chunk is a message object with content
    print(chunk.content, end="", flush=True)
```

**Multiple Stream Modes**:
```python
# Combine modes to get both tokens and state updates
for chunk in graph.stream(
    initial_state,
    stream_mode=["messages", "updates"],  # Multiple modes
):
    # Process both token chunks and state updates
```

### Checkpointing Patterns

#### MemorySaver (In-Memory)

**Basic usage**:
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# Use thread_id to maintain separate sessions
config = {"configurable": {"thread_id": "session-123"}}
result = graph.invoke(initial_state, config=config)

# Later invocation with same thread_id will have access to previous state
result2 = graph.invoke(next_state, config=config)
```

**When to use**:
- Development and testing
- Short-lived workflows
- Single-server deployments
- OK with losing state on restart

#### SqliteSaver (Persistent, Local)

**For future use**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = workflow.compile(checkpointer=checkpointer)
```

**When to use**:
- Local development with persistence
- Single-server production deployments
- Need state to survive restarts
- Small to medium scale

#### PostgresSaver (Persistent, Production)

**For future production use**:
```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)
graph = workflow.compile(checkpointer=checkpointer)
```

**When to use**:
- Multi-server production deployments
- Large scale applications
- Shared state across instances
- Advanced querying of checkpoints

### Error Handling Patterns

#### Graceful Degradation

**Pattern**:
```python
def retrieve_node(state: State) -> dict:
    try:
        results = retriever.invoke(state["question"])
        return {"documents": results}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"documents": []}  # Empty list - continue without context

def generate_node(state: State) -> dict:
    documents = state.get("documents", [])

    if documents:
        # Generate with context
        context = format_docs(documents)
        response = llm.invoke(f"{context}\n\n{state['question']}")
    else:
        # Fallback - generate without context
        response = llm.invoke(state["question"])

    return {"response": response}
```

**Benefits**:
- System remains operational despite failures
- User gets response even if retrieval fails
- Mirrors current RagService behavior

### LangSmith Tracing

**Automatic for LangGraph**:
- All graph executions automatically traced
- No decorator needed (unlike `@traceable` in current code)
- Graph structure visualized in LangSmith UI

**Configuration**:
```bash
# Environment variables (already set in project)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=history_book
```

**Adding tags**:
```python
config = {
    "tags": ["agent", "langgraph", "simple_rag"],
    "metadata": {"session_id": session_id, "user_id": user_id},
}
result = graph.invoke(initial_state, config=config)
```

**Benefits over LCEL tracing**:
- Visual graph structure in UI
- See state at each node
- Understand execution flow
- Easier debugging of complex graphs

### RAG-Specific Patterns

#### Adaptive RAG (Future)

**Pattern**: Route between retrieval strategies based on query analysis

```python
def analyze_query(state: State) -> str:
    """Decide if query needs retrieval"""
    query = state["question"]

    # Simple heuristic (could use LLM classification)
    if needs_retrieval(query):
        return "retrieve"
    else:
        return "direct_answer"

workflow.add_conditional_edges(
    START,
    analyze_query,
    {
        "retrieve": "retrieve_node",
        "direct_answer": "generate_node",
    }
)
```

#### Self-Reflective RAG (Future)

**Pattern**: Grade retrieved documents, rewrite query if needed

```python
def grade_documents(state: State) -> str:
    """Assess document relevance"""
    docs = state["documents"]
    question = state["question"]

    relevant = [d for d in docs if is_relevant(d, question)]

    if len(relevant) >= 3:
        return "generate"
    else:
        return "rewrite_query"

def rewrite_query_node(state: State) -> dict:
    """Improve query for better retrieval"""
    better_query = llm.invoke(f"Rewrite for better search: {state['question']}")
    return {"question": better_query}

# Graph: retrieve → grade → (generate | rewrite → retrieve → grade → ...)
```

#### Corrective RAG (Future)

**Pattern**: Add web search fallback for low-quality retrieval

```python
def assess_retrieval_quality(state: State) -> str:
    docs = state["documents"]

    if quality_score(docs) > 0.7:
        return "generate"
    else:
        return "web_search"

def web_search_node(state: State) -> dict:
    """Fallback to web search"""
    results = tavily_search(state["question"])
    return {"documents": results}

# Graph includes web_search as fallback node
```

### Tool Integration (Future)

**Pattern for adding tools**:

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search external knowledge base"""
    return search_api(query)

@tool
def calculator(expression: str) -> float:
    """Evaluate mathematical expression"""
    return eval(expression)

tools = [search_tool, calculator]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

def tool_calling_node(state: State) -> dict:
    """Node that can call tools"""
    result = llm_with_tools.invoke(state["messages"])

    if result.tool_calls:
        # Execute tools
        tool_results = execute_tools(result.tool_calls, tools)
        return {"tool_results": tool_results}
    else:
        return {"generation": result.content}
```

**State schema additions**:
```python
class AgentState(TypedDict):
    # ... existing fields ...
    tool_calls: list[dict]  # LLM's tool call requests
    tool_results: list[dict]  # Results from tool execution
```

---

## Files to Create/Modify

### New Files (~10 files)

#### Core Implementation

1. **`src/history_book/data_models/graph_state.py`**
   - Defines `AgentState` TypedDict
   - Defines `AgentResult` dataclass
   - ~50 lines

2. **`src/history_book/config/graph_config.py`**
   - Defines `GraphConfig` settings
   - Feature flags and parameters
   - ~40 lines

3. **`src/history_book/services/graph_rag_service.py`**
   - Main LangGraph RAG implementation
   - Graph construction, nodes, execution
   - ~200-250 lines

4. **`src/history_book/services/graph_chat_service.py`**
   - Session orchestration for LangGraph
   - Hybrid memory management
   - ~150-200 lines

#### API Layer

5. **`src/history_book/api/routes/agent.py`**
   - All agent endpoints
   - Dependency injection
   - ~150-200 lines

6. **`src/history_book/api/models/agent_models.py`**
   - Request/response models for agent API
   - Pydantic validation
   - ~80-100 lines

#### Testing

7. **`tests/test_graph_rag_service.py`**
   - Unit tests for GraphRagService
   - Node testing, graph execution
   - ~150 lines

8. **`tests/test_graph_chat_service.py`**
   - Unit tests for GraphChatService
   - Session management, persistence
   - ~120 lines

9. **`tests/test_agent_api.py`**
   - API endpoint tests
   - Integration tests
   - ~150 lines

10. **`tests/test_graph_integration.py`**
    - End-to-end integration tests
    - Performance comparisons
    - ~100 lines

#### Documentation

11. **`src/history_book/services/agents/CLAUDE.md`** (optional directory structure)
    - Or just add section to existing CLAUDE.md files
    - LangGraph architecture documentation
    - ~200-300 lines markdown

12. **`docs/plans/langgraph_rag_implementation_plan.md`**
    - This document!

### Modified Files (~3 files)

1. **`pyproject.toml`**
   - Add LangGraph dependencies:
     ```toml
     langgraph = "^1.0"
     langgraph-checkpoint = "^2.0"
     ```

2. **`src/history_book/api/main.py`**
   - Register agent router:
     ```python
     from history_book.api.routes import agent
     app.include_router(agent.router)
     ```

3. **`CLAUDE.md`** (root)
   - Add agent section to architecture overview
   - Add `/api/agent/*` endpoint documentation
   - Link to detailed agent docs
   - ~20-30 lines added

### Unchanged Files (Everything Else)

**No changes to**:
- Existing `RagService` and `ChatService`
- Existing `/api/chat/*` routes
- Repository layer (fully reused)
- Data models (reused)
- Frontend (API-only implementation initially)
- LLM configuration (reused)
- Weaviate setup
- Tests for existing functionality

**Total new code**: ~1,400-1,700 lines
**Total modified code**: ~30-50 lines
**Existing code**: Unchanged

---

## Testing Strategy

### Unit Testing

**Coverage areas**:
- Individual graph nodes (retrieve, generate)
- State transformations
- LLM interaction (mocked)
- Context formatting
- Message conversion utilities

**Approach**:
```python
# Mock dependencies
@pytest.fixture
def mock_paragraph_repo():
    repo = Mock(ParagraphRepository)
    repo.similarity_search_by_text.return_value = [(paragraph, 0.9)]
    return repo

# Test individual nodes
def test_retrieve_node(mock_paragraph_repo):
    service = GraphRagService(paragraph_repository=mock_paragraph_repo, ...)

    state = {"question": "test query", "retrieved_paragraphs": []}
    result = service._retrieve_node(state)

    assert len(result["retrieved_paragraphs"]) > 0
    mock_paragraph_repo.similarity_search_by_text.assert_called_once()
```

### Integration Testing

**Coverage areas**:
- Full graph execution (end-to-end)
- API → Service → Graph → Database
- Multi-turn conversations
- Streaming flows

**Approach**:
```python
def test_full_agent_workflow(test_db):
    """Test complete message flow through agent"""
    # Setup
    service = GraphChatService(...)
    session_id = await service.create_session("Test Session")

    # Send message
    message, paragraphs = await service.send_message(
        session_id=session_id,
        content="What is RAG?",
    )

    # Verify
    assert message.role == MessageRole.ASSISTANT
    assert len(paragraphs) > 0

    # Verify persistence
    messages = await service.get_session_messages(session_id)
    assert len(messages) == 2  # User + Assistant
```

### Comparison Testing

**Goal**: Ensure LangGraph output matches LCEL quality

**Approach**:
```python
def test_output_parity():
    """Compare LCEL vs LangGraph outputs"""
    test_queries = [
        "What caused World War I?",
        "Explain the Treaty of Versailles",
        # ... more test queries
    ]

    for query in test_queries:
        # LCEL
        lcel_result = rag_service.generate_response(query, [])

        # LangGraph
        graph_result = graph_rag_service.invoke(query, [], "test-session")

        # Compare retrieval
        assert_similar_paragraphs(
            lcel_result.source_paragraphs,
            graph_result["retrieved_paragraphs"]
        )

        # Compare response quality (semantic similarity)
        similarity = semantic_similarity(
            lcel_result.response,
            graph_result["generation"]
        )
        assert similarity > 0.8  # High semantic similarity
```

### Performance Testing

**Metrics to track**:
- Latency (p50, p95, p99)
- Memory usage
- Throughput (requests/second)
- Token streaming latency

**Approach**:
```python
import time
import pytest

@pytest.mark.benchmark
def test_graph_latency():
    """Measure graph execution time"""
    iterations = 100
    times = []

    for _ in range(iterations):
        start = time.time()
        result = graph_rag_service.invoke("test query", [], "session")
        times.append(time.time() - start)

    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    print(f"P50: {p50:.3f}s, P95: {p95:.3f}s")

    # Ensure reasonable performance
    assert p50 < 2.0  # 2 second median
    assert p95 < 4.0  # 4 second 95th percentile
```

### Manual Testing Checklist

**API Testing**:
- [ ] Create session via POST `/api/agent/sessions`
- [ ] Send message via POST `/api/agent/sessions/{id}/messages`
- [ ] Stream message via POST `/api/agent/sessions/{id}/stream`
- [ ] Get graph visualization via GET `/api/agent/sessions/{id}/graph`
- [ ] List sessions via GET `/api/agent/sessions`
- [ ] Delete session via DELETE `/api/agent/sessions/{id}`

**LangSmith Testing**:
- [ ] Verify traces appear in LangSmith UI
- [ ] Check graph structure visualization
- [ ] Inspect state at each node
- [ ] Compare with LCEL traces

**Database Testing**:
- [ ] Verify sessions created in Weaviate
- [ ] Verify messages saved correctly
- [ ] Verify retrieved_paragraphs IDs stored
- [ ] Check session timestamps updated

**Error Testing**:
- [ ] Test with invalid session ID (404)
- [ ] Test with empty message (400)
- [ ] Test with Weaviate down (500)
- [ ] Test with LLM error (graceful fallback)

---

## Future Extensibility

### Tool Calling Architecture

**Goal**: Enable agent to use external tools (search, calculator, APIs, etc.)

#### State Schema Changes

```python
class AgentState(TypedDict):
    # ... existing fields ...

    # Tool calling additions
    tool_calls: list[dict]  # LLM's requested tool calls
    tool_results: list[dict]  # Results from tool execution
    available_tools: list[str]  # Which tools are enabled
```

#### New Graph Structure

```
START → retrieve → generate_or_use_tools → [tools → generate] → END
                                         ↓
                                    [generate] → END
```

#### Tool Node Implementation

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information"""
    # Integration with Tavily, Google, etc.
    pass

@tool
def calculator(expression: str) -> float:
    """Evaluate mathematical expressions"""
    return eval(expression)

def tools_node(state: AgentState) -> dict:
    """Execute requested tools"""
    tool_calls = state.get("tool_calls", [])
    results = []

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        # Execute tool
        result = execute_tool(tool_name, tool_args)
        results.append(result)

    return {"tool_results": results}

def should_use_tools(state: AgentState) -> str:
    """Router: tools or direct generation?"""
    if state.get("tool_calls"):
        return "tools"
    else:
        return "generate"

# Add to graph
workflow.add_node("tools", tools_node)
workflow.add_conditional_edges(
    "generate_or_use_tools",
    should_use_tools,
    {"tools": "tools", "generate": END}
)
workflow.add_edge("tools", "generate")  # Generate after tools
```

#### Configuration

```python
class GraphConfig(BaseSettings):
    # ... existing config ...

    # Tool configuration
    enabled_tools: list[str] = ["search_web", "calculator"]
    max_tool_iterations: int = 3
```

### Multi-Step Reasoning Architecture

**Goal**: Enable planning, self-correction, iterative refinement

#### State Schema Changes

```python
class AgentState(TypedDict):
    # ... existing fields ...

    # Reasoning additions
    reasoning_steps: list[dict]  # Track thought process
    plan: str | None  # High-level plan
    reflection: str | None  # Self-critique
    iteration_count: int  # Prevent infinite loops
```

#### New Graph Structure (Adaptive RAG)

```
START → plan → retrieve → grade_docs → [good → generate → reflect → END]
                                     ↓
                               [poor → rewrite_query → retrieve → ...]
```

#### Example Nodes

```python
def plan_node(state: AgentState) -> dict:
    """Break down complex query into steps"""
    question = state["question"]

    plan_prompt = f"Create a plan to answer: {question}"
    plan = llm.invoke(plan_prompt)

    return {
        "plan": plan.content,
        "reasoning_steps": [{"type": "plan", "content": plan.content}]
    }

def grade_documents_node(state: AgentState) -> dict:
    """Assess relevance of retrieved docs"""
    docs = state["retrieved_paragraphs"]
    question = state["question"]

    # Use LLM to grade each doc
    grades = [grade_doc(doc, question) for doc in docs]
    relevant_docs = [d for d, g in zip(docs, grades) if g == "relevant"]

    return {
        "retrieved_paragraphs": relevant_docs,
        "reasoning_steps": state["reasoning_steps"] + [{
            "type": "grading",
            "num_relevant": len(relevant_docs),
            "num_total": len(docs),
        }]
    }

def reflect_node(state: AgentState) -> dict:
    """Self-critique the generated response"""
    response = state["generation"]
    question = state["question"]

    critique_prompt = f"Critique this answer to '{question}': {response}"
    critique = llm.invoke(critique_prompt)

    return {
        "reflection": critique.content,
        "reasoning_steps": state["reasoning_steps"] + [{
            "type": "reflection",
            "content": critique.content,
        }]
    }

def should_rewrite_query(state: AgentState) -> str:
    """Router: docs good enough or rewrite query?"""
    docs = state["retrieved_paragraphs"]
    iteration = state.get("iteration_count", 0)

    if len(docs) >= 3 or iteration >= 2:
        return "generate"
    else:
        return "rewrite"
```

#### Configuration

```python
class GraphConfig(BaseSettings):
    # ... existing config ...

    # Reasoning configuration
    enable_planning: bool = False
    enable_reflection: bool = False
    max_reasoning_iterations: int = 3
```

### API Enhancements for Future Features

**Enhanced metadata in responses**:

```python
class AgentMessageResponse(BaseModel):
    # ... existing fields ...

    metadata: dict = Field(default_factory=dict)
    # Example metadata:
    # {
    #     "graph_execution": "adaptive_rag",
    #     "nodes_executed": ["plan", "retrieve", "grade", "generate", "reflect"],
    #     "reasoning_steps": [
    #         {"type": "plan", "content": "..."},
    #         {"type": "grading", "num_relevant": 5},
    #         {"type": "reflection", "content": "..."},
    #     ],
    #     "tools_used": ["search_web", "calculator"],
    #     "tool_results": [...],
    #     "iteration_count": 2,
    #     "execution_time_ms": 3456,
    # }
```

**New endpoints for advanced features**:

```python
# Configure tools for a session
@router.post("/sessions/{session_id}/tools")
async def configure_tools(session_id: str, tools: list[str]):
    """Enable specific tools for this session"""
    pass

# Get reasoning trace
@router.get("/sessions/{session_id}/messages/{message_id}/reasoning")
async def get_reasoning_trace(session_id: str, message_id: str):
    """Get detailed reasoning steps for a message"""
    pass
```

### Migration Path to Advanced Features

**Phase 1** (Current plan): Simple RAG
- retrieve → generate
- Foundation for future

**Phase 2** (Near future): Tool calling
- Add tools_node
- Add conditional routing
- Enable web search, calculator, etc.

**Phase 3** (Medium future): Adaptive RAG
- Add query analysis
- Add document grading
- Add query rewriting

**Phase 4** (Long future): Multi-agent
- Multiple specialized graphs
- Coordination layer
- Supervisor pattern

**Key principle**: State schema designed to accommodate all future features without breaking changes. Add fields as needed, old code ignores them.

---

## References

### LangGraph Documentation

**Official Docs**:
- Overview: https://docs.langchain.com/oss/python/langgraph/overview
- Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
- Checkpointing: https://langchain-ai.github.io/langgraph/ (docs have moved)
- API Reference: https://langchain-ai.github.io/langgraph/reference/

**Tutorials**:
- Adaptive RAG: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
- Self-Reflective RAG: https://blog.langchain.com/agentic-rag-with-langgraph/
- Getting Started: https://academy.langchain.com/courses/intro-to-langgraph

### Research & Articles

**2025 Guides**:
- LangGraph Tutorial (Latenode): https://latenode.com/blog/langgraph-tutorial-complete-beginners-guide-to-getting-started
- DataCamp LangGraph Tutorial: https://www.datacamp.com/tutorial/langgraph-tutorial
- Agentic RAG Guide (Medium): https://medium.com/@wendell_89912/building-an-agentic-rag-with-langgraph-a-step-by-step-guide-009c5f0cce0a

**Architecture Patterns**:
- Multi-Agent Orchestration: https://latenode.com/blog/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025
- State Machines: https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4

### LangChain Documentation

**Core Concepts**:
- LCEL (LangChain Expression Language): https://python.langchain.com/docs/expression_language/
- Messages: https://python.langchain.com/docs/modules/model_io/messages/
- Chat Models: https://python.langchain.com/docs/modules/model_io/chat/

**Tools & Agents**:
- Tools: https://python.langchain.com/docs/modules/agents/tools/
- Agents: https://python.langchain.com/docs/modules/agents/

### LangSmith

**Tracing & Debugging**:
- LangSmith Docs: https://docs.langsmith.com/
- Debugging: https://langsmith.docs.com/debugging
- Tracing: https://langsmith.docs.com/tracing

### Project-Specific

**Current Implementation**:
- `/src/history_book/services/CLAUDE.md` - Service layer docs
- `/src/history_book/database/CLAUDE.md` - Repository pattern docs
- `/src/history_book/api/CLAUDE.md` - API docs
- `/CLAUDE.md` - Root project documentation

### Additional Resources

**RAG Patterns**:
- Corrective RAG (CRAG): https://www.datacamp.com/tutorial/corrective-rag-crag
- Self-RAG: https://www.datacamp.com/tutorial/self-rag
- RAG Best Practices: https://md-hadi.medium.com/mastering-rag-build-smarter-ai-with-langchain-and-langgraph-in-2025-cc126fb8a552

**Python Libraries**:
- LangGraph PyPI: https://pypi.org/project/langgraph/
- LangChain PyPI: https://pypi.org/project/langchain/

---

## Appendix: Key Decisions Summary

### Why LangGraph?

**Advantages over LCEL**:
1. **State management**: Built-in state tracking across nodes
2. **Conditional flows**: Easy routing and branching logic
3. **Checkpointing**: Resumable workflows, human-in-the-loop
4. **Extensibility**: Clear path to tool calling, multi-step reasoning
5. **Debugging**: Better tracing and visualization in LangSmith
6. **Composability**: Easier to add/remove/modify nodes

**Trade-offs**:
1. **Complexity**: More overhead for simple RAG (but worth it for future)
2. **Learning curve**: New paradigm vs familiar LCEL chains
3. **Performance**: Small latency overhead from graph orchestration

### Why Hybrid Memory?

**LangGraph MemorySaver** for:
- In-graph state during execution
- Conditional routing decisions
- Human-in-the-loop future feature
- Thread-based isolation

**Weaviate** for:
- Long-term chat history
- Cross-session analytics
- Vector search capabilities
- Production-grade persistence

**Rationale**: Best of both worlds - LangGraph's powerful state management during execution, Weaviate's robust long-term storage.

### Why New Service Layer?

**Alternative considered**: Extend existing RagService/ChatService

**Decision**: New GraphRagService/GraphChatService

**Rationale**:
1. Clean separation of concerns
2. Easier to maintain both implementations
3. No risk of breaking existing chat functionality
4. Allows for different configuration/initialization
5. Follows "add not replace" goal
6. Easier to A/B test and compare

### Why Independent API?

**Alternative considered**: Share `/api/chat/*` namespace

**Decision**: New `/api/agent/*` namespace

**Rationale**:
1. Signals different capabilities (agentic vs simple chat)
2. Freedom to design optimal response format
3. No backwards compatibility constraints
4. Can include graph-specific features (visualization, checkpoints)
5. Easier to version and evolve independently

### Why MemorySaver Only?

**Alternatives considered**: PostgresSaver, SqliteSaver

**Decision**: MemorySaver for initial implementation

**Rationale**:
1. Personal project - no multi-server requirement
2. RAG executes quickly (1-2 seconds) - no long-running workflows
3. Long-term persistence handled by Weaviate
4. Simpler setup and maintenance
5. Easy to upgrade later if needed
6. Sufficient for all planned features

---

## Next Steps

After this plan is approved:

1. **Review and refine** - Read through, ask questions, clarify anything
2. **Set up development environment** - Ensure Weaviate, LangSmith ready
3. **Install dependencies** - Add LangGraph to project
4. **Start with Phase 1** - Core graph implementation
5. **Test incrementally** - Unit tests as you build
6. **Compare with LCEL** - Ensure parity
7. **Iterate** - Refine based on learnings

**Estimated timeline**:
- Phase 1 (Core): 2-3 days
- Phase 2 (API): 1-2 days
- Phase 3 (Features): 1 day
- Phase 4 (Testing): 1-2 days
- Phase 5 (Docs): 1 day

**Total**: ~1 week of focused development

---

**End of Plan Document**

This plan provides a comprehensive roadmap for implementing LangGraph-based RAG while maintaining the existing system and positioning for future agentic capabilities. The hybrid memory approach balances simplicity with extensibility, and the phased implementation reduces risk while enabling incremental progress.
