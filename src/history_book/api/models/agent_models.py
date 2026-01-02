"""Pydantic models for Agent API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field


class AgentSessionCreateRequest(BaseModel):
    """Request to create a new agent session."""

    title: str | None = None


class AgentMessageRequest(BaseModel):
    """Request to send a message to the agent."""

    content: str = Field(..., min_length=1, max_length=10000)

    # Optional retrieval configuration overrides
    max_context_paragraphs: int | None = Field(default=None, ge=1, le=100)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentMessageResponse(BaseModel):
    """Agent message with graph execution metadata."""

    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    session_id: str
    citations: list[str] | None = None  # e.g., ["Page 42", "Page 67"]

    # Graph execution metadata (new compared to chat API)
    metadata: dict = Field(default_factory=dict)
    # Example metadata:
    # {
    #     "num_retrieved_paragraphs": 5,
    #     "graph_execution": "tool_enabled_rag",
    #     "nodes_executed": ["generate", "tools", "generate"],
    #     "tool_calls": [
    #         {"tool": "search_book", "query": "Ancient Rome"}
    #     ],
    #     "tool_iterations": 1
    # }


class AgentSessionResponse(BaseModel):
    """Response containing agent session information."""

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class AgentSessionListResponse(BaseModel):
    """Response containing a list of recent agent sessions."""

    sessions: list[AgentSessionResponse]


class AgentMessageListResponse(BaseModel):
    """Response containing a list of messages for an agent session."""

    messages: list[AgentMessageResponse]


class AgentChatResponse(BaseModel):
    """Response after sending a message - includes the AI's reply and updated session."""

    message: AgentMessageResponse
    session: AgentSessionResponse


class GraphVisualization(BaseModel):
    """Graph structure for debugging."""

    mermaid: str  # Mermaid diagram syntax
    nodes: list[str]
    edges: list[tuple[str, str]]
