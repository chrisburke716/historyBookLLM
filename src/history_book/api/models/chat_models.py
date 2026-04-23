"""Pydantic models for the Chat API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    title: str | None = None


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    id: str
    content: str
    role: str
    timestamp: datetime
    session_id: str
    citations: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]


class ChatResponse(BaseModel):
    """Response after sending a message — includes AI reply and updated session (with title)."""

    message: MessageResponse
    session: SessionResponse


class GraphVisualization(BaseModel):
    mermaid: str
    nodes: list[str]
    edges: list[tuple[str, str]]
