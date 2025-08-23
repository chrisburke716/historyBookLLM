"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class SessionCreateRequest(BaseModel):
    """Request to create a new chat session."""
    title: Optional[str] = None


class MessageRequest(BaseModel):
    """Request to send a message in a chat session."""
    content: str
    enable_retrieval: bool = True
    max_context_paragraphs: int = 5


class SessionResponse(BaseModel):
    """Response containing chat session information."""
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """Response containing a chat message."""
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    session_id: str
    citations: Optional[List[str]] = None  # Simple page references like "Page 123"


class SessionListResponse(BaseModel):
    """Response containing a list of recent sessions."""
    sessions: List[SessionResponse]


class MessageListResponse(BaseModel):
    """Response containing a list of messages for a session."""
    messages: List[MessageResponse]


class ChatResponse(BaseModel):
    """Response after sending a message - includes the AI's reply."""
    message: MessageResponse