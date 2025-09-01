"""Chat API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from history_book.api.models.api_models import (
    ChatResponse,
    MessageListResponse,
    MessageRequest,
    MessageResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
)
from history_book.data_models.entities import ChatMessage, ChatSession
from history_book.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# Dependency to get ChatService instance
# Note: We don't close the service per request since it shares a global client
def get_chat_service():
    """Get a ChatService instance."""
    return ChatService()


def convert_session_to_response(session: ChatSession) -> SessionResponse:
    """Convert ChatSession entity to SessionResponse."""
    return SessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def convert_message_to_response(
    message: ChatMessage, chat_service: ChatService = None
) -> MessageResponse:
    """Convert ChatMessage entity to MessageResponse."""
    citations = None
    if message.retrieved_paragraphs and chat_service:
        # Get actual page numbers from retrieved paragraphs
        citations = []
        for para_id in message.retrieved_paragraphs:
            try:
                # Get paragraph from database to extract page number
                # TODO: don't love the idea of accessing repository_manager from chat_service here. might make that a singleton?
                paragraph = chat_service.repository_manager.paragraphs.get_by_id(
                    para_id
                )
                if paragraph:
                    citations.append(f"Page {paragraph.page}")
            except Exception:
                # Fallback to generic citation if paragraph fetch fails
                citations.append("Source document")
    elif message.retrieved_paragraphs:
        # Fallback when chat_service not available
        citations = [
            f"Source {i}" for i in range(1, len(message.retrieved_paragraphs) + 1)
        ]

    return MessageResponse(
        id=message.id,
        content=message.content,
        role=str(message.role),
        timestamp=message.timestamp,
        session_id=message.session_id,
        citations=citations,
    )


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest, chat_service: ChatService = Depends(get_chat_service)
):
    """Create a new chat session."""
    try:
        session = await chat_service.create_session(request.title)
        return convert_session_to_response(session)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session") from e


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(
    limit: int = 10, chat_service: ChatService = Depends(get_chat_service)
):
    """Get recent chat sessions."""
    try:
        sessions = await chat_service.list_recent_sessions(limit)
        session_responses = [convert_session_to_response(s) for s in sessions]
        return SessionListResponse(sessions=session_responses)
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve sessions"
        ) from e


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str, chat_service: ChatService = Depends(get_chat_service)
):
    """Get all messages for a session."""
    try:
        messages = await chat_service.get_session_messages(session_id)
        message_responses = [
            convert_message_to_response(m, chat_service) for m in messages
        ]
        return MessageListResponse(messages=message_responses)
    except Exception as e:
        logger.error(f"Failed to get messages for session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve messages"
        ) from e


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_message(
    session_id: str,
    request: MessageRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Send a message and get AI response."""
    try:
        # Check if session exists
        session = await chat_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Send message and get response
        ai_message = await chat_service.send_message(
            session_id=session_id,
            user_message=request.content,
            enable_retrieval=request.enable_retrieval,
        )

        response_message = convert_message_to_response(ai_message, chat_service)
        return ChatResponse(message=response_message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message") from e
