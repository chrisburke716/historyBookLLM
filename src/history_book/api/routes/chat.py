"""Chat API routes."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from history_book.api.models.chat_models import (
    ChatResponse,
    GraphVisualization,
    MessageListResponse,
    MessageRequest,
    MessageResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
)
from history_book.data_models.entities import ChatMessage, ChatSession, Paragraph
from history_book.services.chat_service import ChatResult, ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_service() -> ChatService:
    return ChatService()


def _session_response(session: ChatSession) -> SessionResponse:
    return SessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _message_response(
    message: ChatMessage,
    retrieved_paragraphs: list[Paragraph] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MessageResponse:
    citations = None
    if retrieved_paragraphs:
        citations = [f"Page {p.page}" for p in retrieved_paragraphs]
    elif message.retrieved_paragraphs:
        citations = [
            f"Source {i}" for i in range(1, len(message.retrieved_paragraphs) + 1)
        ]

    response_metadata = metadata or {}
    if retrieved_paragraphs is not None:
        response_metadata["num_retrieved_paragraphs"] = len(retrieved_paragraphs)

    return MessageResponse(
        id=message.id,
        content=message.content,
        role=str(message.role),
        timestamp=message.timestamp,
        session_id=message.session_id,
        citations=citations,
        metadata=response_metadata,
    )


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    service: ChatService = Depends(get_chat_service),
) -> SessionResponse:
    try:
        session = await service.create_session(request.title)
        return _session_response(session)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session") from e


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 10,
    service: ChatService = Depends(get_chat_service),
) -> SessionListResponse:
    try:
        sessions = await service.list_recent_sessions(limit)
        return SessionListResponse(sessions=[_session_response(s) for s in sessions])
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve sessions"
        ) from e


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    try:
        success = await service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session") from e


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> MessageListResponse:
    try:
        messages = await service.get_session_messages(session_id)
        return MessageListResponse(messages=[_message_response(m) for m in messages])
    except Exception as e:
        logger.error(f"Failed to get messages for {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve messages"
        ) from e


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_message(
    session_id: str,
    request: MessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        session = await service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        result: ChatResult = await service.send_message(
            session_id=session_id,
            user_message=request.content,
        )

        updated_session = await service.get_session(session_id)
        if not updated_session:
            raise HTTPException(
                status_code=500, detail="Session disappeared during processing"
            )

        return ChatResponse(
            message=_message_response(
                result.message,
                retrieved_paragraphs=result.retrieved_paragraphs,
                metadata=result.metadata,
            ),
            session=_session_response(updated_session),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message to {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message") from e


@router.post("/sessions/{session_id}/stream")
async def stream_message(
    session_id: str,
    request: MessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Send a message with token-by-token streaming (Server-Sent Events)."""
    try:
        session = await service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        async def event_generator() -> AsyncIterator[str]:
            try:
                stream_gen, _ = await service.send_message_stream(
                    session_id=session_id,
                    user_message=request.content,
                )
                async for chunk in stream_gen:
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Streaming error for {session_id}: {e}")
                yield f"data: [ERROR] {e}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start streaming for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start streaming") from e


@router.get("/sessions/{session_id}/graph", response_model=GraphVisualization)
async def get_graph_visualization(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> GraphVisualization:
    """Get the agent graph structure as a Mermaid diagram."""
    try:
        session = await service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            mermaid = service.agent.get_graph().draw_mermaid()
        except Exception:
            mermaid = "graph TD\n    __start__ --> agent\n    agent --> tools\n    tools --> agent\n    agent --> __end__"

        return GraphVisualization(
            mermaid=mermaid,
            nodes=["agent", "tools"],
            edges=[
                ("START", "agent"),
                ("agent", "tools"),
                ("tools", "agent"),
                ("agent", "END"),
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get graph for {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get graph visualization"
        ) from e
