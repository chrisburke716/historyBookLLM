"""Agent API routes for LangGraph-based chat."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from history_book.api.models.agent_models import (
    AgentChatResponse,
    AgentMessageListResponse,
    AgentMessageRequest,
    AgentMessageResponse,
    AgentSessionCreateRequest,
    AgentSessionListResponse,
    AgentSessionResponse,
    GraphVisualization,
)
from history_book.data_models.entities import ChatMessage, ChatSession, Paragraph
from history_book.services.graph_chat_service import GraphChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def get_graph_chat_service():
    """Get a GraphChatService instance."""
    return GraphChatService()


def convert_session_to_response(session: ChatSession) -> AgentSessionResponse:
    """Convert ChatSession entity to AgentSessionResponse."""
    return AgentSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def convert_message_to_response(
    message: ChatMessage,
    retrieved_paragraphs: list[Paragraph] | None = None,
    metadata: dict | None = None,
) -> AgentMessageResponse:
    """Convert ChatMessage entity to AgentMessageResponse."""
    citations = None
    if retrieved_paragraphs:
        citations = [f"Page {para.page}" for para in retrieved_paragraphs]
    elif message.retrieved_paragraphs:
        citations = [
            f"Source {i}" for i in range(1, len(message.retrieved_paragraphs) + 1)
        ]

    # Build metadata
    response_metadata = metadata or {}
    if retrieved_paragraphs is not None:
        response_metadata["num_retrieved_paragraphs"] = len(retrieved_paragraphs)
    response_metadata.setdefault("graph_execution", "simple_rag")

    return AgentMessageResponse(
        id=message.id,
        content=message.content,
        role=str(message.role),
        timestamp=message.timestamp,
        session_id=message.session_id,
        citations=citations,
        metadata=response_metadata,
    )


@router.post("/sessions", response_model=AgentSessionResponse)
async def create_session(
    request: AgentSessionCreateRequest,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """Create a new agent chat session."""
    try:
        session = await service.create_session(request.title)
        return convert_session_to_response(session)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session") from e


@router.get("/sessions", response_model=AgentSessionListResponse)
async def list_sessions(
    limit: int = 10,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """List recent agent sessions."""
    try:
        sessions = await service.list_recent_sessions(limit)
        session_responses = [convert_session_to_response(s) for s in sessions]
        return AgentSessionListResponse(sessions=session_responses)
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve sessions"
        ) from e


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """Delete a session and all its messages."""
    try:
        success = await service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session") from e


@router.get("/sessions/{session_id}/messages", response_model=AgentMessageListResponse)
async def get_messages(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """Get all messages for a session."""
    try:
        messages = await service.get_session_messages(session_id)
        message_responses = [convert_message_to_response(m) for m in messages]
        return AgentMessageListResponse(messages=message_responses)
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve messages"
        ) from e


@router.post("/sessions/{session_id}/messages", response_model=AgentChatResponse)
async def send_message(
    session_id: str,
    request: AgentMessageRequest,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    Send a message to the agent (non-streaming).

    Returns complete response with graph execution metadata.
    """
    try:
        # Verify session exists
        session = await service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Send message through graph
        result = await service.send_message(
            session_id=session_id,
            user_message=request.content,
        )

        # Convert to response with metadata
        message_response = convert_message_to_response(
            result.message,
            retrieved_paragraphs=result.retrieved_paragraphs,
            metadata=result.metadata,
        )

        return AgentChatResponse(message=message_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message") from e


@router.post("/sessions/{session_id}/stream")
async def stream_message(
    session_id: str,
    request: AgentMessageRequest,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    Send a message with streaming response (Server-Sent Events).

    Returns token-by-token chunks as they are generated.
    """
    try:
        # Verify session exists
        session = await service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        async def event_generator():
            """Generate SSE events for streaming response."""
            try:
                # Unpack generator and paragraphs from send_message_stream
                stream_gen, _ = await service.send_message_stream(
                    session_id=session_id,
                    user_message=request.content,
                )

                async for chunk in stream_gen:
                    # Send chunk as SSE data
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}")
        raise HTTPException(status_code=500, detail="Failed to start streaming") from e


@router.get("/sessions/{session_id}/graph", response_model=GraphVisualization)
async def get_graph_visualization(
    session_id: str,
    service: GraphChatService = Depends(get_graph_chat_service),
):
    """
    Get graph structure visualization for debugging.

    Returns Mermaid diagram text and graph structure.
    """
    try:
        # Verify session exists
        session = await service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get graph structure from GraphRagService
        graph = service.graph_rag.graph

        # Generate Mermaid diagram
        try:
            mermaid = graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.warning(f"Failed to generate Mermaid diagram: {e}")
            mermaid = "graph TD\n    START --> retrieve\n    retrieve --> generate\n    generate --> END"

        return GraphVisualization(
            mermaid=mermaid,
            nodes=["retrieve", "generate"],
            edges=[
                ("START", "retrieve"),
                ("retrieve", "generate"),
                ("generate", "END"),
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get graph visualization: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get graph visualization"
        ) from e
