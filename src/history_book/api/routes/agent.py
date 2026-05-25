"""AG-UI agent endpoint — wraps `ChatService.send_message_agui` for SSE.

The route is HTTP/SSE only: validate the request, set up the encoder, wrap the
service iterator with transport encoding and an error-frame fallback. All
persistence, agent invocation, and graph-state introspection live in
`ChatService.send_message_agui`.
"""

import logging

from ag_ui.core import EventType, RunAgentInput, RunErrorEvent
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from history_book.services.chat_service import ChatService

from .chat import get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/agent")
async def agent_endpoint(
    input_data: RunAgentInput,
    request: Request,
    service: ChatService = Depends(get_chat_service),
):
    """AG-UI streaming endpoint for the RAG agent."""
    session_id = input_data.thread_id
    if not session_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    if (await service.get_session(session_id)) is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def event_generator():
        try:
            async for event in service.send_message_agui(input_data):
                yield encoder.encode(event)
        except Exception as e:
            logger.exception(f"AG-UI run failed for session {session_id}: {e}")
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=f"Agent run failed: {e}",
                )
            )

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type(),
    )
