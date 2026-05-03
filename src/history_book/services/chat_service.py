"""Chat service — session orchestration for the RAG agent."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langsmith import traceable

from history_book.chains.title_generation_chain import create_title_generation_chain
from history_book.data_models.entities import (
    ChatMessage,
    ChatSession,
    MessageRole,
    Paragraph,
)
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager
from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError
from history_book.llm.factory import build_chat_model
from history_book.services.agents.context import AgentContext
from history_book.services.agents.rag_agent import build_rag_agent
from history_book.services.kg_service import KGService

logger = logging.getLogger(__name__)

CONTEXT_MIN_RESULTS = 5
CONTEXT_MAX_RESULTS = 40
CONTEXT_SIMILARITY_CUTOFF = 0.4


@dataclass
class ChatResult:
    """Result from a send_message call."""

    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
    metadata: dict[str, Any] | None = None


class ChatService:
    """
    Orchestrates chat sessions using the RAG agent.

    Responsible for Weaviate persistence (sessions + messages) and
    delegating generation to the compiled LangGraph agent.

    Memory strategy:
    - MemorySaver (in-graph): full message history including tool calls/results, lost on restart
    - Weaviate: session + USER/ASSISTANT message storage for the UI and evals
    """

    def __init__(
        self,
        config: WeaviateConfig | None = None,
        llm_config: LLMConfig | None = None,
        min_context_results: int = CONTEXT_MIN_RESULTS,
        max_context_results: int = CONTEXT_MAX_RESULTS,
        context_similarity_cutoff: float = CONTEXT_SIMILARITY_CUTOFF,
        enabled_tools: list[str] | None = None,
    ):
        if config is None:
            config = WeaviateConfig.from_environment()
        self.repository_manager = BookRepositoryManager(config)
        self.kg_service = KGService(repo_manager=self.repository_manager)
        self.volume_graph_name = self._resolve_volume_graph_name()

        self.llm_config = llm_config or LLMConfig.from_environment()
        self.llm_config.validate()
        logger.info(f"LLM: {self.llm_config.provider}/{self.llm_config.model_name}")

        self.min_context_results = min_context_results
        self.max_context_results = max_context_results
        self.context_similarity_cutoff = context_similarity_cutoff
        self.enabled_tools = enabled_tools

        self.agent = build_rag_agent(enabled_tools=enabled_tools)

    def _resolve_volume_graph_name(self) -> str | None:
        """Find a graph_name to use for KG tools.

        Prefers the volume graph; falls back to the largest book graph; logs a
        warning if no graph exists.
        """
        try:
            graphs = self.kg_service.list_graphs()
        except Exception as e:
            logger.warning(f"Failed to list KG graphs: {e}")
            return None
        if not graphs:
            logger.warning("No KG graphs found — KG tools will be unavailable")
            return None
        volume = [g for g in graphs if g.graph_type == "volume"]
        if volume:
            chosen = volume[0]
        else:
            books = sorted(
                [g for g in graphs if g.graph_type == "book"],
                key=lambda g: g.entity_count,
                reverse=True,
            )
            if not books:
                logger.warning(
                    "No volume or book graphs found — KG tools will be unavailable"
                )
                return None
            chosen = books[0]
            logger.warning(
                f"No volume graph; falling back to largest book graph '{chosen.name}'"
            )
        logger.info(
            f"KG graph for chat tools: {chosen.name} "
            f"({chosen.entity_count} entities, {chosen.relationship_count} rels)"
        )
        return chosen.name

    # -------------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------------

    async def create_session(self, title: str | None = None) -> ChatSession:
        session = ChatSession(title=title)
        session_id = self.repository_manager.chat_sessions.create(session)
        session.id = session_id
        logger.info(f"Created session {session_id}")
        return session

    async def get_session(self, session_id: str) -> ChatSession | None:
        try:
            return self.repository_manager.chat_sessions.get_by_id(session_id)
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def list_recent_sessions(self, limit: int = 10) -> list[ChatSession]:
        try:
            return self.repository_manager.chat_sessions.find_recent_sessions(limit)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def get_session_messages(self, session_id: str) -> list[ChatMessage]:
        try:
            return self.repository_manager.chat_messages.find_by_session(session_id)
        except Exception as e:
            logger.error(f"Failed to get messages for {session_id}: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        try:
            messages = await self.get_session_messages(session_id)
            for msg in messages:
                if msg.id:
                    self.repository_manager.chat_messages.delete(msg.id)
            self.repository_manager.chat_sessions.delete(session_id)
            logger.info(f"Deleted session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Messaging
    # -------------------------------------------------------------------------

    @traceable(name="Chat Service: Send Message")
    async def send_message(self, session_id: str, user_message: str) -> ChatResult:
        """
        Send a message and return the AI response.

        Flow:
        1. Save user message to Weaviate
        2. Invoke agent (MemorySaver provides full history including tool messages)
        3. Save AI response to Weaviate
        4. Regenerate session title
        """
        try:
            await self._save_user_message(session_id, user_message)

            ctx = self._build_context()
            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                context=ctx,
                config=self._agent_config(session_id),
            )

            generation = self._extract_generation(result)
            retrieved = result.get("retrieved_paragraphs", [])

            ai_msg = await self._save_ai_message(session_id, generation, retrieved)
            await self._maybe_regenerate_title(session_id)

            return ChatResult(
                message=ai_msg,
                retrieved_paragraphs=retrieved,
                metadata={
                    "num_retrieved_paragraphs": len(retrieved),
                    "tool_iterations": self._count_tool_iterations(result["messages"]),
                },
            )

        except LLMError:
            raise
        except Exception as e:
            logger.error(f"send_message failed: {e}")
            raise

    async def send_message_stream(
        self, session_id: str, user_message: str
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Send a message with token-by-token streaming.

        Yields typed events:
          - {"type": "token", "text": "..."}              — incremental text
          - {"type": "reset"}                             — drop buffered text
            (an intermediate-turn boundary; everything streamed before this
            was preamble before a tool call, not part of the final answer)
          - {"type": "done", "message": {...},
             "session": {...}}                            — terminal event
          - {"type": "error", "message": "..."}           — on failure
        """
        await self._save_user_message(session_id, user_message)
        ctx = self._build_context()
        retrieved: list[Paragraph] = []
        full_response = ""

        try:
            async for mode, data in self.agent.astream(
                {"messages": [HumanMessage(content=user_message)]},
                context=ctx,
                config=self._agent_config(session_id, streaming=True),
                stream_mode=["updates", "messages"],
            ):
                if mode == "messages":
                    token_chunk, _meta = data
                    # `.text` handles both string content and Responses API
                    # list-of-blocks chunks (skipping reasoning blocks).
                    chunk_text = token_chunk.text
                    if chunk_text:
                        full_response += chunk_text
                        yield {"type": "token", "text": chunk_text}
                elif mode == "updates" and "tools" in data:
                    # `data["tools"]` is either a dict (single Command) or a
                    # list of dicts (parallel tool calls — one Command each).
                    tools_update = data["tools"]
                    updates = (
                        tools_update
                        if isinstance(tools_update, list)
                        else [tools_update]
                    )
                    for upd in updates:
                        retrieved.extend(upd.get("retrieved_paragraphs", []))
                    # A tools update means the agent's previous tokens were
                    # preamble before a tool call, not the final answer.
                    # Discard them and tell the client to clear its buffer.
                    full_response = ""
                    yield {"type": "reset"}

            # Save best-effort. If embedding/save fails (e.g., content too
            # long for the embedding model), keep the response visible to
            # the user instead of aborting the whole turn.
            try:
                ai_msg = await self._save_ai_message(
                    session_id, full_response, retrieved
                )
                ai_msg_id = ai_msg.id
                ai_msg_ts = ai_msg.timestamp.isoformat()
            except Exception as save_err:
                logger.warning(
                    f"Failed to persist AI message for {session_id}: {save_err}"
                )
                ai_msg_id = f"unsaved-{datetime.now(UTC).timestamp()}"
                ai_msg_ts = datetime.now(UTC).isoformat()

            try:
                await self._maybe_regenerate_title(session_id)
            except Exception as title_err:
                logger.warning(f"Title regeneration failed: {title_err}")

            session = await self.get_session(session_id)

            citations = [f"Page {p.page}" for p in retrieved] if retrieved else None

            yield {
                "type": "done",
                "message": {
                    "id": ai_msg_id,
                    "content": full_response,
                    "role": "assistant",
                    "timestamp": ai_msg_ts,
                    "session_id": session_id,
                    "citations": citations,
                    "metadata": {
                        "num_retrieved_paragraphs": len(retrieved),
                    },
                },
                "session": {
                    "id": session.id,
                    "title": session.title,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                }
                if session
                else None,
            }
        except Exception as e:
            logger.error(f"send_message_stream failed for {session_id}: {e}")
            yield {"type": "error", "message": str(e)}

    # -------------------------------------------------------------------------
    # Eval support
    # -------------------------------------------------------------------------

    def get_eval_metadata(self) -> dict[str, Any]:
        return {
            "llm_provider": self.llm_config.provider,
            "llm_model": self.llm_config.model_name,
            "llm_temperature": self.llm_config.temperature,
            "llm_max_tokens": self.llm_config.max_tokens,
            "max_context_results": self.max_context_results,
            "context_similarity_cutoff": self.context_similarity_cutoff,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_context(self) -> AgentContext:
        return AgentContext(
            llm_config=self.llm_config,
            repository_manager=self.repository_manager,
            kg_service=self.kg_service,
            volume_graph_name=self.volume_graph_name,
            tool_max_results=self.max_context_results,
            tool_min_similarity=self.context_similarity_cutoff,
        )

    def _agent_config(self, session_id: str, streaming: bool = False) -> dict[str, Any]:
        tags = ["agent", "langgraph", "rag"]
        if streaming:
            tags.append("streaming")
        return {
            "configurable": {"thread_id": session_id},
            "tags": tags,
        }

    async def _save_user_message(
        self, session_id: str, user_message: str
    ) -> ChatMessage:
        """Persist the user message to Weaviate."""
        user_msg = ChatMessage(
            content=user_message, role=MessageRole.USER, session_id=session_id
        )
        user_msg_id = self.repository_manager.chat_messages.create(user_msg)
        user_msg.id = user_msg_id
        return user_msg

    def _extract_generation(self, result: dict[str, Any]) -> str:
        """Pull the final AI response text from graph result messages.

        Uses `AIMessage.text` so we transparently handle both plain-string
        content and the list-of-blocks shape returned by the OpenAI Responses
        API (reasoning blocks are filtered out, text blocks are joined).
        """
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.text:
                return msg.text
        return ""

    def _count_tool_iterations(self, messages: list[BaseMessage]) -> int:
        return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)

    async def _save_ai_message(
        self, session_id: str, content: str, retrieved: list[Paragraph]
    ) -> ChatMessage:
        ai_msg = ChatMessage(
            content=content,
            role=MessageRole.ASSISTANT,
            session_id=session_id,
            retrieved_paragraphs=[p.id for p in retrieved] if retrieved else None,
        )
        ai_msg_id = self.repository_manager.chat_messages.create(ai_msg)
        ai_msg.id = ai_msg_id
        await self._update_session_timestamp(session_id)
        return ai_msg

    async def _update_session_timestamp(self, session_id: str) -> None:
        try:
            self.repository_manager.chat_sessions.update(
                session_id, {"updated_at": datetime.now(UTC)}
            )
        except Exception as e:
            logger.warning(f"Failed to update session timestamp: {e}")

    async def _maybe_regenerate_title(self, session_id: str) -> None:
        messages = await self.get_session_messages(session_id)
        if len(messages) < 2:
            return
        try:
            chat_model = build_chat_model(self.llm_config, temperature_override=0.3)
            chain = create_title_generation_chain(chat_model)

            conversation = "\n\n".join(
                f"{'User' if m.role == MessageRole.USER else 'Assistant'}: {m.content}"
                for m in messages[-20:]
            )
            title = await chain.ainvoke(
                {"conversation": conversation},
                config={
                    "tags": ["title_generation"],
                    "metadata": {"session_id": session_id},
                },
            )
            self.repository_manager.chat_sessions.update(
                session_id,
                {"title": title.strip()[:100], "updated_at": datetime.now(UTC)},
            )
            logger.info(f"Title for {session_id}: '{title.strip()[:100]}'")
        except Exception as e:
            logger.warning(f"Title generation failed for {session_id}: {e}")
