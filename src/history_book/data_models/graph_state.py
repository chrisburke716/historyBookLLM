"""State schema for LangGraph-based RAG agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from history_book.data_models.entities import Paragraph


class AgentState(TypedDict):
    """
    State schema for the RAG agent graph.

    The state is passed between nodes and updated at each step.
    Uses Annotated with add_messages reducer for automatic message list management.

    Attributes:
        messages: Chat history with automatic list management (reducer pattern).
            The add_messages reducer automatically appends new messages rather than
            replacing the entire list.
        question: Current user query being processed.
        retrieved_paragraphs: Context documents retrieved from vector database.
        generation: Generated response from the LLM.
        session_id: Session identifier (maps to Weaviate session_id and LangGraph thread_id).
        metadata: Execution metadata for debugging/tracing (e.g., execution time,
            nodes executed, etc.).

    Future fields (not used initially, uncomment when needed):
        tool_calls: LLM's tool call requests for future tool calling feature.
        tool_results: Results from tool executions.
        reasoning_steps: For multi-step reasoning tracking (planning, reflection, etc.).
    """

    # Messages with automatic list management (reducer pattern)
    messages: Annotated[list[BaseMessage], add_messages]

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
