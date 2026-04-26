"""State schema for the RAG agent graph."""

from typing import Annotated

from langgraph.graph import MessagesState

from history_book.data_models.entities import Paragraph


def add_paragraphs(existing: list[Paragraph], new: list[Paragraph]) -> list[Paragraph]:
    """Accumulate retrieved paragraphs, deduplicating by book/chapter/paragraph position."""
    if not existing:
        return new
    if not new:
        return existing

    seen = {(p.book_index, p.chapter_index, p.paragraph_index) for p in existing}
    result = list(existing)
    for para in new:
        key = (para.book_index, para.chapter_index, para.paragraph_index)
        if key not in seen:
            result.append(para)
            seen.add(key)
    return result


class AgentState(MessagesState):
    """
    Graph state for the RAG agent.

    Extends MessagesState (provides messages: Annotated[list[BaseMessage], add_messages]).
    retrieved_paragraphs accumulates across tool calls via add_paragraphs reducer.
    All other agent data (session_id, question, generation) lives in Runtime context
    or is derived from the messages list — not stored as redundant state fields.
    """

    retrieved_paragraphs: Annotated[list[Paragraph], add_paragraphs]
