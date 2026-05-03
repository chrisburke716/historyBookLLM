"""Tools for the RAG agent."""

from history_book.services.agents.tools.book_search import search_book
from history_book.services.agents.tools.entity_detail import get_entity_detail
from history_book.services.agents.tools.get_paragraphs import get_paragraphs
from history_book.services.agents.tools.registry import (
    TOOL_REGISTRY,
    TOOLS,
    resolve_tools,
)
from history_book.services.agents.tools.relationships_by_period import (
    query_relationships_by_period,
)
from history_book.services.agents.tools.search_entities import search_entities

__all__ = [
    "TOOL_REGISTRY",
    "TOOLS",
    "resolve_tools",
    "search_book",
    "search_entities",
    "get_entity_detail",
    "get_paragraphs",
    "query_relationships_by_period",
]
