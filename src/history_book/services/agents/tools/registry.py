"""Tool registry and name-based resolution for the RAG agent."""

from langchain_core.tools import BaseTool

from history_book.services.agents.tools.book_search import search_book
from history_book.services.agents.tools.entity_detail import get_entity_detail
from history_book.services.agents.tools.entity_neighborhood import (
    get_entity_neighborhood,
)
from history_book.services.agents.tools.get_paragraphs import get_paragraphs
from history_book.services.agents.tools.relationships_by_period import (
    query_relationships_by_period,
)
from history_book.services.agents.tools.search_entities import search_entities

TOOL_REGISTRY: dict[str, BaseTool] = {
    "search_book": search_book,
    "search_entities": search_entities,
    "get_entity_detail": get_entity_detail,
    "get_entity_neighborhood": get_entity_neighborhood,
    "get_paragraphs": get_paragraphs,
    "query_relationships_by_period": query_relationships_by_period,
}

# Backwards-compatible default — full toolset.
TOOLS: list[BaseTool] = list(TOOL_REGISTRY.values())


def resolve_tools(enabled: list[str] | None) -> list[BaseTool]:
    """Resolve a list of tool names against TOOL_REGISTRY.

    `None` means all tools enabled. Unknown names raise ValueError.
    """
    if enabled is None:
        return TOOLS
    unknown = [n for n in enabled if n not in TOOL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown tool name(s): {unknown}. Available: {sorted(TOOL_REGISTRY)}"
        )
    return [TOOL_REGISTRY[n] for n in enabled]
