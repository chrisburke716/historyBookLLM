"""Tool registry and name-based resolution for the RAG agent."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from history_book.services.agents.tools import (
    book_search,
    entity_detail,
    get_paragraphs,
    relationships_by_period,
    search_entities,
)


@dataclass(frozen=True)
class ToolEntry:
    """Bundles a tool with its UI-facing label/summary functions.

    `start_label` and `end_summary` produce short, human-readable strings
    for the streaming chat UI (Perplexity-style step pills). They are
    co-located with each tool's implementation file so that adding a new
    tool means writing all three definitions in one place.
    """

    tool: BaseTool
    start_label: Callable[[dict[str, Any]], str]
    end_summary: Callable[[dict[str, Any]], str | None]


def _no_summary(_update: dict[str, Any]) -> str | None:
    return None


TOOL_REGISTRY: dict[str, ToolEntry] = {
    "search_book": ToolEntry(
        tool=book_search.search_book,
        start_label=book_search.start_label,
        end_summary=book_search.end_summary,
    ),
    "search_entities": ToolEntry(
        tool=search_entities.search_entities,
        start_label=search_entities.start_label,
        end_summary=search_entities.end_summary,
    ),
    "get_entity_detail": ToolEntry(
        tool=entity_detail.get_entity_detail,
        start_label=entity_detail.start_label,
        end_summary=entity_detail.end_summary,
    ),
    "get_paragraphs": ToolEntry(
        tool=get_paragraphs.get_paragraphs,
        start_label=get_paragraphs.start_label,
        end_summary=get_paragraphs.end_summary,
    ),
    "query_relationships_by_period": ToolEntry(
        tool=relationships_by_period.query_relationships_by_period,
        start_label=relationships_by_period.start_label,
        end_summary=relationships_by_period.end_summary,
    ),
}

# Backwards-compatible default — full toolset.
TOOLS: list[BaseTool] = [entry.tool for entry in TOOL_REGISTRY.values()]


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
    return [TOOL_REGISTRY[n].tool for n in enabled]
