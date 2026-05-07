"""Tools for the RAG agent.

The registry is the single source of truth for available tools and their
UI-facing labels. Submodules are imported lazily via the registry, so
tool function names never collide with submodule names in this package.
"""

from history_book.services.agents.tools.registry import (
    TOOL_REGISTRY,
    TOOLS,
    ToolEntry,
    resolve_tools,
)

__all__ = [
    "TOOL_REGISTRY",
    "TOOLS",
    "ToolEntry",
    "resolve_tools",
]
