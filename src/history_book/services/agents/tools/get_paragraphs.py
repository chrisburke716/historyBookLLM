"""Look up specific paragraphs by ID."""

import logging
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from history_book.services.agents.context import AgentContext
from history_book.services.agents.prompts import format_excerpts_for_llm

logger = logging.getLogger(__name__)


def start_label(args: dict[str, Any]) -> str:
    n = len(args.get("paragraph_ids") or [])
    return f"Retrieving {n} source paragraph{'s' if n != 1 else ''}"


def end_summary(update: dict[str, Any]) -> str | None:
    n = len(update.get("retrieved_paragraphs") or [])
    return f"{n} retrieved" if n else None


@tool
def get_paragraphs(
    paragraph_ids: Annotated[
        list[str],
        "List of paragraph UUIDs (from search results, entity source_paragraph_ids, "
        "or relationship paragraph_ids)",
    ],
    runtime: ToolRuntime[AgentContext],
) -> Command:
    """Retrieve specific paragraphs by ID — exact lookup, no search.

    Use to ground claims in source text after `get_entity_detail` returns
    source_paragraph_ids, or to read context around a paragraph already found
    by another tool. Returns the paragraph text with chapter and page metadata
    for citations.
    """
    if not paragraph_ids:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Provide at least one paragraph_id.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    ctx = runtime.context
    paragraphs = []
    missing: list[str] = []
    for pid in paragraph_ids:
        try:
            p = ctx.repository_manager.paragraphs.get_by_id(pid)
        except Exception as e:
            logger.warning(f"get_paragraphs: lookup failed for {pid}: {e}")
            p = None
        if p is None:
            missing.append(pid)
        else:
            paragraphs.append(p)

    if not paragraphs:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"No paragraphs found for the given IDs ({len(missing)} missing).",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    formatted = format_excerpts_for_llm(paragraphs)
    note = (
        f"\n\n(Note: {len(missing)} of {len(paragraph_ids)} IDs were not found.)"
        if missing
        else ""
    )
    logger.info(
        f"get_paragraphs: returned {len(paragraphs)}/{len(paragraph_ids)} paragraphs"
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Retrieved {len(paragraphs)} paragraph(s):\n\n{formatted}{note}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "retrieved_paragraphs": paragraphs,
        }
    )
