"""Get a knowledge graph entity's descriptions, source paragraphs, and 1-hop relationships."""

import logging
import re
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from history_book.services.agents.context import AgentContext
from history_book.services.agents.tools._format import format_relationship_summary

logger = logging.getLogger(__name__)

# Matches the first line of the tool's normal output, e.g.
#   "Entity: Charlemagne [person] (id=…)"
_ENTITY_NAME_RE = re.compile(r"^Entity:\s+([^\[]+?)\s+\[")


def start_label(_args: dict[str, Any]) -> str:
    return "Looking up entity details"


def end_summary(update: dict[str, Any]) -> str | None:
    msgs = update.get("messages") or []
    if not msgs:
        return None
    text = getattr(msgs[0], "content", "") or ""
    m = _ENTITY_NAME_RE.match(text)
    return m.group(1).strip() if m else None


@tool
def get_entity_detail(
    entity_id: Annotated[str, "Entity UUID (from search_entities or another KG tool)"],
    runtime: ToolRuntime[AgentContext],
    limit: Annotated[
        int,
        "Max number of relationships to include; sorted by other-entity prominence",
    ] = 25,
) -> Command:
    """Get an entity's descriptions, source paragraph IDs, and 1-hop relationships.

    Use after `search_entities` to expand a specific entity into its full record:
    accumulated descriptions across the book, occurrence count, source paragraph IDs,
    and the relationships connecting it to other entities (with relation type,
    description, direction, and chapter location). For highly-connected entities,
    relationships are truncated by `limit` and sorted by prominence — the message
    notes the truncation so you can narrow further.

    Pass `source_paragraph_ids` to `get_paragraphs` to retrieve the original text.
    """
    ctx = runtime.context
    try:
        detail = ctx.kg_service.get_entity(entity_id)
    except Exception as e:
        logger.error(f"get_entity_detail failed for {entity_id}: {e}", exc_info=True)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="An error occurred while fetching entity detail.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if detail is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"No entity found with id={entity_id}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    sorted_rels = sorted(
        detail.relationships,
        key=lambda r: r.other_entity_occurrence_count,
        reverse=True,
    )
    total = len(sorted_rels)
    shown = sorted_rels[:limit]

    parts = [
        f"Entity: {detail.name} [{detail.entity_type}] (id={detail.id})",
        f"Aliases: {', '.join(detail.aliases) if detail.aliases else '(none)'}",
        f"Occurrences: {detail.occurrence_count}",
    ]
    if detail.descriptions:
        parts.append("Descriptions:")
        for d in detail.descriptions:
            parts.append(f"- {d}")
    else:
        parts.append("Descriptions: (none)")

    if detail.source_paragraph_ids:
        parts.append(f"source_paragraph_ids ({len(detail.source_paragraph_ids)}):")
        for pid in detail.source_paragraph_ids:
            parts.append(f"- {pid}")

    if total == 0:
        parts.append("Relationships: (none)")
    else:
        header = f"Relationships ({len(shown)} shown"
        if total > len(shown):
            header += (
                f" of {total}, sorted by prominence — increase `limit` to see more"
            )
        header += "):"
        parts.append(header)
        parts.append("")
        for r in shown:
            parts.append(format_relationship_summary(r))
            parts.append("")

    logger.info(f"get_entity_detail: {detail.name} — {total} rels ({len(shown)} shown)")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(parts),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )
