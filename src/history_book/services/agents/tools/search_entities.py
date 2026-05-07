"""Search the knowledge graph for entities by name/description."""

import logging
import re
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from history_book.services.agents.context import AgentContext

logger = logging.getLogger(__name__)

_FOUND_RE = re.compile(r"^Found\s+(\d+)\s+entities", re.MULTILINE)


def start_label(args: dict[str, Any]) -> str:
    return f'Searching the knowledge graph for "{args.get("query", "")}"'


def end_summary(update: dict[str, Any]) -> str | None:
    msgs = update.get("messages") or []
    if not msgs:
        return None
    text = getattr(msgs[0], "content", "") or ""
    m = _FOUND_RE.match(text)
    return f"{m.group(1)} entities" if m else None


@tool
def search_entities(
    query: Annotated[str, "Search query — entity name, alias, or short description"],
    runtime: ToolRuntime[AgentContext],
    entity_types: Annotated[
        list[str] | None,
        "Optional filter: any of 'person', 'polity', 'place', 'event', 'concept'",
    ] = None,
    limit: Annotated[int, "Max results to return"] = 10,
) -> Command:
    """Find entities (people, places, polities, events, concepts) in the knowledge graph.

    Use when the user names or alludes to a specific historical figure, polity, place,
    event, or concept — this is more targeted than text search for entity questions.
    Returns a compact list of entity IDs with names, types, aliases, and match scores.
    Pass an entity ID to `get_entity_detail` to see descriptions and relationships.
    """
    if not query or len(query.strip()) < 2:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Please provide a more specific entity query.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    ctx = runtime.context
    try:
        response = ctx.kg_service.search(
            query=query,
            graph_name=ctx.volume_graph_name,
            entity_types=entity_types or [],
            limit=limit,
        )
    except Exception as e:
        logger.error(f"search_entities failed: {e}", exc_info=True)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="An error occurred during entity search.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if not response.results:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"No entities found for '{query}'. Try a different name or alias.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    lines = [f"Found {len(response.results)} entities for '{query}':"]
    for r in response.results:
        aliases = f" (aliases: {', '.join(r.aliases)})" if r.aliases else ""
        lines.append(
            f"- id={r.id} | {r.name} [{r.entity_type}]{aliases} (score={r.score:.2f})"
        )
    logger.info(f"search_entities: {len(response.results)} hits for '{query[:60]}'")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n".join(lines),
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )
