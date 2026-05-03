"""Query knowledge graph relationships by historical time period."""

import logging
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from history_book.data_models.kg_entities import KGRelationship
from history_book.services.agents.context import AgentContext
from history_book.services.agents.tools._format import format_relationship

logger = logging.getLogger(__name__)


@tool
def query_relationships_by_period(
    start_year: Annotated[
        int, "Start of the period (inclusive). Use negative for BCE (e.g., -100)."
    ],
    end_year: Annotated[int, "End of the period (inclusive)."],
    runtime: ToolRuntime[AgentContext],
    entity_types: Annotated[
        list[str] | None,
        "Optional filter — keep relationships where source OR target entity has one of these types",
    ] = None,
    relation_types: Annotated[
        list[str] | None,
        "Optional filter — only include these relation types (e.g., 'ruled', 'conquered')",
    ] = None,
    limit: Annotated[int, "Max relationships to return"] = 25,
) -> Command:
    """Find relationships dated within a historical period (era-scoped questions).

    Use for "what happened between X and Y", "key alliances of the 12th century",
    or similar time-window queries that are hard to retrieve via text search.
    Filters relationships whose [start_year, end_year] overlap the requested
    period. A relationship with at least one year defined (start_year or
    end_year) is eligible — relationships with no temporal data are excluded.
    """
    if start_year > end_year:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="start_year must be <= end_year.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    ctx = runtime.context
    if ctx.volume_graph_name is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No knowledge graph is configured.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    try:
        # NOTE: this fetches every relationship in the volume graph and filters in
        # Python. Not ideal — a Weaviate-side year-range filter would be cheaper,
        # but the current schema/index doesn't support it directly. Revisit if this
        # becomes a hot path or the volume graph grows much larger.
        all_rels = ctx.repository_manager.kg_relationships.find_by_graph(
            ctx.volume_graph_name
        )
    except Exception as e:
        logger.error(f"query_relationships_by_period failed: {e}", exc_info=True)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="An error occurred while querying relationships.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    type_filter = set(entity_types) if entity_types else None
    rel_filter = set(relation_types) if relation_types else None

    matched: list[KGRelationship] = []
    entity_type_cache: dict[str, str] = {}
    for r in all_rels:
        if not _overlaps(r.start_year, r.end_year, start_year, end_year):
            continue
        if rel_filter and r.relation_type not in rel_filter:
            continue
        if type_filter and not _entity_types_match(
            r, type_filter, ctx, entity_type_cache
        ):
            continue
        matched.append(r)

    matched.sort(
        key=lambda r: (r.start_year if r.start_year is not None else r.end_year or 0)
    )
    total = len(matched)
    shown = matched[:limit]

    if not shown:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"No relationships with temporal data found between "
                            f"{start_year} and {end_year}."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    header = f"Relationships in [{start_year}, {end_year}] ({len(shown)} of {total}"
    if total > len(shown):
        header += ", narrow with entity_types/relation_types"
    header += "):"
    lines = [header, ""]
    for r in shown:
        lines.append(format_relationship(r))
        lines.append("")

    logger.info(
        f"query_relationships_by_period [{start_year},{end_year}]: "
        f"{len(shown)}/{total} shown"
    )
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


def _overlaps(
    rel_start: int | None, rel_end: int | None, q_start: int, q_end: int
) -> bool:
    """Return True if [rel_start, rel_end] overlaps [q_start, q_end].

    Treats unknown bounds permissively: if start_year is None, use end_year as the
    point; if end_year is None, use start_year. If both are None, no overlap.
    """
    if rel_start is None and rel_end is None:
        return False
    s = rel_start if rel_start is not None else rel_end
    e = rel_end if rel_end is not None else rel_start
    return s <= q_end and e >= q_start


def _entity_types_match(
    r: KGRelationship,
    type_filter: set[str],
    ctx: AgentContext,
    cache: dict[str, str],
) -> bool:
    """Return True if either source or target entity is in type_filter."""
    for eid in (r.source_entity_id, r.target_entity_id):
        if eid not in cache:
            try:
                ent = ctx.repository_manager.kg_entities.get_by_id(eid)
                cache[eid] = ent.entity_type if ent else ""
            except Exception:
                cache[eid] = ""
        if cache[eid] in type_filter:
            return True
    return False
