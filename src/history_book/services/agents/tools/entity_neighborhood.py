"""Explore the multi-hop neighborhood around a knowledge graph entity."""

import logging
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from history_book.services.agents.context import AgentContext
from history_book.services.agents.tools._format import format_relationship

logger = logging.getLogger(__name__)


@tool
def get_entity_neighborhood(
    entity_id: Annotated[str, "Central entity UUID"],
    runtime: ToolRuntime[AgentContext],
    hops: Annotated[int, "Number of hops out from the central entity (1, 2, or 3)"] = 2,
    max_nodes: Annotated[int, "Cap on nodes returned"] = 30,
) -> Command:
    """Explore the N-hop neighborhood (people, events, places) around an entity.

    Use to answer questions about clusters or networks (e.g., "who and what
    surrounded Augustus?"). Returns the surrounding entities and the relationships
    connecting them. Capped at `max_nodes` — increase only if necessary, since
    neighborhoods can grow large quickly.
    """
    if hops < 1 or hops > 3:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="hops must be 1, 2, or 3.",
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
        graph_response = ctx.kg_service.get_subgraph(
            entity_id=entity_id, hops=hops, graph_name=ctx.volume_graph_name
        )
    except Exception as e:
        logger.error(
            f"get_entity_neighborhood failed for {entity_id}: {e}", exc_info=True
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="An error occurred while computing the neighborhood.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if not graph_response.nodes:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Entity {entity_id} not found in the graph or has no "
                            f"neighbors within {hops} hops."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    G = ctx.kg_service.get_nx_graph(ctx.volume_graph_name)
    sorted_nodes = sorted(
        graph_response.nodes,
        key=lambda n: 0 if n.id == entity_id else -n.occurrence_count,
    )
    kept_node_ids = {n.id for n in sorted_nodes[:max_nodes]}
    kept_nodes = [n for n in sorted_nodes if n.id in kept_node_ids]
    truncated = len(sorted_nodes) > max_nodes

    parts = [
        f"Neighborhood of entity {entity_id} ({hops}-hop, "
        f"{len(kept_nodes)} of {len(graph_response.nodes)} nodes shown"
        + (", truncated by max_nodes" if truncated else "")
        + "):"
    ]
    parts.append("Entities:")
    for n in kept_nodes:
        marker = " (center)" if n.id == entity_id else ""
        parts.append(f"- id={n.id} | {n.name} [{n.entity_type}]{marker}")

    parts.append("Relationships:")
    rel_lines: list[str] = []
    for u, v, k in G.edges(keys=True):
        if u in kept_node_ids and v in kept_node_ids:
            rel = G.edges[u, v, k].get("relationship")
            if rel is not None:
                rel_lines.append(format_relationship(rel))
    if rel_lines:
        parts.extend(rel_lines)
    else:
        parts.append("(no relationships among kept nodes)")

    logger.info(
        f"get_entity_neighborhood: {len(kept_nodes)}/{len(graph_response.nodes)} "
        f"nodes, {len(rel_lines)} edges"
    )
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
