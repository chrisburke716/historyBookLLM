"""Knowledge Graph API routes."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from history_book.api.models.kg_models import (
    EntityDetail,
    GraphListResponse,
    GraphResponse,
    KGGraphMeta,
    SearchRequest,
    SearchResponse,
)
from history_book.api.routes.dependencies import get_kg_metrics_service, get_kg_service
from history_book.services.kg_service import KGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["kg"])


@router.get("/graphs", response_model=GraphListResponse)
async def list_graphs(
    service: KGService = Depends(get_kg_service),
):
    """List all available knowledge graphs."""
    try:
        graphs = service.list_graphs()
        return GraphListResponse(
            graphs=[
                KGGraphMeta(
                    id=g.id or "",
                    name=g.name,
                    graph_type=g.graph_type,
                    entity_count=g.entity_count,
                    relationship_count=g.relationship_count,
                    book_chapters=g.book_chapters,
                )
                for g in graphs
                if g.id
            ]
        )
    except Exception as e:
        logger.error("Failed to list graphs: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list graphs") from e


@router.get("/graphs/{graph_name}", response_model=GraphResponse)
async def get_graph(
    graph_name: str,
    background_tasks: BackgroundTasks,
    service: KGService = Depends(get_kg_service),
):
    """Get all nodes and links for a named graph.

    As a side effect, triggers background computation of graph-level metrics
    so they are ready by the time the frontend requests them.
    """
    try:
        result = service.get_graph(graph_name)
        metrics_service = get_kg_metrics_service()
        background_tasks.add_task(metrics_service.trigger_graph_metrics, graph_name)
        return result
    except Exception as e:
        logger.error("Failed to get graph %s: %s", graph_name, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve graph") from e


@router.get("/graphs/{graph_name}/subgraph", response_model=GraphResponse)
async def get_subgraph(
    graph_name: str,
    entity_id: str,
    hops: int = 2,
    service: KGService = Depends(get_kg_service),
):
    """Get an N-hop subgraph centered on an entity within a named graph."""
    if hops < 1 or hops > 3:
        raise HTTPException(status_code=400, detail="hops must be between 1 and 3")
    try:
        return service.get_subgraph(entity_id, hops, graph_name)
    except Exception as e:
        logger.error(
            "Failed to get subgraph for %s in %s: %s", entity_id, graph_name, e
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve subgraph"
        ) from e


@router.get("/entities/{entity_id}", response_model=EntityDetail)
async def get_entity(
    entity_id: str,
    service: KGService = Depends(get_kg_service),
):
    """Get detailed entity information including relationships."""
    try:
        detail = service.get_entity(entity_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Entity not found")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity %s: %s", entity_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve entity") from e


@router.post("/search", response_model=SearchResponse)
async def search_entities(
    request: SearchRequest,
    service: KGService = Depends(get_kg_service),
):
    """Search for entities using hybrid (vector + BM25) search."""
    try:
        return service.search(
            query=request.query,
            graph_name=request.graph_name,
            entity_types=request.entity_types,
            limit=request.limit,
        )
    except Exception as e:
        logger.error("Failed to search entities: %s", e)
        raise HTTPException(status_code=500, detail="Search failed") from e
