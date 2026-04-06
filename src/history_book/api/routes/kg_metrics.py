"""Knowledge Graph metrics API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from history_book.api.models.kg_models import (
    GraphMetricsResponse,
    NodePairMetricResponse,
)
from history_book.api.routes.dependencies import get_kg_metrics_service
from history_book.services.kg_metrics_service import KGMetricsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg/metrics", tags=["kg-metrics"])


@router.get("/graph", response_model=GraphMetricsResponse)
async def get_graph_metrics(
    graph_name: str,
    service: KGMetricsService = Depends(get_kg_metrics_service),
) -> JSONResponse:
    """Get graph-level network metrics.

    Returns 200 with data when ready, 202 while still computing.
    The first call for a given graph_name triggers background computation.
    """
    try:
        result, status_code = await service.get_graph_metrics(graph_name)
        return JSONResponse(content=result.model_dump(), status_code=status_code)
    except Exception as e:
        logger.error("Failed to get graph metrics for %s: %s", graph_name, e)
        raise HTTPException(
            status_code=500, detail="Failed to compute graph metrics"
        ) from e


@router.get("/node")
async def get_node_metric(
    graph_name: str,
    metric: str,
    damping: float = 0.85,
    k: int = 5,
    service: KGMetricsService = Depends(get_kg_metrics_service),
) -> JSONResponse:
    """Get node-level metric values for all nodes in a graph.

    Returns NodeMetricResponse for continuous metrics (degree_centrality,
    betweenness_centrality, pagerank, closeness_centrality, kcore_number,
    local_clustering_coefficient) and CommunityMetricResponse for community
    metrics (community_louvain, community_girvan_newman,
    community_label_propagation, community_spectral).

    Returns 200 with data when ready, 202 while still computing.
    Params: damping (PageRank only), k (community_spectral only).
    """
    params: dict[str, float] = {}
    if metric == "pagerank":
        params["damping"] = damping
    elif metric == "community_spectral":
        params["k"] = float(k)

    try:
        result, status_code = await service.get_node_metric(graph_name, metric, params)
        return JSONResponse(content=result.model_dump(), status_code=status_code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to get node metric %s for %s: %s", metric, graph_name, e)
        raise HTTPException(
            status_code=500, detail="Failed to compute node metric"
        ) from e


@router.get("/node-pair", response_model=NodePairMetricResponse)
async def get_node_pair_metric(
    graph_name: str,
    focus_entity_id: str,
    metric: str,
    service: KGMetricsService = Depends(get_kg_metrics_service),
) -> NodePairMetricResponse:
    """Get node-pair metric values relative to a focus entity (always 200).

    Supported metrics: cosine_similarity, jaccard_similarity, adamic_adar,
    common_neighbor_count, shortest_path_length.
    resistance_distance returns 501.
    """
    if metric == "resistance_distance":
        raise HTTPException(
            status_code=501,
            detail="Resistance distance is not implemented (computationally infeasible on large graphs)",
        )
    try:
        return service.get_node_pair_metric(graph_name, focus_entity_id, metric, {})
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Failed to get node-pair metric %s for focus %s in %s: %s",
            metric,
            focus_entity_id,
            graph_name,
            e,
        )
        raise HTTPException(
            status_code=500, detail="Failed to compute node-pair metric"
        ) from e
