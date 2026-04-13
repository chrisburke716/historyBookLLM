"""Service for computing and caching network metrics on knowledge graphs."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

from history_book.api.models.kg_models import (
    CommunityMetricResponse,
    GraphMetricsResponse,
    NodeMetricResponse,
    NodePairMetricResponse,
)
from history_book.services.kg_service import KGService

logger = logging.getLogger(__name__)

# Maximum size for expensive O(V*E) graph metrics (avg path length, diameter)
_MAX_COMPONENT_FOR_PATH_METRICS = 500

# Percentile bounds used for node-pair metric color scale normalization.
# Clipping at these percentiles prevents a single distant outlier node from
# collapsing all other nodes into a narrow band of the color scale.
_NORM_PERCENTILE_LOW = 2
_NORM_PERCENTILE_HIGH = 98


def _to_simple_graph(G: nx.DiGraph) -> nx.Graph:
    """Convert a MultiDiGraph to a simple undirected Graph.

    Collapses parallel edges and removes self-loops, both of which cause
    NetworkXNotImplemented errors in algorithms like core_number, clustering,
    and articulation_points.
    """
    U = nx.Graph(G)
    U.remove_edges_from(nx.selfloop_edges(U))
    return U


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _normalize(values: dict[str, float]) -> tuple[float, float]:
    """Return (min, max) across all values."""
    if not values:
        return 0.0, 0.0
    v = list(values.values())
    return min(v), max(v)


def _community_list_to_dict(communities: list[frozenset]) -> dict[str, int]:
    """Convert list-of-frozensets to {node_id: community_id} dict."""
    result: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            result[node] = cid
    return result


class KGMetricsService:
    """Computes and caches network metrics for knowledge graphs.

    All slow metrics (betweenness centrality, community detection, etc.) are
    computed in a background thread pool. Callers receive a 202 placeholder
    immediately and should poll until the cache entry is ready.

    Cache key: (graph_name, metric_name, *sorted_param_items)
    """

    def __init__(self, kg_service: KGService):
        self._kg = kg_service
        self._cache: dict[tuple, Any] = {}
        self._computing: set[tuple] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="kg-metrics"
        )

    # ------------------------------------------------------------------
    # Graph-level metrics
    # ------------------------------------------------------------------

    def trigger_graph_metrics(self, graph_name: str) -> None:
        """Fire-and-forget trigger called as a FastAPI BackgroundTask.

        Starts background computation if not already cached or computing.
        """
        key = (graph_name, "_graph_metrics")
        if key in self._cache or key in self._computing:
            return
        self._computing.add(key)
        self._executor.submit(self._compute_graph_metrics, graph_name, key)

    async def get_graph_metrics(
        self, graph_name: str
    ) -> tuple[GraphMetricsResponse, int]:
        """Return (response, http_status). Status 202 while still computing."""
        key = (graph_name, "_graph_metrics")
        if key in self._cache:
            return self._cache[key], 200
        if key not in self._computing:
            self._computing.add(key)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                self._executor, self._compute_graph_metrics, graph_name, key
            )
        return self._computing_graph_placeholder(graph_name), 202

    def _compute_graph_metrics(self, graph_name: str, key: tuple) -> None:
        try:
            G = self._kg.get_nx_graph(graph_name)
            U = _to_simple_graph(G)

            density = nx.density(G)
            n = len(G.nodes)

            weak_components = list(nx.weakly_connected_components(G))
            max_comp = max(weak_components, key=len) if weak_components else set()
            giant_ratio = len(max_comp) / max(n, 1)
            num_components = len(weak_components)

            if len(max_comp) <= _MAX_COMPONENT_FOR_PATH_METRICS and len(max_comp) > 1:
                sub_u = U.subgraph(max_comp)
                if nx.is_connected(sub_u):
                    avg_path: float | None = nx.average_shortest_path_length(sub_u)
                    diameter: int | None = nx.diameter(sub_u)
                else:
                    avg_path = None
                    diameter = None
            else:
                avg_path = None
                diameter = None

            global_cc = nx.average_clustering(U)
            communities = list(nx.community.louvain_communities(U, seed=42))
            artic_pts = list(nx.articulation_points(U))

            result = GraphMetricsResponse(
                graph_name=graph_name,
                density=density,
                giant_component_ratio=giant_ratio,
                num_connected_components=num_components,
                avg_shortest_path_length=avg_path,
                diameter=diameter,
                global_clustering_coefficient=global_cc,
                num_communities=len(communities),
                articulation_point_count=len(artic_pts),
                status="ready",
            )
            self._cache[key] = result
        except Exception:
            logger.exception("Failed to compute graph metrics for %s", graph_name)
        finally:
            self._computing.discard(key)

    @staticmethod
    def _computing_graph_placeholder(graph_name: str) -> GraphMetricsResponse:
        return GraphMetricsResponse(
            graph_name=graph_name,
            density=0.0,
            giant_component_ratio=0.0,
            num_connected_components=0,
            avg_shortest_path_length=None,
            diameter=None,
            global_clustering_coefficient=0.0,
            num_communities=0,
            articulation_point_count=0,
            status="computing",
        )

    # ------------------------------------------------------------------
    # Node-level metrics
    # ------------------------------------------------------------------

    async def get_node_metric(
        self, graph_name: str, metric: str, params: dict[str, float]
    ) -> tuple[NodeMetricResponse | CommunityMetricResponse, int]:
        """Return (response, http_status). Status 202 while still computing."""
        key = (graph_name, metric, *sorted(params.items()))
        if key in self._cache:
            return self._cache[key], 200
        if key not in self._computing:
            self._computing.add(key)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                self._executor,
                self._compute_node_metric,
                graph_name,
                metric,
                params,
                key,
            )
        return self._computing_node_placeholder(graph_name, metric, params), 202

    def _compute_node_metric(
        self, graph_name: str, metric: str, params: dict[str, float], key: tuple
    ) -> None:
        try:
            G = self._kg.get_nx_graph(graph_name)
            U = _to_simple_graph(G)
            result = self._dispatch_node_metric(G, U, graph_name, metric, params)
            self._cache[key] = result
        except Exception:
            logger.exception(
                "Failed to compute node metric %s for %s", metric, graph_name
            )
        finally:
            self._computing.discard(key)

    def _dispatch_node_metric(
        self,
        G: nx.DiGraph,
        U: nx.Graph,
        graph_name: str,
        metric: str,
        params: dict[str, float],
    ) -> NodeMetricResponse | CommunityMetricResponse:
        # --- Community metrics (return CommunityMetricResponse) ---
        if metric == "community_louvain":
            communities = list(nx.community.louvain_communities(U, seed=42))
            values = _community_list_to_dict(communities)
            return CommunityMetricResponse(
                graph_name=graph_name,
                metric=metric,
                params=params,
                values=values,
                num_communities=len(communities),
                status="ready",
            )

        if metric == "community_girvan_newman":
            comp = nx.community.girvan_newman(U)
            communities = list(next(comp))
            values = _community_list_to_dict(communities)
            return CommunityMetricResponse(
                graph_name=graph_name,
                metric=metric,
                params=params,
                values=values,
                num_communities=len(communities),
                status="ready",
            )

        if metric == "community_label_propagation":
            communities = list(nx.community.label_propagation_communities(U))
            values = _community_list_to_dict(communities)
            return CommunityMetricResponse(
                graph_name=graph_name,
                metric=metric,
                params=params,
                values=values,
                num_communities=len(communities),
                status="ready",
            )

        if metric == "community_spectral":
            k = int(params.get("k", 5))
            node_list = list(U.nodes)
            adj = nx.to_numpy_array(U, nodelist=node_list)
            sc = SpectralClustering(
                n_clusters=k, affinity="precomputed", random_state=42, n_init=10
            )
            labels = sc.fit_predict(adj)
            values = {node_list[i]: int(labels[i]) for i in range(len(node_list))}
            return CommunityMetricResponse(
                graph_name=graph_name,
                metric=metric,
                params=params,
                values=values,
                num_communities=k,
                status="ready",
            )

        # --- Continuous node metrics (return NodeMetricResponse) ---
        if metric == "degree_centrality":
            raw = nx.degree_centrality(G)
        elif metric == "betweenness_centrality":
            raw = nx.betweenness_centrality(G)
        elif metric == "pagerank":
            damping = params.get("damping", 0.85)
            raw = nx.pagerank(G, alpha=damping)
        elif metric == "closeness_centrality":
            raw = nx.closeness_centrality(G)
        elif metric == "kcore_number":
            raw = {n: float(v) for n, v in nx.core_number(U).items()}
        elif metric == "local_clustering_coefficient":
            raw = nx.clustering(U)
        else:
            msg = f"Unknown node metric: {metric}"
            raise ValueError(msg)

        values = {k: float(v) for k, v in raw.items()}
        norm_min, norm_max = _normalize(values)
        return NodeMetricResponse(
            graph_name=graph_name,
            metric=metric,
            params=params,
            values=values,
            norm_min=norm_min,
            norm_max=norm_max,
            status="ready",
        )

    @staticmethod
    def _computing_node_placeholder(
        graph_name: str, metric: str, params: dict[str, float]
    ) -> NodeMetricResponse:
        return NodeMetricResponse(
            graph_name=graph_name,
            metric=metric,
            params=params,
            values={},
            norm_min=0.0,
            norm_max=0.0,
            status="computing",
        )

    # ------------------------------------------------------------------
    # Node-pair metrics (always synchronous)
    # ------------------------------------------------------------------

    def get_node_pair_metric(
        self, graph_name: str, focus_id: str, metric: str, params: dict[str, float]
    ) -> NodePairMetricResponse:
        """Compute node-pair metric relative to focus_id. Always returns immediately."""
        G = self._kg.get_nx_graph(graph_name)
        U = nx.Graph(G)  # collapse MultiDiGraph → simple Graph

        if focus_id not in G:
            return NodePairMetricResponse(
                graph_name=graph_name,
                focus_entity_id=focus_id,
                metric=metric,
                params=params,
                values={},
                norm_min=0.0,
                norm_max=0.0,
            )

        node_ids = list(G.nodes)
        other_ids = [n for n in node_ids if n != focus_id]
        pairs = [(focus_id, n) for n in other_ids]

        values: dict[str, float] = {}

        if metric == "jaccard_similarity":
            for _u, v, s in nx.jaccard_coefficient(U, pairs):
                values[v] = s

        elif metric == "adamic_adar":
            for _u, v, s in nx.adamic_adar_index(U, pairs):
                values[v] = s

        elif metric == "common_neighbor_count":
            focus_neighbors = set(U.neighbors(focus_id))
            for n in other_ids:
                values[n] = float(len(focus_neighbors & set(U.neighbors(n))))

        elif metric == "shortest_path_length":
            lengths = dict(nx.single_source_shortest_path_length(U, focus_id))
            for n in other_ids:
                values[n] = float(lengths.get(n, -1))

        elif metric == "cosine_similarity":
            vectors = self._kg.get_entity_vectors(graph_name)
            focus_vec = vectors.get(focus_id)
            if focus_vec is not None:
                for n in other_ids:
                    if n in vectors:
                        values[n] = _cosine_sim(focus_vec, vectors[n])

        elif metric == "resistance_distance":
            # Effective resistance via Laplacian pseudoinverse.
            # R(i,j) = L⁺[i,i] + L⁺[j,j] - 2·L⁺[i,j]  (O(n³) — slow on large graphs)
            # Nodes in different components get -1 (unreachable).
            node_list = list(U.nodes)
            if focus_id not in U:
                pass  # values stays empty
            else:
                focus_idx = node_list.index(focus_id)
                focus_component = nx.node_connected_component(U, focus_id)
                L = nx.laplacian_matrix(U, nodelist=node_list).toarray().astype(float)
                L_pinv = np.linalg.pinv(L)
                for i, n in enumerate(node_list):
                    if n == focus_id:
                        continue
                    if n not in focus_component:
                        values[n] = -1.0
                    else:
                        values[n] = float(
                            L_pinv[focus_idx, focus_idx]
                            + L_pinv[i, i]
                            - 2 * L_pinv[focus_idx, i]
                        )

        else:
            msg = f"Unknown node-pair metric: {metric}"
            raise ValueError(msg)

        # For distance metrics, self-distance is 0 (not max).
        # For similarity metrics, self-similarity is the maximum value.
        _distance_metrics = {"resistance_distance", "shortest_path_length"}
        if metric in _distance_metrics:
            values[focus_id] = 0.0
        else:
            values[focus_id] = max(values.values()) if values else 0.0

        # Exclude sentinel -1 values (unreachable nodes) from normalization bounds.
        # Use percentile clipping so a single outlier doesn't collapse the color scale.
        finite_vals = [v for v in values.values() if v >= 0]
        if finite_vals:
            norm_min = float(np.percentile(finite_vals, _NORM_PERCENTILE_LOW))
            norm_max = float(np.percentile(finite_vals, _NORM_PERCENTILE_HIGH))
        else:
            norm_min, norm_max = 0.0, 0.0

        return NodePairMetricResponse(
            graph_name=graph_name,
            focus_entity_id=focus_id,
            metric=metric,
            params=params,
            values=values,
            norm_min=norm_min,
            norm_max=norm_max,
        )
