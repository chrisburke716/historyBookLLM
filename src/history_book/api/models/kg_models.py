"""Pydantic models for the Knowledge Graph API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    name: str
    entity_type: str
    occurrence_count: int
    aliases: list[str] = []


class GraphLink(BaseModel):
    source: str  # source_entity_id UUID
    target: str  # target_entity_id UUID
    relation_type: str
    description: str = ""


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphLink]
    graph_name: str
    node_count: int
    edge_count: int


class RelationshipSummary(BaseModel):
    relationship_id: str
    relation_type: str
    description: str
    direction: Literal["outgoing", "incoming"]
    other_entity_id: str
    other_entity_name: str
    book_index: int
    chapter_index: int


class EntityDetail(BaseModel):
    id: str
    name: str
    entity_type: str
    aliases: list[str]
    descriptions: list[str]
    occurrence_count: int
    relationships: list[RelationshipSummary]


class SearchResult(BaseModel):
    id: str
    name: str
    entity_type: str
    aliases: list[str]
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str


class SearchRequest(BaseModel):
    query: str
    graph_name: str | None = None
    entity_types: list[str] = []
    limit: int = 10


class KGGraphMeta(BaseModel):
    id: str
    name: str
    graph_type: str
    entity_count: int
    relationship_count: int
    book_chapters: list[str]


class GraphListResponse(BaseModel):
    graphs: list[KGGraphMeta]


# ---------------------------------------------------------------------------
# Metric enums
# ---------------------------------------------------------------------------


class NodeSizeMetric(str, Enum):
    OCCURRENCE_COUNT = "occurrence_count"
    DEGREE_CENTRALITY = "degree_centrality"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    PAGERANK = "pagerank"  # params: damping (default 0.85)
    CLOSENESS_CENTRALITY = "closeness_centrality"
    KCORE_NUMBER = "kcore_number"


class NodeColorMetric(str, Enum):
    ENTITY_TYPE = "entity_type"
    COMMUNITY_LOUVAIN = "community_louvain"
    COMMUNITY_GIRVAN_NEWMAN = "community_girvan_newman"
    COMMUNITY_LABEL_PROPAGATION = "community_label_propagation"
    COMMUNITY_SPECTRAL = "community_spectral"  # params: k (default 5)
    LOCAL_CLUSTERING_COEFFICIENT = "local_clustering_coefficient"
    KCORE_NUMBER = "kcore_number"


class NodePairMetric(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    JACCARD_SIMILARITY = "jaccard_similarity"
    ADAMIC_ADAR = "adamic_adar"
    COMMON_NEIGHBOR_COUNT = "common_neighbor_count"
    SHORTEST_PATH_LENGTH = "shortest_path_length"
    RESISTANCE_DISTANCE = "resistance_distance"  # 501 Not Implemented


# ---------------------------------------------------------------------------
# Metric response models
# ---------------------------------------------------------------------------


class GraphMetricsResponse(BaseModel):
    graph_name: str
    density: float
    giant_component_ratio: float
    num_connected_components: int
    avg_shortest_path_length: float | None  # None if main component > 500 nodes
    diameter: int | None  # None if main component > 500 nodes
    global_clustering_coefficient: float
    num_communities: int  # from Louvain
    articulation_point_count: int
    status: Literal["ready", "computing"]


class NodeMetricResponse(BaseModel):
    graph_name: str
    metric: str
    params: dict[str, float] = {}
    values: dict[str, float]  # entity UUID → metric value
    norm_min: float
    norm_max: float
    status: Literal["ready", "computing"]


class CommunityMetricResponse(BaseModel):
    graph_name: str
    metric: str
    params: dict[str, float] = {}
    values: dict[str, int]  # entity UUID → community ID (int)
    num_communities: int
    status: Literal["ready", "computing"]


class NodePairMetricResponse(BaseModel):
    graph_name: str
    focus_entity_id: str
    metric: str
    params: dict[str, float] = {}
    values: dict[str, float]  # entity UUID → value relative to focus
    norm_min: float
    norm_max: float
