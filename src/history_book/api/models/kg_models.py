"""Pydantic models for the Knowledge Graph API."""

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


class EntityDetail(BaseModel):
    id: str
    name: str
    entity_type: str
    aliases: list[str]
    description: str
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
