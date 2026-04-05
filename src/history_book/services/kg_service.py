"""Service for querying the knowledge graph."""

import logging

import networkx as nx

from history_book.api.models.kg_models import (
    EntityDetail,
    GraphLink,
    GraphNode,
    GraphResponse,
    RelationshipSummary,
    SearchResponse,
    SearchResult,
)
from history_book.data_models.kg_entities import KGEntity, KGRelationship
from history_book.database.config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

logger = logging.getLogger(__name__)


class KGService:
    """Service for reading knowledge graph data."""

    def __init__(self, repo_manager: BookRepositoryManager | None = None):
        if repo_manager is None:
            config = WeaviateConfig.from_environment()
            repo_manager = BookRepositoryManager(config)
        self.repo_manager = repo_manager
        self._nx_cache: dict[str, nx.DiGraph] = {}

    def list_graphs(self):
        """Return all available KGGraph metadata objects."""
        return self.repo_manager.kg_graphs.list_all()

    def get_graph(self, graph_name: str) -> GraphResponse:
        """Return all nodes and links for a named graph."""
        entities = self.repo_manager.kg_entities.find_by_graph(graph_name)
        relationships = self.repo_manager.kg_relationships.find_by_graph(graph_name)
        # Build and cache NX graph while we have the data, avoiding a second
        # DB round-trip when get_subgraph is called subsequently.
        if graph_name not in self._nx_cache:
            self._nx_cache[graph_name] = self._build_nx_graph(entities, relationships)
        return self._build_graph_response(entities, relationships, graph_name)

    def get_subgraph(self, entity_id: str, hops: int, graph_name: str) -> GraphResponse:
        """Return an N-hop subgraph centered on entity_id within graph_name."""
        G = self._get_nx_graph(graph_name)
        if entity_id not in G:
            return GraphResponse(
                nodes=[], links=[], graph_name=graph_name, node_count=0, edge_count=0
            )

        sub = nx.ego_graph(G, entity_id, radius=hops, undirected=True)

        entities = [G.nodes[n]["entity"] for n in sub.nodes if "entity" in G.nodes[n]]
        relationships = [
            G.edges[u, v, k]["relationship"]
            for u, v, k in sub.edges(keys=True)
            if "relationship" in G.edges[u, v, k]
        ]
        return self._build_graph_response(entities, relationships, graph_name)

    def get_entity(self, entity_id: str) -> EntityDetail | None:
        """Return detailed entity info including denormalized relationships."""
        entity = self.repo_manager.kg_entities.get_by_id(entity_id)
        if entity is None:
            return None

        relationships = self.repo_manager.kg_relationships.find_by_entities(
            [entity_id], entity.graph_name
        )

        rel_summaries: list[RelationshipSummary] = []
        for r in relationships:
            if r.source_entity_id == entity_id:
                direction = "outgoing"
                other_id = r.target_entity_id
                other_name = r.target_entity_name
            else:
                direction = "incoming"
                other_id = r.source_entity_id
                other_name = r.source_entity_name

            rel_summaries.append(
                RelationshipSummary(
                    relationship_id=r.id or "",
                    relation_type=r.relation_type,
                    description=r.description,
                    direction=direction,
                    other_entity_id=other_id,
                    other_entity_name=other_name,
                    book_index=r.book_index,
                    chapter_index=r.chapter_index,
                )
            )

        return EntityDetail(
            id=entity.id or entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            aliases=entity.aliases,
            descriptions=entity.descriptions,
            occurrence_count=entity.occurrence_count,
            relationships=rel_summaries,
        )

    def search(
        self,
        query: str,
        graph_name: str | None,
        entity_types: list[str],
        limit: int,
    ) -> SearchResponse:
        """Hybrid search for entities, optionally filtered by graph and type."""
        results = self.repo_manager.kg_entities.search_entities_hybrid(
            query_text=query,
            graph_name=graph_name,
            limit=limit * 3 if entity_types else limit,  # over-fetch when filtering
        )

        search_results: list[SearchResult] = []
        for entity, score in results:
            if entity_types and entity.entity_type not in entity_types:
                continue
            if entity.id is None:
                continue
            search_results.append(
                SearchResult(
                    id=entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases,
                    score=score,
                )
            )
            if len(search_results) >= limit:
                break

        return SearchResponse(results=search_results, query=query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nx_graph(self, graph_name: str) -> nx.DiGraph:
        """Return cached NX graph, building from DB if not yet cached."""
        if graph_name not in self._nx_cache:
            entities = self.repo_manager.kg_entities.find_by_graph(graph_name)
            relationships = self.repo_manager.kg_relationships.find_by_graph(graph_name)
            self._nx_cache[graph_name] = self._build_nx_graph(entities, relationships)
        return self._nx_cache[graph_name]

    def _build_nx_graph(
        self, entities: list[KGEntity], relationships: list[KGRelationship]
    ) -> nx.DiGraph:
        """Build a NetworkX DiGraph from entity and relationship lists."""
        G: nx.DiGraph = nx.MultiDiGraph()
        for e in entities:
            if e.id:
                G.add_node(e.id, entity=e)
        for r in relationships:
            if r.source_entity_id in G and r.target_entity_id in G:
                G.add_edge(r.source_entity_id, r.target_entity_id, relationship=r)
        return G

    def _build_graph_response(
        self,
        entities: list[KGEntity],
        relationships: list[KGRelationship],
        graph_name: str,
    ) -> GraphResponse:
        nodes = [
            GraphNode(
                id=e.id or "",
                name=e.name,
                entity_type=e.entity_type,
                occurrence_count=e.occurrence_count,
                aliases=e.aliases,
            )
            for e in entities
            if e.id
        ]
        links = [
            GraphLink(
                source=r.source_entity_id,
                target=r.target_entity_id,
                relation_type=r.relation_type,
                description=r.description,
            )
            for r in relationships
        ]
        return GraphResponse(
            nodes=nodes,
            links=links,
            graph_name=graph_name,
            node_count=len(nodes),
            edge_count=len(links),
        )
