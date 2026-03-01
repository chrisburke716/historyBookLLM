"""Knowledge graph repository implementations."""

import logging
from typing import TYPE_CHECKING

from weaviate.classes.query import Filter

from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories.weaviate_repository import WeaviateRepository

if TYPE_CHECKING:
    from history_book.data_models.kg_entities import KGEntity, KGGraph, KGRelationship

logger = logging.getLogger(__name__)


class KGEntityRepository(WeaviateRepository["KGEntity"]):
    """Repository for knowledge graph entity storage and retrieval."""

    def __init__(self, config: WeaviateConfig):
        from history_book.data_models.kg_entities import KGEntity  # noqa: PLC0415

        super().__init__(
            config=config,
            collection_name="KGEntities",
            entity_class=KGEntity,
        )

    def find_by_graph(self, graph_name: str) -> list["KGEntity"]:
        """Find all entities belonging to a specific graph."""
        return self.find_by_criteria({"graph_name": graph_name})

    def find_by_paragraph(self, paragraph_id: str, graph_name: str) -> list["KGEntity"]:
        """Find entities that appear in a specific paragraph.

        Uses ContainsAny filter on the source_paragraph_ids TEXT_ARRAY field.
        """
        try:
            results = self.collection.query.fetch_objects(
                filters=(
                    Filter.by_property("source_paragraph_ids").contains_any(
                        [paragraph_id]
                    )
                    & Filter.by_property("graph_name").equal(graph_name)
                ),
                limit=1000,
            )
            return [
                self._weaviate_object_to_entity(obj)
                for obj in results.objects
                if self._weaviate_object_to_entity(obj) is not None
            ]
        except Exception as e:
            logger.error("Failed to find entities by paragraph: %s", e)
            return []

    def search_entities(
        self,
        query_text: str,
        graph_name: str | None = None,
        limit: int = 100,
        threshold: float | None = 0.65,
    ) -> list[tuple["KGEntity", float]]:
        """Search entities by semantic similarity on entity properties.

        Args:
            query_text: Text to search for.
            graph_name: Optional graph to scope the search to.
            limit: Maximum number of results.
            threshold: Minimum similarity threshold (cosine similarity cutoff).
        """
        where_filter = None
        if graph_name:
            where_filter = {"graph_name": graph_name}
        return self.similarity_search_by_text(
            query_text=query_text,
            limit=limit,
            threshold=threshold,
            where_filter=where_filter,
        )

    def delete_by_graph(self, graph_name: str) -> int:
        """Delete all entities for a graph. Returns count deleted."""
        entities = self.find_by_graph(graph_name)
        count = 0
        for entity in entities:
            if entity.id and self.delete(entity.id):
                count += 1
        return count


class KGRelationshipRepository(WeaviateRepository["KGRelationship"]):
    """Repository for knowledge graph relationship storage and retrieval."""

    def __init__(self, config: WeaviateConfig):
        from history_book.data_models.kg_entities import (  # noqa: PLC0415
            KGRelationship,
        )

        super().__init__(
            config=config,
            collection_name="KGRelationships",
            entity_class=KGRelationship,
        )

    def find_by_graph(self, graph_name: str) -> list["KGRelationship"]:
        """Find all relationships belonging to a specific graph."""
        return self.find_by_criteria({"graph_name": graph_name})

    def find_by_entities(
        self, entity_ids: list[str], graph_name: str
    ) -> list["KGRelationship"]:
        """Find relationships involving any of the given entity IDs.

        Uses ContainsAny filter on the entity_ids TEXT_ARRAY field.
        """
        if not entity_ids:
            return []
        try:
            results = self.collection.query.fetch_objects(
                filters=(
                    Filter.by_property("entity_ids").contains_any(entity_ids)
                    & Filter.by_property("graph_name").equal(graph_name)
                ),
                limit=10000,
            )
            return [
                self._weaviate_object_to_entity(obj)
                for obj in results.objects
                if self._weaviate_object_to_entity(obj) is not None
            ]
        except Exception as e:
            logger.error("Failed to find relationships by entities: %s", e)
            return []

    def delete_by_graph(self, graph_name: str) -> int:
        """Delete all relationships for a graph. Returns count deleted."""
        rels = self.find_by_graph(graph_name)
        count = 0
        for rel in rels:
            if rel.id and self.delete(rel.id):
                count += 1
        return count


class KGGraphRepository(WeaviateRepository["KGGraph"]):
    """Repository for knowledge graph metadata."""

    def __init__(self, config: WeaviateConfig):
        from history_book.data_models.kg_entities import KGGraph  # noqa: PLC0415

        super().__init__(
            config=config,
            collection_name="KGGraphs",
            entity_class=KGGraph,
        )

    def find_by_name(self, name: str) -> "KGGraph | None":
        """Find a graph by its unique name."""
        results = self.find_by_criteria({"name": name})
        return results[0] if results else None

    def find_by_type(self, graph_type: str) -> list["KGGraph"]:
        """Find all graphs of a specific type (chapter, book, volume)."""
        return self.find_by_criteria({"graph_type": graph_type})
