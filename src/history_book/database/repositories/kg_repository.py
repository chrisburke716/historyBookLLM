"""Knowledge graph repository implementations."""

import logging
from typing import TYPE_CHECKING

from weaviate.classes.query import Filter, Sort

from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories.weaviate_repository import WeaviateRepository

if TYPE_CHECKING:
    from history_book.data_models.kg_entities import (
        KGEntity,
        KGGraph,
        KGMergeDecision,
        KGRelationship,
    )

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
        results = self.find_by_criteria({"graph_name": graph_name})
        return [r for r in results if r.graph_name == graph_name]

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
                e
                for obj in results.objects
                if (e := self._weaviate_object_to_entity(obj)) is not None
                and e.graph_name == graph_name
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

    def search_entities_hybrid(
        self,
        query_text: str,
        graph_name: str | None = None,
        limit: int = 100,
        alpha: float = 0.5,
    ) -> list[tuple["KGEntity", float]]:
        """Search entities using hybrid search (vector + BM25).

        Args:
            query_text: Text to search for.
            graph_name: Optional graph to scope the search to.
            limit: Maximum number of results.
            alpha: Balance between vector (1.0) and BM25 (0.0) search.
        """
        kwargs = {}
        if graph_name:
            kwargs["filters"] = self._build_where_filter({"graph_name": graph_name})
        return self.hybrid_search(
            query_text=query_text,
            limit=limit,
            alpha=alpha,
            **kwargs,
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
        results = self.find_by_criteria({"graph_name": graph_name})
        return [r for r in results if r.graph_name == graph_name]

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
                r
                for obj in results.objects
                if (r := self._weaviate_object_to_entity(obj)) is not None
                and r.graph_name == graph_name
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
        """Find a graph by its unique name.

        Uses exact string matching because Weaviate TEXT fields use word
        tokenization (e.g. 'book3' would also match 'book3_ch2').
        """
        results = self.find_by_criteria({"name": name})
        for r in results:
            if r.name == name:
                return r
        return None

    def find_by_type(self, graph_type: str) -> list["KGGraph"]:
        """Find all graphs of a specific type (chapter, book, volume)."""
        return self.find_by_criteria({"graph_type": graph_type})


class KGMergeDecisionRepository(WeaviateRepository["KGMergeDecision"]):
    """Repository for KG merge audit decisions."""

    def __init__(self, config: WeaviateConfig):
        from history_book.data_models.kg_entities import KGMergeDecision  # noqa: PLC0415, I001

        super().__init__(
            config=config,
            collection_name="KGMergeDecisions",
            entity_class=KGMergeDecision,
        )

    def find_by_graph(self, graph_name: str) -> list["KGMergeDecision"]:
        """Find all merge decisions for a specific graph.

        Uses exact string match (same pattern as KGGraphRepository.find_by_name)
        to avoid Weaviate tokenization matching adjacent graph names.
        """
        results = self.find_by_criteria({"graph_name": graph_name})
        return sorted(
            [r for r in results if r.graph_name == graph_name],
            key=lambda d: d.occurrence_count_after,
        )

    def find_by_entity_name(
        self, name: str, graph_name: str | None = None
    ) -> list["KGMergeDecision"]:
        """Find merge decisions involving an entity by canonical or entity name."""
        try:
            name_filter = (
                Filter.by_property("entity1_name").equal(name)
                | Filter.by_property("entity2_name").equal(name)
                | Filter.by_property("canonical_name").equal(name)
            )
            combined = (
                name_filter & Filter.by_property("graph_name").equal(graph_name)
                if graph_name
                else name_filter
            )
            results = self.collection.query.fetch_objects(
                filters=combined,
                sort=Sort.by_property("occurrence_count_after", ascending=True),
                limit=1000,
            )
            return [
                d
                for obj in results.objects
                if (d := self._weaviate_object_to_entity(obj)) is not None
                and (graph_name is None or d.graph_name == graph_name)
            ]
        except Exception as e:
            logger.error("Failed to find merge decisions by entity name: %s", e)
            return []

    def find_by_entity(
        self, name: str, entity_type: str, graph_name: str | None = None
    ) -> list["KGMergeDecision"]:
        """Find merge decisions involving a specific entity (name + type).

        Matches name against entity1_name, entity2_name, and canonical_name,
        and type against entity1_type or entity2_type.
        """
        try:
            name_filter = (
                Filter.by_property("entity1_name").equal(name)
                | Filter.by_property("entity2_name").equal(name)
                | Filter.by_property("canonical_name").equal(name)
            )
            type_filter = (
                Filter.by_property("entity1_type").equal(entity_type)
                | Filter.by_property("entity2_type").equal(entity_type)
            )
            combined = name_filter & type_filter
            if graph_name:
                combined = combined & Filter.by_property("graph_name").equal(graph_name)
            results = self.collection.query.fetch_objects(
                filters=combined,
                sort=Sort.by_property("occurrence_count_after", ascending=True),
                limit=1000,
            )
            return [
                d
                for obj in results.objects
                if (d := self._weaviate_object_to_entity(obj)) is not None
                and (graph_name is None or d.graph_name == graph_name)
            ]
        except Exception as e:
            logger.error("Failed to find merge decisions by entity: %s", e)
            return []

    def delete_by_graph(self, graph_name: str) -> int:
        """Delete all merge decisions for a graph. Returns count deleted."""
        decisions = self.find_by_graph(graph_name)
        count = 0
        for d in decisions:
            if d.id and self.delete(d.id):
                count += 1
        return count
