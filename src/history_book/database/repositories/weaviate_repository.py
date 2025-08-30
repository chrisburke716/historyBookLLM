"""Weaviate implementation of the vector repository interface."""

import logging
from typing import Any, TypeVar

import weaviate
from weaviate import WeaviateClient
from weaviate.collections import Collection

from ..config.database_config import WeaviateConfig
from ..exceptions.database_exceptions import (
    BatchOperationError,
    CollectionError,
    ConnectionError,
    QueryError,
    ValidationError,
    VectorError,
)
from ..interfaces.vector_repository_interface import VectorRepository

logger = logging.getLogger(__name__)

T = TypeVar("T")

"""
NOTE: This code was largely ai-generated and may not be fully functional.
"""


class WeaviateRepository(VectorRepository[T]):
    """
    Weaviate implementation of the vector repository interface.

    This class provides concrete implementations of all repository operations
    for Weaviate vector database, handling connection management, error handling,
    and Weaviate-specific operations.
    """

    def __init__(
        self,
        config: WeaviateConfig,
        collection_name: str,
        entity_class: type[T],
        client: WeaviateClient | None = None,
    ):
        """
        Initialize the Weaviate repository.

        Args:
            config: Weaviate configuration
            collection_name: Name of the collection to operate on
            entity_class: The entity class this repository manages
            client: Optional pre-configured Weaviate client
        """
        self.config = config
        self.collection_name = (
            collection_name.capitalize()
        )  # Weaviate uses capitalized names
        self.entity_class = entity_class
        self._client = client
        self._collection: Collection | None = None

    @property
    def client(self) -> WeaviateClient:
        """Get or create the Weaviate client."""
        if self._client is None:
            try:
                if self.config.is_local:
                    self._client = weaviate.connect_to_local(
                        port=self.config.port, grpc_port=self.config.grpc_port
                    )
                else:
                    # For remote connections, you'd configure differently
                    # This is a placeholder for future remote connection support
                    raise NotImplementedError(
                        "Remote Weaviate connections not yet implemented"
                    )

                logger.info(f"Connected to Weaviate at {self.config.connection_string}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}", e)

        return self._client

    @property
    def collection(self) -> Collection:
        """Get or create the collection."""
        if self._collection is None:
            try:
                # Check if collection exists
                if self.collection_name in self.client.collections.list_all().keys():
                    self._collection = self.client.collections.get(self.collection_name)
                    logger.info(
                        f"Retrieved existing collection: {self.collection_name}"
                    )
                else:
                    # Create collection if it doesn't exist
                    logger.info(f"Creating new collection: {self.collection_name}")
                    self._collection = self._create_collection()

            except Exception as e:
                if isinstance(e, CollectionError):
                    raise
                raise CollectionError(
                    f"Failed to access collection '{self.collection_name}': {str(e)}", e
                )

        return self._collection

    def _create_collection(self) -> Collection:
        """Create a new collection with proper configuration."""
        from ..collections import create_collection_from_pydantic

        return create_collection_from_pydantic(
            client=self.client,
            model_class=self.entity_class,
            class_name=self.collection_name,
        )

    def close(self):
        """Close the database connection."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("Weaviate connection closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate connection: {str(e)}")
            finally:
                self._client = None
                self._collection = None

    # Sync versions of the interface methods (for now, we'll implement sync versions)
    # In the future, you could add async support by wrapping these in asyncio.run()

    def create(self, entity: T, **kwargs) -> str:
        """Synchronous version of create."""
        try:
            # Convert entity to dict, filtering out None values and internal fields
            entity_data = self._entity_to_dict(entity)
            entity_id = entity_data.pop("id", None)

            # Insert into collection
            result = self.collection.data.insert(
                properties=entity_data, uuid=entity_id, **kwargs
            )

            logger.debug(f"Created entity with ID: {result}")
            return str(result)

        except Exception as e:
            raise CollectionError(f"Failed to create entity: {str(e)}", e)

    def get_by_id(self, entity_id: str, **kwargs) -> T | None:
        """Synchronous version of get_by_id."""
        try:
            # Fetch object with vector included
            result = self.collection.query.fetch_object_by_id(
                entity_id, include_vector=True, **kwargs
            )
            if result is None:
                return None

            # Convert Weaviate object back to entity
            return self._weaviate_object_to_entity(result)

        except Exception as e:
            logger.error(f"Failed to retrieve entity {entity_id}: {str(e)}")
            return None

    def update(self, entity_id: str, updates: dict[str, Any], **kwargs) -> bool:
        """Synchronous version of update."""
        try:
            self.collection.data.update(uuid=entity_id, properties=updates, **kwargs)
            logger.debug(f"Updated entity {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update entity {entity_id}: {str(e)}")
            return False

    def delete(self, entity_id: str, **kwargs) -> bool:
        """Synchronous version of delete."""
        try:
            self.collection.data.delete_by_id(entity_id, **kwargs)
            logger.debug(f"Deleted entity {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete entity {entity_id}: {str(e)}")
            return False

    def list_all(
        self, limit: int | None = None, offset: int | None = None, **kwargs
    ) -> list[T]:
        """Synchronous version of list_all."""
        try:
            query = self.collection.query.fetch_objects(
                limit=limit, offset=offset, include_vector=True, **kwargs
            )

            entities = []
            for obj in query.objects:
                entity = self._weaviate_object_to_entity(obj)
                if entity:
                    entities.append(entity)

            return entities

        except Exception as e:
            raise QueryError(f"Failed to list entities: {str(e)}", e)

    def count(self, **kwargs) -> int:
        """Synchronous version of count."""
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            return result.total_count or 0

        except Exception as e:
            logger.error(f"Failed to count entities: {str(e)}")
            return 0

    def exists(self, entity_id: str, **kwargs) -> bool:
        """Synchronous version of exists."""
        return self.get_by_id(entity_id, **kwargs) is not None

    def find_by_criteria(self, criteria: dict[str, Any], **kwargs) -> list[T]:
        """Synchronous version of find_by_criteria."""
        try:
            # For now, let's implement a simple version that fetches all and filters in Python
            # This is not efficient but will allow us to test the interface
            query = self.collection.query.fetch_objects(**kwargs)

            entities = []
            for obj in query.objects:
                entity = self._weaviate_object_to_entity(obj)
                if entity:
                    # Apply criteria filtering in Python
                    if self._entity_matches_criteria(entity, criteria):
                        entities.append(entity)

            return entities

        except Exception as e:
            raise QueryError(f"Failed to find entities by criteria: {str(e)}", e)

    def _entity_matches_criteria(self, entity: T, criteria: dict[str, Any]) -> bool:
        """Check if an entity matches the given criteria."""
        for field, value in criteria.items():
            if hasattr(entity, field):
                entity_value = getattr(entity, field)
                if entity_value != value:
                    return False
            else:
                return False
        return True

    # Vector-specific methods

    def similarity_search(
        self,
        query_vector: list[float],
        limit: int = 10,
        threshold: float | None = None,
        where_filter: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[tuple[T, float]]:
        """Synchronous version of similarity_search."""
        try:
            # Build where filter if provided
            where = None
            if where_filter:
                where = self._build_where_filter(where_filter)

            query = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                distance=threshold,
                filters=where,
                return_metadata=["distance"],  # Include distance in results
                **kwargs,
            )

            results = []
            for obj in query.objects:
                entity = self._weaviate_object_to_entity(obj)
                if entity:
                    # Get distance/score from metadata
                    score = getattr(obj.metadata, "distance", 0.0)
                    results.append(
                        (entity, 1.0 - score)
                    )  # Convert distance to similarity

            return results

        except Exception as e:
            raise VectorError(f"Vector similarity search failed: {str(e)}", e)

    def similarity_search_by_text(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float | None = None,
        where_filter: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[tuple[T, float]]:
        """Synchronous version of similarity_search_by_text."""
        try:
            # Build where filter if provided
            where = None
            if where_filter:
                where = self._build_where_filter(where_filter)

            query = self.collection.query.near_text(
                query=query_text,
                limit=limit,
                distance=threshold,
                filters=where,
                return_metadata=["distance"],  # Include distance in results
                **kwargs,
            )

            results = []
            for obj in query.objects:
                entity = self._weaviate_object_to_entity(obj)
                if entity:
                    # Get distance/score from metadata, handle None case
                    distance = getattr(obj.metadata, "distance", None)
                    if distance is not None:
                        score = 1.0 - distance  # Convert distance to similarity
                    else:
                        score = 0.0  # Default score if distance is None
                    results.append((entity, score))

            return results

        except Exception as e:
            raise VectorError(f"Text similarity search failed: {str(e)}", e)

    def create_with_vector(
        self, entity: T, vector: list[float] | None = None, **kwargs
    ) -> str:
        """Synchronous version of create_with_vector."""
        try:
            entity_data = self._entity_to_dict(entity)
            entity_id = entity_data.pop("id", None)

            insert_kwargs = kwargs.copy()
            if vector:
                insert_kwargs["vector"] = vector

            result = self.collection.data.insert(
                properties=entity_data, uuid=entity_id, **insert_kwargs
            )

            logger.debug(f"Created entity with vector, ID: {result}")
            return str(result)

        except Exception as e:
            raise VectorError(f"Failed to create entity with vector: {str(e)}", e)

    def update_vector(self, entity_id: str, vector: list[float], **kwargs) -> bool:
        """Synchronous version of update_vector."""
        try:
            self.collection.data.update(uuid=entity_id, vector=vector, **kwargs)
            logger.debug(f"Updated vector for entity {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update vector for entity {entity_id}: {str(e)}")
            return False

    def get_vector(self, entity_id: str, **kwargs) -> list[float] | None:
        """Synchronous version of get_vector."""
        try:
            result = self.collection.query.fetch_object_by_id(
                entity_id, include_vector=True, **kwargs
            )

            if result and result.vector:
                # Return the first vector (assuming single vector per object)
                return (
                    list(result.vector.values())[0] if result.vector.values() else None
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get vector for entity {entity_id}: {str(e)}")
            return None

    def batch_create_with_vectors(
        self, entities_and_vectors: list[tuple[T, list[float] | None]], **kwargs
    ) -> list[str]:
        """Synchronous version of batch_create_with_vectors."""
        try:
            # Prepare batch data
            # objects_to_insert = []
            created_ids = []
            with self.collection.batch.fixed_size(batch_size=100) as batch:
                for entity, vector in entities_and_vectors:
                    entity_data = self._entity_to_dict(entity)
                    entity_id = entity_data.pop("id", None)

                    obj_data = {
                        "properties": entity_data,
                        "uuid": entity_id,  # Use entity's ID if available
                    }

                    if vector:
                        obj_data["vector"] = vector

                    id_added = batch.add_object(**obj_data)
                    if id_added:
                        created_ids.append(str(id_added))

            logger.warning(f"{batch.number_errors} errors in batch create operation")
            logger.info(f"Batch created {len(created_ids)} entities")
            return created_ids

        except Exception as e:
            raise BatchOperationError(
                f"Batch create operation failed: {str(e)}", original_error=e
            )

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float] | None = None,
        alpha: float = 0.5,
        limit: int = 10,
        **kwargs,
    ) -> list[tuple[T, float]]:
        """Synchronous version of hybrid_search."""
        try:
            # Use Weaviate's hybrid search if available
            query = self.collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                **kwargs,
            )

            results = []
            for obj in query.objects:
                entity = self._weaviate_object_to_entity(obj)
                if entity:
                    score = getattr(obj.metadata, "score", 0.0)
                    results.append((entity, score))

            return results

        except Exception as e:
            raise VectorError(f"Hybrid search failed: {str(e)}", e)

    # Helper methods

    def _entity_to_dict(self, entity: T) -> dict[str, Any]:
        """Convert entity to dictionary suitable for Weaviate."""
        if hasattr(entity, "model_dump"):
            # Pydantic model
            data = entity.model_dump()
        elif hasattr(entity, "__dict__"):
            # Regular class
            data = entity.__dict__.copy()
        else:
            raise ValidationError(
                f"Cannot convert entity of type {type(entity)} to dict"
            )

        # Filter out None values and internal fields
        filtered_data = {}
        for key, value in data.items():
            if key in ["client", "collection"]:
                continue
            if value is not None:
                filtered_data[key] = value

        return filtered_data

    def _weaviate_object_to_entity(self, weaviate_obj) -> T | None:
        """Convert Weaviate object back to entity."""
        try:
            # Extract properties - handle both direct properties and nested 'properties' field
            if hasattr(weaviate_obj, "properties") and isinstance(
                weaviate_obj.properties, dict
            ):
                properties = weaviate_obj.properties.copy()

                # Check if data is nested in a 'properties' sub-field (common Weaviate issue)
                if "properties" in properties and isinstance(
                    properties["properties"], dict
                ):
                    # Use the nested properties as the actual data
                    nested_props = properties["properties"]
                    # Keep top-level fields that aren't duplicated
                    for key, value in properties.items():
                        if key != "properties" and key not in nested_props:
                            nested_props[key] = value
                    properties = nested_props
            else:
                properties = {}

            # Add the ID
            properties["id"] = str(weaviate_obj.uuid)

            # Add vector embedding if available and entity supports it
            if hasattr(weaviate_obj, "vector") and weaviate_obj.vector is not None:
                if "embedding" in self.entity_class.model_fields:
                    # Handle different vector formats
                    if isinstance(weaviate_obj.vector, dict):
                        # For named vectors, look for text_vector first, then default
                        if "text_vector" in weaviate_obj.vector and isinstance(
                            weaviate_obj.vector["text_vector"], list
                        ):
                            properties["embedding"] = weaviate_obj.vector["text_vector"]
                        elif "default" in weaviate_obj.vector and isinstance(
                            weaviate_obj.vector["default"], list
                        ):
                            properties["embedding"] = weaviate_obj.vector["default"]
                        else:
                            # Handle other dict formats or skip if unrecognized
                            logger.debug(
                                f"Skipping vector - unrecognized dict format: {list(weaviate_obj.vector.keys())}"
                            )
                    elif isinstance(weaviate_obj.vector, list):
                        properties["embedding"] = weaviate_obj.vector
                    else:
                        logger.debug(
                            f"Skipping vector - unrecognized format: {type(weaviate_obj.vector)}"
                        )

            # Filter out legacy fields that don't belong in pure entities
            legacy_fields = {"client", "collection", "uuid"}
            filtered_properties = {
                k: v
                for k, v in properties.items()
                if k not in legacy_fields and v is not None
            }

            # Create entity instance using model_construct for better compatibility
            if hasattr(self.entity_class, "model_construct"):
                # For Pydantic models, use model_construct to bypass validation
                entity = self.entity_class.model_construct(**filtered_properties)
                return entity
            else:
                # Regular class - create directly
                return self.entity_class(**filtered_properties)

        except Exception as e:
            logger.error(f"Failed to convert Weaviate object to entity: {str(e)}")
            logger.error(f"Properties: {getattr(weaviate_obj, 'properties', 'N/A')}")
            return None

    def _build_where_filter(self, criteria: dict[str, Any]):
        """Build Weaviate where filter from criteria dictionary."""
        from weaviate.classes.query import Filter

        if not criteria:
            return None

        # Build filter conditions
        filters = []
        for field, value in criteria.items():
            if isinstance(value, str) or isinstance(value, (int, float)) or isinstance(value, bool):
                filters.append(Filter.by_property(field).equal(value))

        if len(filters) == 1:
            return filters[0]
        elif len(filters) > 1:
            # Combine with AND
            result = filters[0]
            for filter_condition in filters[1:]:
                result = result & filter_condition
            return result
        else:
            return None
