"""Data models for knowledge graph entities stored in Weaviate."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    """Allowed entity types for knowledge graph entities."""

    PERSON = "person"
    POLITY = "polity"
    PLACE = "place"
    EVENT = "event"
    CONCEPT = "concept"


class KGEntity(BaseModel):
    """A normalized entity in a knowledge graph.

    Stored in Weaviate with vector search on name, type, description, and aliases.
    """

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    graph_name: str
    name: str
    entity_type: str  # validated via EntityType enum in extraction chain
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    occurrence_count: int = 1
    book_indices: list[int] = Field(default_factory=list)
    source_book_chapters: list[str] = Field(
        default_factory=list
    )  # ["3:2", "3:3"] — book:chapter provenance
    source_pages: list[int] = Field(default_factory=list)
    source_paragraph_ids: list[str] = Field(default_factory=list)
    merged_from_count: int = 0

    vectorize_fields: ClassVar[list[str]] = [
        "name",
        "entity_type",
        "description",
        "aliases",
    ]


class KGRelationship(BaseModel):
    """A relationship between two entities in a knowledge graph."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    graph_name: str
    source_entity_id: str
    target_entity_id: str
    entity_ids: list[str] = Field(
        default_factory=list
    )  # [source_entity_id, target_entity_id] for ContainsAny queries
    source_entity_name: str
    target_entity_name: str
    relation_type: str  # ruled, conquered, fought, etc.
    temporal_context: str = ""
    start_year: int | None = None
    end_year: int | None = None
    temporal_precision: str | None = None
    paragraph_id: str
    book_index: int
    chapter_index: int
    page: int


class KGGraph(BaseModel):
    """Metadata for a named knowledge graph."""

    id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # "book3_ch2", "book3", "volume_full"
    graph_type: str  # "chapter", "book", "volume"
    book_chapters: list[str] = Field(
        default_factory=list
    )  # ["3:0", "3:1", "4:0"] — what's included
    entity_count: int = 0
    relationship_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
