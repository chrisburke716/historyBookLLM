"""
Extract entities and relationships from historical text and visualize as a knowledge graph.

This script implements the full KG pipeline:
1. Entity extraction with GPT-4o (LangChain structured outputs)
2. ID assignment with bidirectional links
3. Entity normalization (exact, alias-based, optional fuzzy)
4. Graph construction with NetworkX
5. Interactive visualization with PyVis

Usage:
    python extract_and_visualize_kg.py --input selected_5_paragraphs.json --output kg_test.html
    python extract_and_visualize_kg.py --book 3 --chapter 4 --output rome_chapter_kg.html
"""

import argparse
import json
import sys
import time
import uuid as uuid_module
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import networkx as nx
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import BaseModel, Field
from pyvis.network import Network
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Add src to path for database access
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories.book_repository import BookRepositoryManager


# ============================================================================
# Pydantic Models
# ============================================================================


class Entity(BaseModel):
    """Extracted entity from historical text (LLM output)."""

    name: str
    type: str  # "person", "place", "collective_entity", "event", "temporal", "cultural"
    subtype: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    attributes: Optional[dict[str, str]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """Relationship between two entities (LLM output)."""

    source_entity: str  # Entity name
    relation_type: str
    target_entity: str  # Entity name
    temporal_context: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Result of entity extraction from a paragraph."""

    entities: list[Entity]
    relationships: list[Relationship]
    paragraph_id: str


class EntityWithId(BaseModel):
    """Entity with UUID assigned after extraction."""

    id: str  # UUID
    name: str
    type: str
    subtype: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    attributes: Optional[dict[str, str]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    paragraph_id: str
    relationship_ids: list[str] = Field(default_factory=list)  # Bidirectional link


class RelationshipWithId(BaseModel):
    """Relationship with UUIDs for entities."""

    id: str  # UUID
    source_id: str  # Entity UUID
    target_id: str  # Entity UUID
    relation_type: str
    temporal_context: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    paragraph_id: str


class NormalizedEntity(BaseModel):
    """Normalized entity after merging duplicates."""

    id: str  # Canonical UUID
    name: str
    type: str
    subtype: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    description: str  # Combined descriptions
    attributes: Optional[dict[str, str]] = None
    source_paragraph_ids: list[str]
    occurrence_count: int
    merged_from_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)


class NormalizedRelationship(BaseModel):
    """Normalized relationship with canonical entity IDs."""

    id: str  # UUID (preserved from original)
    source_id: str  # Normalized entity ID
    target_id: str  # Normalized entity ID
    relation_type: str
    temporal_context: Optional[str] = None
    confidence: float
    paragraph_id: str


# ============================================================================
# Extraction
# ============================================================================

EXTRACTION_PROMPT = """You are analyzing text from "The Penguin History of the World".

Extract ALL entities and relationships from the provided paragraph.

**ENTITY TYPES**:
- person: Individuals (rulers, leaders, historical figures)
- place: Geographic locations (cities, regions, rivers, etc.)
- collective_entity: Groups, states, organizations, peoples, leagues
- event: Historical events, political actions, battles, reforms
- temporal: Time references (centuries, years, dates)
- cultural: Cultural concepts, traditions, civilizations

**SUBTYPES**:
- place: "city", "region", "river", "sea"
- collective_entity: "state", "people", "organization", "league"
- temporal: "century" (single century), "year" (single year), "date" (specific date), "range" (time range)

**RELATIONSHIP TYPES** (examples - extract any you find):
- Political: "ruled", "conquered", "allied-with", "subordinated", "revolted-against", "succeeded"
- Geographic: "located-on", "located-in", "bordered-by"
- Cultural: "influenced-by", "came-from", "accessed-through"
- Temporal: "happened-in", "occurred-during"

**IMPORTANT GUIDELINES**:
1. Extract entities FROM THIS PARAGRAPH ONLY - do not use external knowledge
2. Extract relationships that are EXPLICITLY STATED in the text
3. Include aliases if the entity is referred to by multiple names (e.g., "Octavian" also called "Augustus")
4. For titles/roles, store as attributes (e.g., Caesar's "dictator for life")
5. DO NOT extract:
   - Generic unnamed groups ("his men", "the soldiers")
   - Entities mentioned only as comparisons ("like Athens")
   - Vague references without clear identity
6. Include temporal_context in relationships when dates/times are mentioned
7. Note: Some entities may have dual nature (e.g., Rome as both a city and a political state) - extract both if clear from context
8. The extracted entities and relationships will be used to build a knowledge graph. To this end, relationships must be clearly defined between identified entities.

Extract entities and relationships from this paragraph:

{paragraph_text}
"""


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def extract_entities_gpt4o(paragraph_text: str, paragraph_id: str) -> ExtractionResult:
    """
    Extract entities and relationships using GPT-4o with LangChain structured outputs.

    Includes retry logic with exponential backoff for rate limit errors:
    - Retries up to 5 times
    - Waits 4s, 8s, 16s, 32s, 60s between retries
    - Re-raises exception if all retries fail
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    model_with_structure = model.with_structured_output(ExtractionResult)

    system_message = "You are an expert at extracting structured historical entities and relationships from text."
    user_message = EXTRACTION_PROMPT.format(paragraph_text=paragraph_text)

    result = model_with_structure.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )

    result.paragraph_id = paragraph_id
    return result


# ============================================================================
# ID Assignment
# ============================================================================


def assign_ids_to_extraction_results(
    all_results: list[ExtractionResult],
) -> tuple[list[EntityWithId], list[RelationshipWithId]]:
    """
    Assign UUIDs to entities and relationships, create bidirectional links.

    Processes one paragraph at a time for clarity.

    Returns:
        (entities_with_ids, relationships_with_ids)
    """
    all_entities_with_ids = []
    all_relationships_with_ids = []
    skipped_relationships = 0

    # Process each paragraph
    for result in all_results:
        paragraph_entities = {}  # entity_name -> EntityWithId for this paragraph
        paragraph_relationships = []

        # Step 1: Create entities with IDs for this paragraph
        for entity in result.entities:
            entity_id = str(uuid_module.uuid4())
            entity_with_id = EntityWithId(
                id=entity_id,
                name=entity.name,
                type=entity.type,
                subtype=entity.subtype,
                aliases=entity.aliases,
                description=entity.description,
                attributes=entity.attributes,
                confidence=entity.confidence,
                paragraph_id=result.paragraph_id,
                relationship_ids=[],  # Will populate as we process relationships
            )
            paragraph_entities[entity.name] = entity_with_id

        # Step 2: Create relationships with IDs and build bidirectional links
        for rel in result.relationships:
            source_entity = paragraph_entities.get(rel.source_entity)
            target_entity = paragraph_entities.get(rel.target_entity)

            # Only create relationship if both entities exist
            if source_entity and target_entity:
                rel_id = str(uuid_module.uuid4())
                rel_with_id = RelationshipWithId(
                    id=rel_id,
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    relation_type=rel.relation_type,
                    temporal_context=rel.temporal_context,
                    confidence=rel.confidence,
                    paragraph_id=result.paragraph_id,
                )
                paragraph_relationships.append(rel_with_id)

                # Build bidirectional links
                source_entity.relationship_ids.append(rel_id)
                target_entity.relationship_ids.append(rel_id)
            else:
                skipped_relationships += 1

        # Add to global lists
        all_entities_with_ids.extend(paragraph_entities.values())
        all_relationships_with_ids.extend(paragraph_relationships)

    print(f"Created {len(all_entities_with_ids)} entities with IDs")
    print(f"Created {len(all_relationships_with_ids)} relationships with IDs")
    if skipped_relationships > 0:
        print(
            f"Skipped {skipped_relationships} relationships (referenced entities not found)"
        )

    return all_entities_with_ids, all_relationships_with_ids


# ============================================================================
# Normalization
# ============================================================================


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _alias_based_merging(exact_groups: dict) -> dict:
    """Merge entity groups if one group's name appears in another group's aliases."""
    merged_groups = dict(exact_groups)

    # Build alias lookup
    alias_to_canonical = {}
    for canonical_name, entities in merged_groups.items():
        for entity in entities:
            if entity.aliases:
                for alias in entity.aliases:
                    alias_key = alias.lower().strip()
                    if alias_key and alias_key != canonical_name:
                        alias_to_canonical[alias_key] = canonical_name

    print(f"  Found {len(alias_to_canonical)} alias mappings")

    # Merge based on aliases (with frequency-based canonical selection)
    merged_any = True
    merge_count = 0

    while merged_any:
        merged_any = False
        for name in list(merged_groups.keys()):
            if name not in merged_groups:
                continue

            if name in alias_to_canonical:
                other_name = alias_to_canonical[name]
                if other_name in merged_groups and other_name != name:
                    # Frequency-based canonical selection
                    name_count = len(merged_groups[name])
                    other_count = len(merged_groups[other_name])

                    if name_count > other_count:
                        canonical, merge_from = name, other_name
                    elif other_count > name_count:
                        canonical, merge_from = other_name, name
                    else:
                        # Tie-break: shorter name wins
                        canonical = (
                            name if len(name) < len(other_name) else other_name
                        )
                        merge_from = other_name if canonical == name else name

                    merged_groups[canonical].extend(merged_groups[merge_from])
                    del merged_groups[merge_from]
                    merged_any = True
                    merge_count += 1
                    print(f"  Merged '{merge_from}' into '{canonical}'")

                    # Rebuild alias lookup
                    alias_to_canonical = {}
                    for canonical_name, entities in merged_groups.items():
                        for entity in entities:
                            if entity.aliases:
                                for alias in entity.aliases:
                                    alias_key = alias.lower().strip()
                                    if alias_key and alias_key != canonical_name:
                                        alias_to_canonical[alias_key] = canonical_name
                    break

    print(f"  Completed {merge_count} alias-based merges")
    return merged_groups


def normalize_entities_and_relationships(
    entities_with_ids: list[EntityWithId],
    relationships_with_ids: list[RelationshipWithId],
    fuzzy_threshold: float = 0.90,
    fuzzy_merge_pairs: list = None,
) -> tuple[list[NormalizedEntity], list[NormalizedRelationship]]:
    """
    Normalize entities by merging duplicates, update relationships to use normalized IDs.

    Returns:
        (normalized_entities, normalized_relationships)
    """
    print(f"\nStarting normalization with {len(entities_with_ids)} entities")

    # Stage 1: Group entities by exact name match (case-insensitive)
    exact_groups = defaultdict(list)
    for entity in entities_with_ids:
        key = entity.name.lower().strip()
        exact_groups[key].append(entity)

    print(f"Unique names after exact matching: {len(exact_groups)}")

    # Stage 2: Alias-based merging
    merged_groups = _alias_based_merging(exact_groups)
    print(f"Unique names after alias-based merging: {len(merged_groups)}")

    # Stage 3: Fuzzy matching (optional - for now we skip)
    if fuzzy_merge_pairs:
        print(f"Applying {len(fuzzy_merge_pairs)} fuzzy merges...")
        # Apply fuzzy merges if provided (not implemented in v1)

    # Stage 4: Create normalized entities and track ID mappings
    normalized_entities = []
    old_id_to_normalized_id = {}  # old entity ID -> normalized entity ID

    for name, entity_group in merged_groups.items():
        # Create normalized entity
        canonical_id = str(uuid_module.uuid4())
        merged_from_ids = [e.id for e in entity_group]

        # Aggregate data from all entities in group
        all_aliases = set()
        all_descriptions = []
        all_paragraphs = set()
        all_attributes = {}
        all_relationship_ids = set()
        base_entity = entity_group[0]

        for entity in entity_group:
            if entity.aliases:
                all_aliases.update(entity.aliases)
            if entity.description:
                all_descriptions.append(entity.description)
            all_paragraphs.add(entity.paragraph_id)
            if entity.attributes:
                all_attributes.update(entity.attributes)
            all_relationship_ids.update(entity.relationship_ids)

            # Track old ID -> normalized ID mapping
            old_id_to_normalized_id[entity.id] = canonical_id

        normalized_entity = NormalizedEntity(
            id=canonical_id,
            name=base_entity.name,
            type=base_entity.type,
            subtype=base_entity.subtype,
            aliases=list(all_aliases),
            description=" | ".join(all_descriptions) if all_descriptions else "",
            attributes=all_attributes if all_attributes else None,
            source_paragraph_ids=list(all_paragraphs),
            occurrence_count=len(entity_group),
            merged_from_ids=merged_from_ids,
            relationship_ids=list(
                all_relationship_ids
            ),  # Will update after normalizing relationships
        )
        normalized_entities.append(normalized_entity)

    print(f"Created {len(normalized_entities)} normalized entities")

    # Stage 5: Normalize relationships (update entity IDs, rebuild bidirectional links)
    normalized_relationships = []
    normalized_entity_id_to_obj = {e.id: e for e in normalized_entities}

    # Reset relationship_ids on normalized entities (will rebuild)
    for entity in normalized_entities:
        entity.relationship_ids = []

    for rel in relationships_with_ids:
        # Map old entity IDs to normalized IDs
        norm_source_id = old_id_to_normalized_id.get(rel.source_id)
        norm_target_id = old_id_to_normalized_id.get(rel.target_id)

        if norm_source_id and norm_target_id:
            # Create normalized relationship
            norm_rel = NormalizedRelationship(
                id=rel.id,  # Preserve original relationship ID
                source_id=norm_source_id,
                target_id=norm_target_id,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                confidence=rel.confidence,
                paragraph_id=rel.paragraph_id,
            )
            normalized_relationships.append(norm_rel)

            # Rebuild bidirectional links
            if norm_source_id in normalized_entity_id_to_obj:
                normalized_entity_id_to_obj[norm_source_id].relationship_ids.append(
                    rel.id
                )
            if norm_target_id in normalized_entity_id_to_obj:
                normalized_entity_id_to_obj[norm_target_id].relationship_ids.append(
                    rel.id
                )

    print(f"Normalized {len(normalized_relationships)} relationships")
    skipped = len(relationships_with_ids) - len(normalized_relationships)
    if skipped > 0:
        print(f"Skipped {skipped} relationships (entities not found)")

    return normalized_entities, normalized_relationships


# ============================================================================
# Graph Construction
# ============================================================================


def build_knowledge_graph(
    normalized_entities: list[NormalizedEntity],
    normalized_relationships: list[NormalizedRelationship],
) -> nx.DiGraph:
    """
    Build a directed graph from normalized entities and relationships.

    Returns:
        NetworkX DiGraph with entity nodes and relationship edges
    """
    G = nx.DiGraph()

    # Add nodes for each normalized entity
    for entity in normalized_entities:
        G.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            subtype=entity.subtype,
            aliases=entity.aliases,
            description=entity.description,
            attributes=entity.attributes,
            occurrence_count=entity.occurrence_count,
            source_paragraphs=entity.source_paragraph_ids,
            relationship_count=len(entity.relationship_ids),
        )

    print(f"\nAdded {G.number_of_nodes()} nodes to graph")

    # Add edges for relationships
    for rel in normalized_relationships:
        if rel.source_id in G and rel.target_id in G:
            # Check if edge already exists (from another relationship)
            if G.has_edge(rel.source_id, rel.target_id):
                # Aggregate relation types
                edge_data = G[rel.source_id][rel.target_id]
                if rel.relation_type not in edge_data.get("relation_types", []):
                    edge_data.setdefault("relation_types", []).append(rel.relation_type)
            else:
                # New edge
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_id=rel.id,
                    relation_type=rel.relation_type,
                    relation_types=[rel.relation_type],
                    temporal_context=rel.temporal_context,
                    confidence=rel.confidence,
                    source_paragraph=rel.paragraph_id,
                )

    print(f"Added {G.number_of_edges()} edges to graph")

    return G


# ============================================================================
# Visualization
# ============================================================================


def visualize_with_pyvis(
    G: nx.DiGraph,
    normalized_entities: list[NormalizedEntity],
    normalized_relationships: list[NormalizedRelationship],
    output_file: str = "knowledge_graph.html",
    height: str = "900px",
    width: str = "100%",
):
    """
    Create interactive PyVis visualization.

    Args:
        G: NetworkX DiGraph
        normalized_entities: List of NormalizedEntity objects
        normalized_relationships: List of NormalizedRelationship objects
        output_file: HTML file to save
        height: Canvas height
        width: Canvas width
    """
    # Create PyVis network
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        bgcolor="#F8F9FA",
        font_color="#333333",
    )

    # Configure physics for better layout
    net.set_options(
        """
    {
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """
    )

    # Define colors for entity types
    type_colors = {
        "person": "#FF6B6B",
        "place": "#4ECDC4",
        "collective_entity": "#45B7D1",
        "event": "#FFA07A",
        "temporal": "#98D8C8",
        "cultural": "#C7CEEA",
    }

    # Create entity ID lookup
    entity_lookup = {e.id: e for e in normalized_entities}

    # Add nodes
    for entity in normalized_entities:
        color = type_colors.get(entity.type, "#CCCCCC")
        size = 15 + (entity.occurrence_count * 5)

        # Build title (hover text)
        title = f"<b>{entity.name}</b><br>"
        title += f"Type: {entity.type}"
        if entity.subtype:
            title += f" ({entity.subtype})"
        title += f"<br>Occurrences: {entity.occurrence_count}"
        title += f"<br>Relationships: {len(entity.relationship_ids)}"
        if entity.aliases:
            title += f"<br>Aliases: {', '.join(entity.aliases[:3])}"
        if entity.attributes:
            title += f"<br>Attributes: {entity.attributes}"

        net.add_node(
            entity.id, label=entity.name, title=title, color=color, size=size, font={"size": 14}
        )

    # Add edges with labels
    for rel in normalized_relationships:
        if rel.source_id in entity_lookup and rel.target_id in entity_lookup:
            # Create edge label and title
            label = rel.relation_type.replace("-", " ").replace("_", " ")
            title = f"{entity_lookup[rel.source_id].name} → {entity_lookup[rel.target_id].name}<br>"
            title += f"Relationship: {rel.relation_type}"
            if rel.temporal_context:
                title += f"<br>When: {rel.temporal_context}"
                label += f" ({rel.temporal_context})"

            net.add_edge(
                rel.source_id,
                rel.target_id,
                label=label,
                title=title,
                arrows="to",
                color={"color": "#888888", "highlight": "#333333"},
                width=2,
                font={"size": 10, "align": "middle"},
            )

    # Save
    net.save_graph(output_file)
    print(f"\n✅ Interactive graph saved to: {output_file}")
    print("\nFeatures:")
    print("  • Drag nodes to rearrange")
    print("  • Click node to highlight connections")
    print("  • Hover over nodes/edges for details")
    print("  • Scroll to zoom, drag canvas to pan")
    print("  • Use navigation buttons (bottom right)")

    return net


# ============================================================================
# Data Loading
# ============================================================================


def load_paragraphs_from_json(json_file: Path) -> list[dict]:
    """Load paragraphs from JSON file."""
    with open(json_file) as f:
        paragraphs = json.load(f)
    return paragraphs


def load_paragraphs_from_database(book_index: int, chapter_index: int) -> list[dict]:
    """Load paragraphs from Weaviate database."""
    config = WeaviateConfig.from_environment()
    manager = BookRepositoryManager(config)

    paragraphs = manager.paragraphs.find_by_chapter_index(
        book_index=book_index, chapter_index=chapter_index
    )

    return [
        {
            "id": p.id,
            "text": p.text,
            "page": p.page,
            "paragraph_index": p.paragraph_index,
            "book_index": p.book_index,
            "chapter_index": p.chapter_index,
        }
        for p in paragraphs
    ]


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities and relationships, build and visualize knowledge graph"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON file with paragraphs (must have 'id' and 'text' fields)",
    )
    parser.add_argument("--book", type=int, help="Book index (1-indexed)")
    parser.add_argument(
        "--chapter", type=int, help="Chapter index (0-indexed, 0=Introduction)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="knowledge_graph.html",
        help="Output HTML file for visualization",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results (extraction, normalization) to JSON",
    )

    args = parser.parse_args()

    # Validate input
    if args.input and (args.book or args.chapter):
        print("Error: Specify either --input OR --book/--chapter, not both")
        sys.exit(1)
    if not args.input and not (args.book and args.chapter):
        print("Error: Must specify either --input or both --book and --chapter")
        sys.exit(1)

    # Load paragraphs
    print("=" * 80)
    print("STEP 1: Loading paragraphs")
    print("=" * 80)

    if args.input:
        print(f"Loading from JSON: {args.input}")
        paragraphs = load_paragraphs_from_json(args.input)
    else:
        print(f"Loading from database: Book {args.book}, Chapter {args.chapter}")
        paragraphs = load_paragraphs_from_database(args.book, args.chapter)

    print(f"Loaded {len(paragraphs)} paragraphs\n")

    # Extract entities and relationships
    print("=" * 80)
    print("STEP 2: Extracting entities and relationships (GPT-4o)")
    print("=" * 80)

    all_results = []
    for i, para in enumerate(paragraphs, 1):
        print(f"Processing paragraph {i}/{len(paragraphs)}...", end=" ")
        result = extract_entities_gpt4o(para["text"], para["id"])
        all_results.append(result)
        print(f"({len(result.entities)} entities, {len(result.relationships)} rels)")

    total_entities = sum(len(r.entities) for r in all_results)
    total_relationships = sum(len(r.relationships) for r in all_results)
    print(f"\nTotal extracted: {total_entities} entities, {total_relationships} relationships")

    if args.save_intermediate:
        extraction_file = args.output.parent / f"{args.output.stem}_extraction.json"
        with open(extraction_file, "w") as f:
            json.dump(
                [
                    {
                        "paragraph_id": r.paragraph_id,
                        "entities": [e.model_dump() for e in r.entities],
                        "relationships": [rel.model_dump() for rel in r.relationships],
                    }
                    for r in all_results
                ],
                f,
                indent=2,
            )
        print(f"Saved extraction results to: {extraction_file}")

    # Assign IDs
    print("\n" + "=" * 80)
    print("STEP 3: Assigning IDs and building bidirectional links")
    print("=" * 80)

    entities_with_ids, relationships_with_ids = assign_ids_to_extraction_results(
        all_results
    )

    # Normalize entities and relationships
    print("\n" + "=" * 80)
    print("STEP 4: Normalizing entities and relationships")
    print("=" * 80)

    normalized_entities, normalized_relationships = (
        normalize_entities_and_relationships(entities_with_ids, relationships_with_ids)
    )

    print(f"\nReduction: {len(entities_with_ids)} → {len(normalized_entities)} entities")

    if args.save_intermediate:
        normalization_file = (
            args.output.parent / f"{args.output.stem}_normalized.json"
        )
        with open(normalization_file, "w") as f:
            json.dump(
                {
                    "entities": [e.model_dump() for e in normalized_entities],
                    "relationships": [r.model_dump() for r in normalized_relationships],
                },
                f,
                indent=2,
            )
        print(f"Saved normalized results to: {normalization_file}")

    # Build graph
    print("\n" + "=" * 80)
    print("STEP 5: Building knowledge graph")
    print("=" * 80)

    kg = build_knowledge_graph(normalized_entities, normalized_relationships)

    # Display statistics
    print("\nGraph statistics:")
    print(f"  Nodes: {kg.number_of_nodes()}")
    print(f"  Edges: {kg.number_of_edges()}")
    if kg.number_of_nodes() > 0:
        print(
            f"  Avg degree: {sum(dict(kg.degree()).values()) / kg.number_of_nodes():.2f}"
        )
        print(f"  Density: {nx.density(kg):.3f}")

    # Visualize
    print("\n" + "=" * 80)
    print("STEP 6: Creating interactive visualization")
    print("=" * 80)

    visualize_with_pyvis(
        kg,
        normalized_entities,
        normalized_relationships,
        output_file=str(args.output),
    )

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
