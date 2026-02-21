"""
Extract entities and relationships from historical text and visualize as a knowledge graph.

This script implements the full KG pipeline:
1. Entity extraction with GPT-4o (LangChain structured outputs)
2. ID assignment with bidirectional links
3. Entity normalization (rule-based or LLM-based)
   - Rule-based: exact + alias matching
   - LLM-based: semantic similarity + LLM merge decisions
4. Graph construction with NetworkX
5. Interactive visualization with PyVis

Usage:
    # Rule-based normalization (default)
    python extract_and_visualize_kg.py --input selected_5_paragraphs.json --output kg_test.html

    # LLM-based normalization
    python extract_and_visualize_kg.py --book 3 --chapter 4 --normalization-method llm-based --output rome_chapter_kg_llm.html

    # LLM-based with custom parameters
    python extract_and_visualize_kg.py --input data.json --normalization-method llm-based --similarity-threshold 0.7 --max-llm-candidates 50 --output kg_llm.html
"""

import argparse
import json
import sys
import uuid as uuid_module
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

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
    subtype: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """Relationship between two entities (LLM output)."""

    source_entity: str  # Entity name
    relation_type: str
    target_entity: str  # Entity name
    temporal_context: str | None = None
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
    subtype: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    paragraph_id: str
    relationship_ids: list[str] = Field(default_factory=list)  # Bidirectional link


class RelationshipWithId(BaseModel):
    """Relationship with UUIDs for entities."""

    id: str  # UUID
    source_id: str  # Entity UUID
    target_id: str  # Entity UUID
    relation_type: str
    temporal_context: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    paragraph_id: str


class NormalizedEntity(BaseModel):
    """Normalized entity after merging duplicates."""

    id: str  # Canonical UUID
    name: str
    type: str
    subtype: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str  # Combined descriptions
    attributes: dict[str, str] | None = None
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
    temporal_context: str | None = None
    confidence: float
    paragraph_id: str


# ============================================================================
# LLM-Based Normalization Models
# ============================================================================


class EntityMergeDecision(BaseModel):
    """LLM decision on whether two entities should be merged."""

    should_merge: bool = Field(
        description="True if entities refer to the same historical entity"
    )
    reasoning: str = Field(description="Brief explanation of the decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    merged_entity: EntityWithId | None = Field(
        default=None,
        description="The merged entity if should_merge=True, otherwise None",
    )


class RelationshipWithIdExtended(BaseModel):
    """Relationship with original entity names preserved (for LLM normalization)."""

    id: str  # UUID
    source_id: str  # Entity UUID
    target_id: str  # Entity UUID
    source_entity_name: str  # Original entity name from extraction
    target_entity_name: str  # Original entity name from extraction
    relation_type: str
    temporal_context: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    paragraph_id: str


class NormalizedRelationshipExtended(BaseModel):
    """Normalized relationship with original entity names preserved."""

    id: str  # UUID (preserved from original)
    source_id: str  # Normalized entity ID
    target_id: str  # Normalized entity ID
    source_entity_name: str  # Original entity name (NOT normalized name)
    target_entity_name: str  # Original entity name (NOT normalized name)
    relation_type: str
    temporal_context: str | None = None
    confidence: float
    paragraph_id: str


# ============================================================================
# Extraction
# ============================================================================

EXTRACTION_PROMPT = """You are analyzing text from "The Penguin History of the World".

Extract noteworthy entities and relationships from the provided paragraph.

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
2. Extract *noteworthy* entities (think: proper nouns, significant concepts)
3. Extract relationships that are EXPLICITLY STATED in the text
4. Be specific; you may infer details where they are obvious (e.g., "the capital" in the context of Byzantium -> Constantinople)
5. Include aliases if the entity is referred to by multiple names (e.g., "Octavian" also called "Augustus")
6. For titles/roles, store as attributes
7. DO NOT extract:
   - Generic unnamed groups ("his men", "the soldiers")
   - Entities mentioned only as comparisons ("like Athens")
   - Vague references without clear identity
8. Include temporal_context in relationships when dates/times are mentioned
9. Note: Some entities may have dual nature (e.g., Rome as both a city and a political state) - extract both if clear from context
10. Only extract entities if they appear in a relationship, and only extract relationships between extracted entities.

Extract entities and relationships from this paragraph:

{paragraph_text}
"""


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def extract_entities_gpt(
    paragraph_text: str, paragraph_id: str, model="gpt-4.1-mini-2025-04-14"
) -> ExtractionResult:
    """
    Extract entities and relationships using GPT-4o with LangChain structured outputs.

    Includes retry logic with exponential backoff for rate limit errors:
    - Retries up to 5 times
    - Waits 4s, 8s, 16s, 32s, 60s between retries
    - Re-raises exception if all retries fail
    """
    model = ChatOpenAI(model=model, temperature=0.0)
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
                        canonical = name if len(name) < len(other_name) else other_name
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

    for _name, entity_group in merged_groups.items():
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
# LLM-Based Normalization
# ============================================================================

ENTITY_MERGE_PROMPT = """You are an expert historian analyzing entity mentions from "The Penguin History of the World".

Given two entities extracted from different paragraphs, determine if they refer to the SAME historical entity.

**Entity 1:**
Name: {entity1_name}
Type: {entity1_type}
Subtype: {entity1_subtype}
Aliases: {entity1_aliases}
Description: {entity1_description}
Attributes: {entity1_attributes}
Confidence: {entity1_confidence}

**Entity 2:**
Name: {entity2_name}
Type: {entity2_type}
Subtype: {entity2_subtype}
Aliases: {entity2_aliases}
Description: {entity2_description}
Attributes: {entity2_attributes}
Confidence: {entity2_confidence}

**Instructions:**
1. Determine if these refer to the SAME historical entity (person, place, organization, etc.)
2. Consider:
   - Are the names referring to the same entity? (e.g., "Octavian" and "Augustus" are the same person)
   - Do the types/subtypes match or are compatible?
   - Do descriptions align or contradict?
   - Are attributes consistent?
3. If they should be merged:
   - Choose the most canonical/common name
   - Write a consolidated description (combine key information, ~2-3 sentences)
   - Merge aliases (include both original names if not already aliases)
   - Combine attributes (later values override if conflict)
   - Set confidence as average of both entities

**Decision Guidelines:**
- Different people with same last name (e.g., "Julius Caesar" vs "Augustus Caesar") → DO NOT merge
- Same person at different life stages (e.g., "Octavian" vs "Augustus") → MERGE
- Same place in different contexts (e.g., "Rome" the city vs "Rome" the empire) → DO NOT merge (different subtypes)
- Obvious typos or variations (e.g., "Byzanthium" vs "Byzantium") → MERGE
"""


def setup_llm_merge_chain():
    """Create LangChain chain for entity merge decisions."""
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_structure = llm.with_structured_output(EntityMergeDecision)

    prompt = ChatPromptTemplate.from_template(ENTITY_MERGE_PROMPT)
    merge_chain = prompt | llm_with_structure

    return merge_chain


def format_entity_for_prompt(entity: EntityWithId) -> dict:
    """Extract entity fields for prompt template."""
    return {
        "name": entity.name,
        "type": entity.type,
        "subtype": entity.subtype or "None",
        "aliases": ", ".join(entity.aliases) if entity.aliases else "None",
        "description": entity.description or "None",
        "attributes": str(entity.attributes) if entity.attributes else "None",
        "confidence": f"{entity.confidence:.2f}",
    }


def decide_entity_merge(
    entity1: EntityWithId, entity2: EntityWithId, chain
) -> EntityMergeDecision:
    """Run the merge decision chain on two entities."""
    # Format inputs
    e1_fields = format_entity_for_prompt(entity1)
    e2_fields = format_entity_for_prompt(entity2)

    # Prefix keys
    inputs = {}
    for key, val in e1_fields.items():
        inputs[f"entity1_{key}"] = val
    for key, val in e2_fields.items():
        inputs[f"entity2_{key}"] = val

    # Invoke chain
    result = chain.invoke(inputs)
    return result


def find_merge_groups(llm_results: list[dict]) -> list[set]:
    """
    Find groups of entities that should be merged together using connected components.

    Args:
        llm_results: List of dicts with 'entity1', 'entity2', 'llm_decision' keys

    Returns:
        List of sets, each set contains entity IDs that should be merged
    """
    # Build adjacency list of merge decisions
    merge_pairs = []
    for result in llm_results:
        if result["llm_decision"]:
            merge_pairs.append((result["entity1"].id, result["entity2"].id))

    # Find connected components using simple graph traversal
    entity_to_group = {}
    groups = []

    for e1_id, e2_id in merge_pairs:
        group1 = entity_to_group.get(e1_id)
        group2 = entity_to_group.get(e2_id)

        if group1 is None and group2 is None:
            # Create new group
            new_group = {e1_id, e2_id}
            groups.append(new_group)
            entity_to_group[e1_id] = new_group
            entity_to_group[e2_id] = new_group
        elif group1 is not None and group2 is None:
            # Add e2 to e1's group
            group1.add(e2_id)
            entity_to_group[e2_id] = group1
        elif group1 is None and group2 is not None:
            # Add e1 to e2's group
            group2.add(e1_id)
            entity_to_group[e1_id] = group2
        elif group1 is not group2:
            # Merge two groups
            group1.update(group2)
            for eid in group2:
                entity_to_group[eid] = group1
            groups.remove(group2)

    return groups


def reevaluate_large_groups(
    groups: list[set], entities_with_ids: list[EntityWithId], chain
) -> list[set]:
    """
    For groups with >2 entities, use LLM to confirm all pairwise merges.

    Args:
        groups: List of entity ID sets that should potentially be merged
        entities_with_ids: All entities with IDs
        chain: LLM merge chain

    Returns:
        Refined groups where all pairwise merges are confirmed
    """
    entity_id_to_obj = {e.id: e for e in entities_with_ids}
    refined_groups = []

    for group in groups:
        if len(group) <= 2:
            # Small groups are already validated by pairwise comparison
            refined_groups.append(group)
            continue

        print(f"  Re-evaluating group of {len(group)} entities for transitivity...")

        # Test all pairwise combinations in this group
        confirmed_pairs = []
        entity_ids = list(group)

        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                e1 = entity_id_to_obj[entity_ids[i]]
                e2 = entity_id_to_obj[entity_ids[j]]

                decision = decide_entity_merge(e1, e2, chain)

                if decision.should_merge:
                    confirmed_pairs.append((entity_ids[i], entity_ids[j]))

        # Find connected components from confirmed pairs
        sub_groups = find_merge_groups(
            [
                {
                    "entity1": entity_id_to_obj[e1],
                    "entity2": entity_id_to_obj[e2],
                    "llm_decision": True,
                }
                for e1, e2 in confirmed_pairs
            ]
        )

        if len(sub_groups) > 1:
            print(f"    ⚠️  Split into {len(sub_groups)} groups after re-evaluation")

        refined_groups.extend(sub_groups)

    return refined_groups


def normalize_entities_llm(
    entities_with_ids: list[EntityWithId],
    relationships_with_ids: list[RelationshipWithId],
    similarity_threshold: float = 0.6,
    max_candidates: int = 100,
) -> tuple[list[NormalizedEntity], list[NormalizedRelationshipExtended]]:
    """
    LLM-based entity normalization using semantic similarity and LLM merge decisions.

    Args:
        entities_with_ids: All entities with IDs
        relationships_with_ids: All relationships with IDs
        similarity_threshold: Minimum embedding similarity to consider for LLM evaluation
        max_candidates: Maximum number of entity pairs to evaluate with LLM

    Returns:
        (normalized_entities, normalized_relationships_extended)
    """
    import numpy as np
    from langchain_openai import OpenAIEmbeddings

    print(f"\nStarting LLM-based normalization with {len(entities_with_ids)} entities")

    # Step 1: Compute embeddings for entity names
    print("Computing embeddings for entity names...")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    entity_names = [e.name for e in entities_with_ids]
    embeddings = embeddings_model.embed_documents(entity_names)
    embeddings_array = np.array(embeddings)

    # Step 2: Find similarity candidates
    print(f"Finding similarity candidates (threshold: {similarity_threshold})...")
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(embeddings_array)
    candidates = []

    for i in range(len(entities_with_ids)):
        for j in range(i + 1, len(entities_with_ids)):
            sim = similarity_matrix[i][j]
            if sim >= similarity_threshold:
                candidates.append(
                    {
                        "entity1": entities_with_ids[i],
                        "entity2": entities_with_ids[j],
                        "similarity": sim,
                    }
                )

    # Sort by similarity (highest first)
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    candidates = candidates[:max_candidates]

    print(f"Found {len(candidates)} candidate pairs (showing top {max_candidates})")

    # Step 3: Run LLM merge decisions
    print("Running LLM merge decisions...")
    chain = setup_llm_merge_chain()
    llm_results = []

    for i, candidate in enumerate(candidates, 1):
        entity1 = candidate["entity1"]
        entity2 = candidate["entity2"]
        similarity = candidate["similarity"]

        print(
            f"  [{i}/{len(candidates)}] Evaluating: {entity1.name} <-> {entity2.name}"
        )

        decision = decide_entity_merge(entity1, entity2, chain)

        llm_results.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "embedding_similarity": similarity,
                "llm_decision": decision.should_merge,
                "llm_confidence": decision.confidence,
                "llm_reasoning": decision.reasoning,
                "merged_entity": decision.merged_entity,
            }
        )

        if decision.should_merge:
            print(f"    → MERGE (confidence: {decision.confidence:.2f})")
        else:
            print(f"    → SEPARATE (confidence: {decision.confidence:.2f})")

    # Step 4: Find merge groups (connected components)
    print("\nFinding merge groups...")
    merge_groups = find_merge_groups(llm_results)
    print(f"Found {len(merge_groups)} merge groups")

    # Step 5: Re-evaluate large groups for transitivity
    print("Re-evaluating large groups for transitivity...")
    refined_groups = reevaluate_large_groups(merge_groups, entities_with_ids, chain)
    print(f"Final merge groups: {len(refined_groups)}")

    # Step 6: Create normalized entities
    print("Creating normalized entities...")
    normalized_entities = []
    old_id_to_normalized_id = {}

    # Track which entities are in merge groups
    entities_in_groups = set()
    for group in refined_groups:
        entities_in_groups.update(group)

    # Process merge groups
    for group in refined_groups:
        canonical_id = str(uuid_module.uuid4())
        entity_ids = list(group)

        # Find LLM-generated merged entity (use first pairwise merge result involving these entities)
        merged_entity_obj = None
        for result in llm_results:
            if (
                result["llm_decision"]
                and result["merged_entity"]
                and (
                    result["entity1"].id in entity_ids
                    and result["entity2"].id in entity_ids
                )
            ):
                merged_entity_obj = result["merged_entity"]
                break

        # Get all entities in group
        entities_in_group = [e for e in entities_with_ids if e.id in entity_ids]

        # Aggregate data
        all_paragraphs = set()
        all_relationship_ids = set()
        avg_confidence = 0.0

        for entity in entities_in_group:
            all_paragraphs.add(entity.paragraph_id)
            all_relationship_ids.update(entity.relationship_ids)
            avg_confidence += entity.confidence
            old_id_to_normalized_id[entity.id] = canonical_id

        avg_confidence /= len(entities_in_group)

        # Use LLM-generated merged entity if available, otherwise use first entity
        if merged_entity_obj:
            normalized_entity = NormalizedEntity(
                id=canonical_id,
                name=merged_entity_obj.name,
                type=merged_entity_obj.type,
                subtype=merged_entity_obj.subtype,
                aliases=merged_entity_obj.aliases,
                description=merged_entity_obj.description or "",
                attributes=merged_entity_obj.attributes,
                source_paragraph_ids=list(all_paragraphs),
                occurrence_count=len(entities_in_group),
                merged_from_ids=entity_ids,
                relationship_ids=list(all_relationship_ids),
            )
        else:
            base_entity = entities_in_group[0]
            normalized_entity = NormalizedEntity(
                id=canonical_id,
                name=base_entity.name,
                type=base_entity.type,
                subtype=base_entity.subtype,
                aliases=base_entity.aliases,
                description=base_entity.description or "",
                attributes=base_entity.attributes,
                source_paragraph_ids=list(all_paragraphs),
                occurrence_count=len(entities_in_group),
                merged_from_ids=entity_ids,
                relationship_ids=list(all_relationship_ids),
            )

        normalized_entities.append(normalized_entity)

    # Add entities that weren't merged
    for entity in entities_with_ids:
        if entity.id not in entities_in_groups:
            canonical_id = str(uuid_module.uuid4())
            old_id_to_normalized_id[entity.id] = canonical_id

            normalized_entity = NormalizedEntity(
                id=canonical_id,
                name=entity.name,
                type=entity.type,
                subtype=entity.subtype,
                aliases=entity.aliases,
                description=entity.description or "",
                attributes=entity.attributes,
                source_paragraph_ids=[entity.paragraph_id],
                occurrence_count=1,
                merged_from_ids=[entity.id],
                relationship_ids=entity.relationship_ids,
            )
            normalized_entities.append(normalized_entity)

    print(f"Created {len(normalized_entities)} normalized entities")

    # Step 7: Create extended relationships with original entity names
    print("Creating normalized relationships with original entity names...")

    # Create entity lookup
    entity_id_to_obj = {e.id: e for e in entities_with_ids}
    normalized_entity_id_to_obj = {e.id: e for e in normalized_entities}

    # Reset relationship_ids on normalized entities
    for entity in normalized_entities:
        entity.relationship_ids = []

    normalized_relationships_extended = []

    for rel in relationships_with_ids:
        norm_source_id = old_id_to_normalized_id.get(rel.source_id)
        norm_target_id = old_id_to_normalized_id.get(rel.target_id)

        if norm_source_id and norm_target_id:
            # Get original entity names
            source_entity_name = entity_id_to_obj[rel.source_id].name
            target_entity_name = entity_id_to_obj[rel.target_id].name

            # Create normalized relationship with original names preserved
            norm_rel = NormalizedRelationshipExtended(
                id=rel.id,
                source_id=norm_source_id,
                target_id=norm_target_id,
                source_entity_name=source_entity_name,
                target_entity_name=target_entity_name,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                confidence=rel.confidence,
                paragraph_id=rel.paragraph_id,
            )
            normalized_relationships_extended.append(norm_rel)

            # Rebuild bidirectional links
            if norm_source_id in normalized_entity_id_to_obj:
                normalized_entity_id_to_obj[norm_source_id].relationship_ids.append(
                    rel.id
                )
            if norm_target_id in normalized_entity_id_to_obj:
                normalized_entity_id_to_obj[norm_target_id].relationship_ids.append(
                    rel.id
                )

    print(f"Normalized {len(normalized_relationships_extended)} relationships")
    skipped = len(relationships_with_ids) - len(normalized_relationships_extended)
    if skipped > 0:
        print(f"Skipped {skipped} relationships (entities not found)")

    return normalized_entities, normalized_relationships_extended


# ============================================================================
# Graph Construction
# ============================================================================


def build_knowledge_graph(
    normalized_entities: list[NormalizedEntity],
    normalized_relationships: list,  # Can be NormalizedRelationship or NormalizedRelationshipExtended
) -> nx.DiGraph:
    """
    Build a directed graph from normalized entities and relationships.

    Args:
        normalized_entities: List of NormalizedEntity objects
        normalized_relationships: List of NormalizedRelationship or NormalizedRelationshipExtended objects

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
    normalized_relationships: list,  # Can be NormalizedRelationship or NormalizedRelationshipExtended
    output_file: str = "knowledge_graph.html",
    height: str = "900px",
    width: str = "100%",
):
    """
    Create interactive PyVis visualization.

    Args:
        G: NetworkX DiGraph
        normalized_entities: List of NormalizedEntity objects
        normalized_relationships: List of NormalizedRelationship or NormalizedRelationshipExtended objects
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
            entity.id,
            label=entity.name,
            title=title,
            color=color,
            size=size,
            font={"size": 14},
        )

    # Add edges with labels
    for rel in normalized_relationships:
        if rel.source_id in entity_lookup and rel.target_id in entity_lookup:
            # Check if this is NormalizedRelationshipExtended (has original entity names)
            has_original_names = hasattr(rel, "source_entity_name") and hasattr(
                rel, "target_entity_name"
            )

            if has_original_names:
                # Use original entity names for display (LLM normalization)
                source_name = rel.source_entity_name
                target_name = rel.target_entity_name
                label = f"{source_name} → {target_name}"
                label += f"\n{rel.relation_type.replace('-', ' ').replace('_', ' ')}"

                # Title shows both original and normalized names
                title = f"{source_name} → {target_name}<br>"
                title += f"Relationship: {rel.relation_type}<br>"
                title += f"Normalized: {entity_lookup[rel.source_id].name} → {entity_lookup[rel.target_id].name}"
            else:
                # Use normalized entity names (rule-based normalization)
                label = rel.relation_type.replace("-", " ").replace("_", " ")
                title = f"{entity_lookup[rel.source_id].name} → {entity_lookup[rel.target_id].name}<br>"
                title += f"Relationship: {rel.relation_type}"

            if rel.temporal_context:
                title += f"<br>When: {rel.temporal_context}"
                if not has_original_names:
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
    parser.add_argument(
        "--normalization-method",
        choices=["rule-based", "llm-based"],
        default="rule-based",
        help="Entity normalization method: rule-based (exact + alias matching) or llm-based (semantic similarity + LLM decisions)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Embedding similarity threshold for LLM normalization (default: 0.6)",
    )
    parser.add_argument(
        "--max-llm-candidates",
        type=int,
        default=100,
        help="Maximum number of entity pairs to evaluate with LLM (default: 100)",
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
    print("STEP 2: Extracting entities and relationships (GPT-5-mini)")
    print("=" * 80)

    all_results = []
    for i, para in enumerate(paragraphs, 1):
        print(f"Processing paragraph {i}/{len(paragraphs)}...", end=" ")
        result = extract_entities_gpt(para["text"], para["id"])
        all_results.append(result)
        print(f"({len(result.entities)} entities, {len(result.relationships)} rels)")

    total_entities = sum(len(r.entities) for r in all_results)
    total_relationships = sum(len(r.relationships) for r in all_results)
    print(
        f"\nTotal extracted: {total_entities} entities, {total_relationships} relationships"
    )

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
    print(
        f"STEP 4: Normalizing entities and relationships ({args.normalization_method})"
    )
    print("=" * 80)

    if args.normalization_method == "llm-based":
        normalized_entities, normalized_relationships = normalize_entities_llm(
            entities_with_ids,
            relationships_with_ids,
            similarity_threshold=args.similarity_threshold,
            max_candidates=args.max_llm_candidates,
        )
    else:
        normalized_entities, normalized_relationships = (
            normalize_entities_and_relationships(
                entities_with_ids, relationships_with_ids
            )
        )

    print(
        f"\nReduction: {len(entities_with_ids)} → {len(normalized_entities)} entities"
    )

    if args.save_intermediate:
        normalization_file = args.output.parent / f"{args.output.stem}_normalized.json"
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
