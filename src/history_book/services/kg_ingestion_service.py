"""KG ingestion service — extraction, merge, and Weaviate storage pipeline.

Processes paragraphs from the database, extracts entities and relationships,
normalizes via rule-based + embedding + LLM merging, and stores in Weaviate.
DB is the sole storage layer — no file I/O.
"""

import logging
import time
import uuid as uuid_module
from collections import defaultdict
from datetime import UTC, datetime

import networkx as nx
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity

from history_book.chains.kg_extraction_chain import (
    ExtractionResult,
    create_extraction_chain,
)
from history_book.chains.kg_merge_chain import (
    create_merge_chain,
    create_rule_filter_chain,
)
from history_book.chains.kg_temporal_chain import create_temporal_chain
from history_book.data_models.kg_entities import (
    KGEntity,
    KGGraph,
    KGMergeDecision,
    KGRelationship,
)
from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories.book_repository import BookRepositoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "extraction_model": "gpt-4.1-mini",
    "extraction_temperature": 0.0,
    "merge_model": "gpt-5-mini",
    "merge_temperature": 0.0,
    "rule_filter_model": "gpt-4.1-mini",
    "rule_filter_temperature": 0.0,
    "temporal_model": "gpt-4.1-nano",
    "temporal_temperature": 0.0,
    "embedding_model": "text-embedding-3-small",
    "similarity_threshold": 0.65,
    "max_llm_candidates": 100,
    "reasoning_effort": "minimal",
}

# ---------------------------------------------------------------------------
# Pipeline-internal models
# ---------------------------------------------------------------------------


class SourceDescription(BaseModel):
    text: str
    paragraph_id: str


class NormalizedEntity(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    descriptions: list[SourceDescription] = Field(default_factory=list)
    source_paragraph_ids: list[str]
    book_indices: list[int] = Field(default_factory=list)
    source_book_chapters: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    occurrence_count: int
    merged_from_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    source_graph: str = ""  # graph this entity was loaded from; "" for freshly-extracted entities


class NormalizedRelationship(BaseModel):
    id: str
    source_id: str
    target_id: str
    source_entity_name: str
    target_entity_name: str
    relation_type: str
    description: str | None = None
    temporal_context: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    temporal_precision: str | None = None
    paragraph_id: str
    book_index: int | None = None
    chapter_index: int | None = None
    page: int | None = None


class UnionFind:
    """Union-Find structure for tracking LLM entity merges."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.representative: dict[str, NormalizedEntity] = {}

    def add(self, entity: NormalizedEntity) -> None:
        self.parent[entity.id] = entity.id
        self.representative[entity.id] = entity

    def find(self, entity_id: str) -> str:
        while self.parent[entity_id] != entity_id:
            self.parent[entity_id] = self.parent[self.parent[entity_id]]
            entity_id = self.parent[entity_id]
        return entity_id

    def union(self, id1: str, id2: str, canonical_name: str | None) -> None:
        root1, root2 = self.find(id1), self.find(id2)
        if root1 == root2:
            return
        self.parent[root2] = root1
        rep, other = self.representative[root1], self.representative[root2]
        _apply_merge_identity(
            rep, other.name, other.aliases, other.descriptions, canonical_name
        )
        rep.source_paragraph_ids = list(
            set(rep.source_paragraph_ids + other.source_paragraph_ids)
        )
        rep.book_indices = sorted(set(rep.book_indices + other.book_indices))
        rep.source_book_chapters = sorted(
            set(rep.source_book_chapters + other.source_book_chapters)
        )
        rep.source_pages = sorted(set(rep.source_pages + other.source_pages))
        rep.occurrence_count += other.occurrence_count
        rep.merged_from_ids = list(set(rep.merged_from_ids + other.merged_from_ids))
        rep.relationship_ids = list(set(rep.relationship_ids + other.relationship_ids))

    def finalize(
        self,
        master_entities: list[NormalizedEntity],
        master_relationships: list[NormalizedRelationship],
    ) -> tuple[list[NormalizedEntity], list[NormalizedRelationship], dict[str, str]]:
        """Apply merges and return (final_entities, final_relationships, id_mapping)."""
        # Ensure all master entities are in the UF
        for e in master_entities:
            if e.id not in self.parent:
                self.add(e)

        groups: dict[str, list[str]] = defaultdict(list)
        for eid in self.parent:
            root = self.find(eid)
            groups[root].append(eid)

        final_entities: list[NormalizedEntity] = []
        master_id_to_final_id: dict[str, str] = {}

        for root_id, member_ids in groups.items():
            rep = self.representative[root_id]
            final_id = str(uuid_module.uuid4())
            final_entity = rep.model_copy(
                deep=True, update={"id": final_id, "relationship_ids": []}
            )
            final_entities.append(final_entity)
            for mid in member_ids:
                master_id_to_final_id[mid] = final_id

        final_relationships: list[NormalizedRelationship] = []
        final_entity_lookup = {e.id: e for e in final_entities}

        for rel in master_relationships:
            final_source = master_id_to_final_id.get(rel.source_id)
            final_target = master_id_to_final_id.get(rel.target_id)
            if final_source and final_target:
                final_rel = rel.model_copy(
                    update={"source_id": final_source, "target_id": final_target}
                )
                final_relationships.append(final_rel)
                if final_source in final_entity_lookup:
                    final_entity_lookup[final_source].relationship_ids.append(rel.id)
                if final_target in final_entity_lookup:
                    final_entity_lookup[final_target].relationship_ids.append(rel.id)

        return final_entities, final_relationships, master_id_to_final_id


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

TYPE_COLORS = {
    "person": "#FF6B6B",
    "polity": "#45B7D1",
    "place": "#4ECDC4",
    "event": "#FFA07A",
    "concept": "#59A14F",
}


def assign_ids_single(
    result: ExtractionResult,
    paragraph_meta: dict | None = None,
) -> tuple[list[NormalizedEntity], list[NormalizedRelationship]]:
    """Assign UUIDs to entities and relationships from a single extraction.
    Drops orphaned entities (not referenced by any relationship)."""
    para_entities: dict[str, NormalizedEntity] = {}
    relationships: list[NormalizedRelationship] = []
    skipped = 0

    # Build aggregate location fields from paragraph metadata
    book_indices: list[int] = []
    source_book_chapters: list[str] = []
    source_pages: list[int] = []
    if paragraph_meta:
        bi = paragraph_meta.get("book_index")
        ci = paragraph_meta.get("chapter_index")
        page = paragraph_meta.get("page")
        if bi is not None:
            book_indices = [bi]
        if bi is not None and ci is not None:
            source_book_chapters = [f"{bi}:{ci}"]
        if page is not None:
            source_pages = [page]

    for entity in result.entities:
        entity_id = str(uuid_module.uuid4())
        para_entities[entity.name] = NormalizedEntity(
            id=entity_id,
            name=entity.name,
            type=entity.type,
            aliases=entity.aliases,
            descriptions=[
                SourceDescription(
                    text=entity.description, paragraph_id=result.paragraph_id
                )
            ]
            if entity.description
            else [],
            source_paragraph_ids=[result.paragraph_id],
            book_indices=list(book_indices),
            source_book_chapters=list(source_book_chapters),
            source_pages=list(source_pages),
            occurrence_count=1,
            merged_from_ids=[entity_id],
            relationship_ids=[],
        )

    for rel in result.relationships:
        source = para_entities.get(rel.source_entity)
        target = para_entities.get(rel.target_entity)
        if source and target:
            rel_id = str(uuid_module.uuid4())
            rel_with_id = NormalizedRelationship(
                id=rel_id,
                source_id=source.id,
                target_id=target.id,
                source_entity_name=rel.source_entity,
                target_entity_name=rel.target_entity,
                relation_type=rel.relation_type,
                description=rel.description,
                temporal_context=rel.temporal_context,
                start_year=rel.start_year,
                end_year=rel.end_year,
                temporal_precision=rel.temporal_precision,
                paragraph_id=result.paragraph_id,
                book_index=paragraph_meta.get("book_index") if paragraph_meta else None,
                chapter_index=paragraph_meta.get("chapter_index")
                if paragraph_meta
                else None,
                page=paragraph_meta.get("page") if paragraph_meta else None,
            )
            relationships.append(rel_with_id)
            source.relationship_ids.append(rel_id)
            target.relationship_ids.append(rel_id)
        else:
            skipped += 1

    all_entities = [e for e in para_entities.values() if e.relationship_ids]
    orphaned = [e for e in para_entities.values() if not e.relationship_ids]

    if skipped:
        logger.debug("    Skipped %d rels (entity not found)", skipped)
    if orphaned:
        logger.debug(
            "    Dropped %d orphans: %s",
            len(orphaned),
            ", ".join(e.name for e in orphaned),
        )

    return all_entities, relationships


def create_entity_text(entity: NormalizedEntity) -> str:
    """Create text representation of entity for embedding comparisons."""
    parts = [f"Name: {entity.name}", f"Type: {entity.type}"]
    if entity.descriptions:
        parts.append(f"Description: {' | '.join(d.text for d in entity.descriptions)}")
    if entity.aliases:
        parts.append(f"Aliases: {', '.join(entity.aliases)}")
    return " | ".join(parts)


def merge_rule_based(
    new_entities: list[NormalizedEntity],
    new_relationships: list[NormalizedRelationship],
    master_entities: list[NormalizedEntity],
    master_relationships: list[NormalizedRelationship],
    rule_filter_chain=None,
    max_concurrency: int = 5,
    remap_relationship_ids: bool = False,
) -> tuple[
    list[NormalizedEntity], list[NormalizedRelationship], list[NormalizedEntity], list[dict]
]:
    """Merge new entities into master graph using exact name + alias matching.

    If rule_filter_chain is provided, rule-matched pairs are verified by a lightweight
    LLM call before merging (rejects clearly distinct entities sharing a name/alias).

    Args:
        remap_relationship_ids: If True, mint new UUIDs for relationships (use for
            cross-chapter merge to avoid ID collisions).

    Returns (updated_master_entities, updated_master_relationships, newly_added_entities, rule_decisions).
    """
    # Build one-to-many index: each name/alias key maps to all master entities that use it.
    name_to_masters: dict[str, list[NormalizedEntity]] = {}
    for me in master_entities:
        for key in [me.name.lower().strip()] + [
            a.lower().strip() for a in me.aliases if a.strip()
        ]:
            name_to_masters.setdefault(key, []).append(me)

    # --- Phase 1: Collect matches ---
    # For each new entity, gather all unique master candidates that share a name or alias,
    # filtered to same type and sorted by occurrence_count descending (most established first).
    pending_merges: list[tuple[NormalizedEntity, NormalizedEntity]] = []
    unmatched: list[NormalizedEntity] = []

    for entity in new_entities:
        seen_ids: set[str] = set()
        candidates: list[NormalizedEntity] = []
        for key in [entity.name.lower().strip()] + [
            a.lower().strip() for a in entity.aliases if a.strip()
        ]:
            for candidate in name_to_masters.get(key, []):
                if candidate.id not in seen_ids:
                    seen_ids.add(candidate.id)
                    if candidate.type == entity.type:
                        candidates.append(candidate)
                    else:
                        logger.debug(
                            "    Type mismatch: '%s' (%s) vs '%s' (%s) — skipping",
                            entity.name,
                            entity.type,
                            candidate.name,
                            candidate.type,
                        )
        candidates.sort(key=lambda e: e.occurrence_count, reverse=True)

        if candidates:
            for candidate in candidates:
                pending_merges.append((entity, candidate))
        else:
            unmatched.append(entity)

    # --- Phase 2: LLM filter ---
    n_filtered = 0
    approved_merges: list[tuple[NormalizedEntity, NormalizedEntity, str | None]] = []
    if rule_filter_chain and pending_merges:
        inputs = [_build_merge_inputs(e, m) for e, m in pending_merges]
        decisions = _batch_with_retry(rule_filter_chain, inputs, max_concurrency)

        all_approved: list[tuple[NormalizedEntity, NormalizedEntity, str | None]] = []
        rejected_entity_ids: set[str] = set()
        entity_by_id: dict[str, NormalizedEntity] = {e.id: e for e, _ in pending_merges}

        for (entity, match), decision in zip(pending_merges, decisions, strict=True):
            if decision.should_merge:
                all_approved.append((entity, match, decision.canonical_name))
            else:
                n_filtered += 1
                logger.info(
                    "  LLM filter rejected: '%s' (%s) != '%s' (%s)",
                    entity.name,
                    entity.type,
                    match.name,
                    match.type,
                )
                rejected_entity_ids.add(entity.id)

        # Take the first accepted candidate per entity (list is sorted by occ_count desc).
        seen_accepted: set[str] = set()
        for entity, match, canonical_name in all_approved:
            if entity.id not in seen_accepted:
                seen_accepted.add(entity.id)
                approved_merges.append((entity, match, canonical_name))

        # Only treat as unmatched if no candidate was accepted.
        for entity_id in rejected_entity_ids - seen_accepted:
            unmatched.append(entity_by_id[entity_id])
    else:
        # No filter: take the first (highest occ_count) candidate per entity.
        seen_accepted = set()
        for e, m in pending_merges:
            if e.id not in seen_accepted:
                seen_accepted.add(e.id)
                approved_merges.append((e, m, None))

    # --- Phase 3: Apply merges ---
    old_id_to_master_id: dict[str, str] = {}
    newly_added: list[NormalizedEntity] = []
    rule_decisions: list[dict] = []

    for entity, match, canonical_name in approved_merges:
        logger.debug("    Rule merge: '%s' -> '%s'", entity.name, match.name)
        # Capture pre-merge names before _apply_merge_identity renames match in-place
        partial = {
            "merge_type": "rule",
            "entity1_name": entity.name,
            "entity1_type": entity.type,
            "entity1_aliases": list(entity.aliases),
            "entity1_source_graph": entity.source_graph,
            "entity2_name": match.name,
            "entity2_type": match.type,
            "entity2_aliases": list(match.aliases),
            "canonical_name": canonical_name or "",
            "similarity": None,
            "reasoning": "",
        }
        _apply_merge_identity(
            match, entity.name, entity.aliases, entity.descriptions, canonical_name
        )
        match.source_paragraph_ids = list(
            set(match.source_paragraph_ids + entity.source_paragraph_ids)
        )
        match.book_indices = sorted(set(match.book_indices + entity.book_indices))
        match.source_book_chapters = sorted(
            set(match.source_book_chapters + entity.source_book_chapters)
        )
        match.source_pages = sorted(set(match.source_pages + entity.source_pages))
        match.occurrence_count += entity.occurrence_count
        match.merged_from_ids = list(
            set(match.merged_from_ids + entity.merged_from_ids)
        )
        old_id_to_master_id[entity.id] = match.id
        partial["occurrence_count_after"] = match.occurrence_count
        rule_decisions.append(partial)

    for entity in unmatched:
        new_master = entity.model_copy(
            deep=True, update={"id": str(uuid_module.uuid4()), "relationship_ids": []}
        )
        master_entities.append(new_master)
        newly_added.append(new_master)
        old_id_to_master_id[entity.id] = new_master.id
        rule_decisions.append({
            "merge_type": "root",
            "entity1_name": new_master.name,
            "entity1_type": new_master.type,
            "entity1_aliases": list(new_master.aliases),
            "entity1_source_graph": new_master.source_graph,
            "entity2_name": new_master.name,
            "entity2_type": new_master.type,
            "entity2_aliases": [],
            "canonical_name": new_master.name,
            "occurrence_count_after": new_master.occurrence_count,
            "similarity": None,
            "reasoning": "",
        })

    # --- Remap relationships ---
    master_entity_lookup = {e.id: e for e in master_entities}
    for rel in new_relationships:
        new_source = old_id_to_master_id.get(rel.source_id)
        new_target = old_id_to_master_id.get(rel.target_id)
        if new_source and new_target:
            updates = {"source_id": new_source, "target_id": new_target}
            if remap_relationship_ids:
                updates["id"] = str(uuid_module.uuid4())
            new_rel = rel.model_copy(update=updates)
            master_relationships.append(new_rel)
            if new_source in master_entity_lookup:
                master_entity_lookup[new_source].relationship_ids.append(new_rel.id)
            if new_target in master_entity_lookup:
                master_entity_lookup[new_target].relationship_ids.append(new_rel.id)

    n_merged = len(approved_merges)
    if n_merged or n_filtered:
        logger.debug(
            "    rule: %d matched, %d filtered, %d merged, %d new",
            n_merged + n_filtered,
            n_filtered,
            n_merged,
            len(newly_added),
        )

    return master_entities, master_relationships, newly_added, rule_decisions


def find_candidates_against_master(
    new_entities: list[NormalizedEntity],
    new_embeddings: np.ndarray,
    master_entities: list[NormalizedEntity],
    master_embeddings: np.ndarray,
    threshold: float,
) -> list[dict]:
    """Find merge candidates: new entities vs existing master entities."""
    if len(new_embeddings) == 0 or len(master_embeddings) == 0:
        return []

    sim_matrix = cosine_similarity(new_embeddings, master_embeddings)

    candidates = []
    for i in range(len(new_entities)):
        for j in range(len(master_entities)):
            sim = sim_matrix[i, j]
            if sim >= threshold and new_entities[i].type == master_entities[j].type:
                candidates.append(
                    {
                        "new_entity": new_entities[i],
                        "master_entity": master_entities[j],
                        "similarity": float(sim),
                    }
                )

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates


def format_entity_for_prompt(entity: NormalizedEntity) -> dict:
    """Format an entity for the merge prompt."""
    return {
        "name": entity.name,
        "type": entity.type,
        "aliases": ", ".join(entity.aliases) if entity.aliases else "None",
        "description": " | ".join(d.text for d in entity.descriptions)
        if entity.descriptions
        else "None",
    }


def _apply_merge_identity(
    master: NormalizedEntity,
    other_name: str,
    other_aliases: list[str],
    other_descriptions: list[SourceDescription],
    canonical_name: str | None = None,
) -> None:
    """Apply identity merge into master entity (in-place).

    Sets canonical name, unions aliases (excluding canonical), extends descriptions.
    """
    if canonical_name:
        master.name = canonical_name
    # Union all aliases: both entity names + all aliases, excluding canonical
    canonical = master.name
    all_names = {master.name, other_name} | set(master.aliases) | set(other_aliases)
    master.aliases = [
        a for a in all_names if a.lower().strip() != canonical.lower().strip()
    ]
    # Extend descriptions
    master.descriptions.extend(other_descriptions)


def _build_merge_inputs(entity1, entity2) -> dict:
    """Build prompt inputs for a merge decision between two entities."""
    e1 = format_entity_for_prompt(entity1)
    e2 = format_entity_for_prompt(entity2)
    inputs = {}
    for key, val in e1.items():
        inputs[f"entity1_{key}"] = val
    for key, val in e2.items():
        inputs[f"entity2_{key}"] = val
    return inputs


def _batch_with_retry(chain, inputs_list: list, max_concurrency: int) -> list:
    """Run chain.batch() with exponential backoff on rate limit / timeout errors."""
    max_attempts = 5
    retryable_keywords = ["rate", "timeout", "timed out", "connection", "could not parse"]
    for attempt in range(max_attempts):
        try:
            return chain.batch(inputs_list, config={"max_concurrency": max_concurrency})
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in retryable_keywords)
            if is_retryable and attempt < max_attempts - 1:
                wait = min(4 * (2**attempt), 60)
                logger.warning(
                    "Batch failed (attempt %d/%d), retrying in %ds... (%s)",
                    attempt + 1,
                    max_attempts,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                raise


def _log_profile(timings: dict[str, list[float]], stages: list[str]) -> None:
    """Log a profiling summary table for the given stages."""
    logger.info("")
    logger.info("%-16s %10s %8s %8s", "Stage", "Total(s)", "Calls", "Avg(s)")
    logger.info("-" * 46)
    total_time = 0.0
    for stage in stages:
        vals = timings[stage]
        stage_total = sum(vals)
        total_time += stage_total
        if vals:
            logger.info(
                "%-16s %10.1f %8d %8.2f",
                stage,
                stage_total,
                len(vals),
                stage_total / len(vals),
            )
        else:
            logger.info("%-16s %10.1f %8d %8s", stage, 0.0, 0, "-")
    logger.info("-" * 46)
    logger.info("%-16s %10.1f", "total", total_time)


def build_knowledge_graph(
    entities: list[NormalizedEntity],
    relationships: list[NormalizedRelationship],
) -> nx.DiGraph:
    """Build a directed graph from normalized entities and relationships."""
    G = nx.DiGraph()

    for entity in entities:
        G.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.type,
            aliases=entity.aliases,
            descriptions=entity.descriptions,
            occurrence_count=entity.occurrence_count,
            merged_from_ids=entity.merged_from_ids,
            paragraph_ids=entity.source_paragraph_ids,
        )

    for rel in relationships:
        if rel.source_id in G and rel.target_id in G:
            if G.has_edge(rel.source_id, rel.target_id):
                edge_data = G[rel.source_id][rel.target_id]
                if rel.relation_type not in edge_data.get("relation_types", []):
                    edge_data.setdefault("relation_types", []).append(rel.relation_type)
                edge_data.setdefault("original_name_pairs", []).append(
                    (rel.source_entity_name, rel.target_entity_name)
                )
            else:
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    relation_types=[rel.relation_type],
                    temporal_context=rel.temporal_context,
                    source_paragraph=rel.paragraph_id,
                    source_entity_name=rel.source_entity_name,
                    target_entity_name=rel.target_entity_name,
                    original_name_pairs=[
                        (rel.source_entity_name, rel.target_entity_name)
                    ],
                )

    return G


def visualize_with_pyvis(
    G: nx.DiGraph,
    entities: list[NormalizedEntity],
    relationships: list[NormalizedRelationship],
    output_file: str,
) -> None:
    """Create interactive PyVis visualization."""
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#F8F9FA",
        font_color="#333333",
    )
    net.set_options("""
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
            "hover": true, "tooltipDelay": 100,
            "navigationButtons": true, "keyboard": true
        }
    }
    """)

    entity_lookup = {e.id: e for e in entities}

    for entity in entities:
        color = TYPE_COLORS.get(entity.type, "#CCCCCC")
        size = 15 + (entity.occurrence_count * 5)

        title = f"<b>{entity.name}</b><br>"
        title += f"Type: {entity.type}"
        title += f"<br>Occurrences: {entity.occurrence_count}"
        title += f"<br>Relationships: {len(entity.relationship_ids)}"
        if entity.aliases:
            title += f"<br>Aliases: {', '.join(entity.aliases[:5])}"
        if entity.descriptions:
            desc = " | ".join(d.text for d in entity.descriptions)
            desc = desc[:150] + "..." if len(desc) > 150 else desc
            title += f"<br><br>{desc}"
        if len(entity.merged_from_ids) > 1:
            title += f"<br><br>Merged from {len(entity.merged_from_ids)} entities"

        net.add_node(
            entity.id,
            label=entity.name,
            title=title,
            color=color,
            size=size,
            font={"size": 14},
            borderWidth=2,
        )

    for rel in relationships:
        if rel.source_id in entity_lookup and rel.target_id in entity_lookup:
            label = f"{rel.source_entity_name} \u2192 {rel.target_entity_name}"
            label += f"\n{rel.relation_type.replace('-', ' ').replace('_', ' ')}"

            title = f"{rel.source_entity_name} \u2192 {rel.target_entity_name}<br>"
            title += f"Relationship: {rel.relation_type}<br>"
            title += f"Normalized: {entity_lookup[rel.source_id].name} \u2192 {entity_lookup[rel.target_id].name}"
            if rel.description:
                title += f"<br>{rel.description}"
            if rel.temporal_context:
                title += f"<br>When: {rel.temporal_context}"

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

    net.save_graph(output_file)
    logger.info("Saved visualization to %s", output_file)


# ---------------------------------------------------------------------------
# Shared embedding merge helper
# ---------------------------------------------------------------------------


def _run_embedding_merge(
    newly_added: list[NormalizedEntity],
    uf: UnionFind,
    embeddings_model: OpenAIEmbeddings,
    merge_chain,
    master_entities: list[NormalizedEntity],
    master_embeddings: np.ndarray | None,
    master_entity_order: list[str],
    config: dict,
    timings: dict[str, list[float]],
    max_concurrency: int,
    result_context: dict,
) -> tuple[np.ndarray, list[dict], int, int, int]:
    """Embed newly added entities, find candidates vs master, run LLM merge.

    Returns (new_embeddings, llm_results, n_candidates, n_checked, n_merged).
    """
    threshold = config["similarity_threshold"]

    for ne in newly_added:
        if ne.id not in uf.parent:
            uf.add(ne)

    t0 = time.perf_counter()
    new_texts = [create_entity_text(e) for e in newly_added]
    new_embs = np.array(embeddings_model.embed_documents(new_texts))
    timings["embedding"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    candidates = []
    if master_embeddings is not None and len(master_entity_order) > 0:
        master_lookup = {e.id: e for e in master_entities}
        existing_entities = [
            master_lookup[eid] for eid in master_entity_order if eid in master_lookup
        ]
        existing_embs = master_embeddings[: len(existing_entities)]
        candidates = find_candidates_against_master(
            newly_added, new_embs, existing_entities, existing_embs, threshold
        )
    timings["similarity"].append(time.perf_counter() - t0)

    n_candidates = len(candidates)
    n_llm_checked = 0
    n_llm_merged = 0
    llm_results: list[dict] = []

    t0 = time.perf_counter()
    trimmed = candidates[: config["max_llm_candidates"]]
    pairs_to_check = []
    for c in trimmed:
        ne, me = c["new_entity"], c["master_entity"]
        root_new = uf.find(ne.id)
        if me.id not in uf.parent:
            uf.add(me)
        root_master = uf.find(me.id)
        if root_new != root_master:
            rep_new = uf.representative[root_new]
            rep_master = uf.representative[root_master]
            pairs_to_check.append((c, rep_new, rep_master))

    if pairs_to_check:
        logger.info("  Batch merging %d candidates...", len(pairs_to_check))
        inputs_list = [
            _build_merge_inputs(rep_new, rep_master)
            for _, rep_new, rep_master in pairs_to_check
        ]
        decisions = _batch_with_retry(merge_chain, inputs_list, max_concurrency)
        n_llm_checked = len(decisions)

        for (c, rep_new, rep_master), decision in zip(
            pairs_to_check, decisions, strict=False
        ):
            if decision.should_merge:
                uf.union(
                    c["new_entity"].id,
                    c["master_entity"].id,
                    decision.canonical_name,
                )
                new_root = uf.find(c["master_entity"].id)
                rep_after = uf.representative[new_root]
                llm_results.append(
                    {
                        "merge_type": "llm",
                        "entity1_name": rep_new.name,
                        "entity1_type": rep_new.type,
                        "entity1_aliases": list(rep_new.aliases),
                        "entity1_source_graph": c["new_entity"].source_graph,
                        "entity2_name": rep_master.name,
                        "entity2_type": rep_master.type,
                        "entity2_aliases": list(rep_master.aliases),
                        "similarity": c["similarity"],
                        "reasoning": decision.reasoning or "",
                        "canonical_name": decision.canonical_name or "",
                        "occurrence_count_after": rep_after.occurrence_count,
                    }
                )
                n_llm_merged += 1
                logger.info(
                    "  ** LLM MERGE: %s + %s (cos:%.3f)",
                    rep_new.name,
                    rep_master.name,
                    c["similarity"],
                )

    timings["llm_merge"].append(time.perf_counter() - t0)

    return new_embs, llm_results, n_candidates, n_llm_checked, n_llm_merged


# ---------------------------------------------------------------------------
# Pipeline: single-chapter extraction
# ---------------------------------------------------------------------------


def run_pipeline(
    paragraphs: list[dict],
    config: dict,
    *,
    profile: bool = False,
    max_concurrency: int = 5,
) -> tuple[list[NormalizedEntity], list[NormalizedRelationship], list[dict]]:
    """Run the full incremental KG extraction pipeline.

    Returns (final_entities, final_relationships, llm_merge_results).
    """
    extraction_chain = create_extraction_chain(config)
    embeddings_model = OpenAIEmbeddings(model=config["embedding_model"])
    merge_chain = create_merge_chain(config)
    rule_filter_chain = (
        create_rule_filter_chain(config) if config.get("rule_filter_model") else None
    )
    threshold = config["similarity_threshold"]

    master_entities: list[NormalizedEntity] = []
    master_relationships: list[NormalizedRelationship] = []
    master_embeddings: np.ndarray | None = None
    master_entity_order: list[str] = []
    uf = UnionFind()

    timings: dict[str, list[float]] = defaultdict(list)
    all_merge_decisions: list[dict] = []
    logger.info("Similarity threshold: %.2f", threshold)

    # --- Phase 1: Batch extract all paragraphs ---
    t0 = time.perf_counter()
    all_inputs = [{"paragraph_text": para["text"]} for para in paragraphs]
    logger.info(
        "Batch extracting %d paragraphs (max_concurrency=%d)...",
        len(paragraphs),
        max_concurrency,
    )
    extraction_results = _batch_with_retry(
        extraction_chain, all_inputs, max_concurrency
    )
    extraction_time = time.perf_counter() - t0
    timings["extraction"].append(extraction_time)
    logger.info(
        "Extraction complete: %d paragraphs in %.1fs (%.2fs avg)",
        len(paragraphs),
        extraction_time,
        extraction_time / len(paragraphs),
    )

    # Assign paragraph IDs to results
    for para, result in zip(paragraphs, extraction_results, strict=False):
        result.paragraph_id = para["id"]

    # --- Phase 1b: Parse temporal context ---
    temporal_chain = (
        create_temporal_chain(config) if config.get("temporal_model") else None
    )
    if temporal_chain:
        temporal_inputs = []
        temporal_indices: list[tuple[int, int]] = []
        for r_idx, result in enumerate(extraction_results):
            for rel_idx, rel in enumerate(result.relationships):
                if rel.temporal_context:
                    temporal_inputs.append(
                        {
                            "source_entity": rel.source_entity,
                            "relation_type": rel.relation_type,
                            "target_entity": rel.target_entity,
                            "temporal_context": rel.temporal_context,
                        }
                    )
                    temporal_indices.append((r_idx, rel_idx))

        if temporal_inputs:
            t0 = time.perf_counter()
            parsed = _batch_with_retry(temporal_chain, temporal_inputs, max_concurrency)
            temporal_time = time.perf_counter() - t0
            timings["temporal"].append(temporal_time)

            n_with_dates = 0
            for (r_idx, rel_idx), tp in zip(temporal_indices, parsed, strict=True):
                rel = extraction_results[r_idx].relationships[rel_idx]
                rel.start_year = tp.start_year
                rel.end_year = tp.end_year
                rel.temporal_precision = tp.precision
                if tp.start_year is not None:
                    n_with_dates += 1

            logger.info(
                "Temporal parsing: %d relationships parsed in %.1fs "
                "(%d with dates, %d unparseable)",
                len(temporal_inputs),
                temporal_time,
                n_with_dates,
                len(temporal_inputs) - n_with_dates,
            )

    # --- Phase 2: Incremental processing ---
    for i, (para, result) in enumerate(
        zip(paragraphs, extraction_results, strict=False)
    ):
        entities, relationships = assign_ids_single(result, paragraph_meta=para)

        if not entities:
            logger.info(
                "[%d] p%s para %s | 0 entities after filtering",
                i,
                para["page"],
                para["paragraph_index"],
            )
            continue

        # Rule-based merge into master
        t0 = time.perf_counter()
        master_entities, master_relationships, newly_added, rule_decisions = merge_rule_based(
            entities,
            relationships,
            master_entities,
            master_relationships,
            rule_filter_chain=rule_filter_chain,
            max_concurrency=max_concurrency,
        )
        all_merge_decisions.extend(rule_decisions)
        timings["rule_merge"].append(time.perf_counter() - t0)
        n_rule = len(entities) - len(newly_added)

        n_candidates = 0
        n_llm_checked = 0
        n_llm_merged = 0

        if newly_added:
            new_embs, llm_results, n_candidates, n_llm_checked, n_llm_merged = (
                _run_embedding_merge(
                    newly_added,
                    uf,
                    embeddings_model,
                    merge_chain,
                    master_entities,
                    master_embeddings,
                    master_entity_order,
                    config,
                    timings,
                    max_concurrency,
                    result_context={"paragraph_idx": i, "page": para["page"]},
                )
            )
            all_merge_decisions.extend(llm_results)

            # Update master embeddings
            if master_embeddings is None:
                master_embeddings = new_embs
            else:
                master_embeddings = np.vstack([master_embeddings, new_embs])
            master_entity_order.extend(e.id for e in newly_added)

        logger.info(
            "[%d] p%s para %s | +%d ext, %d kept | "
            "rule: %d merged, %d new | "
            "llm: %d cand, %d checked, %d merged | "
            "master: %de %dr",
            i,
            para["page"],
            para["paragraph_index"],
            len(result.entities),
            len(entities),
            n_rule,
            len(newly_added),
            n_candidates,
            n_llm_checked,
            n_llm_merged,
            len(master_entities),
            len(master_relationships),
        )

    # --- Finalize: apply Union-Find merges ---
    final_entities, final_relationships, _ = uf.finalize(
        master_entities, master_relationships
    )

    llm_merge_count = sum(1 for d in all_merge_decisions if d["merge_type"] == "llm")
    logger.info(
        "Final: %d entities, %d relationships",
        len(final_entities),
        len(final_relationships),
    )
    logger.info(
        "Pipeline: %d paragraphs -> %d after rule-based -> %d final (%d LLM merges)",
        len(paragraphs),
        len(master_entities),
        len(final_entities),
        llm_merge_count,
    )

    if profile:
        _log_profile(
            timings,
            [
                "extraction",
                "temporal",
                "rule_merge",
                "embedding",
                "similarity",
                "llm_merge",
            ],
        )

    return final_entities, final_relationships, all_merge_decisions


# ---------------------------------------------------------------------------
# Pipeline: cross-chapter merge
# ---------------------------------------------------------------------------


def _seed_master(
    seed_entities: list[NormalizedEntity],
    seed_relationships: list[NormalizedRelationship],
    uf: UnionFind,
    embeddings_model: OpenAIEmbeddings,
    timings: dict[str, list[float]],
) -> tuple[list[NormalizedEntity], list[NormalizedRelationship], np.ndarray, list[str]]:
    """Copy seed entities/relationships into fresh master state with new IDs.

    Returns (master_entities, master_relationships, master_embeddings, master_entity_order).
    """
    master_entities: list[NormalizedEntity] = []
    master_relationships: list[NormalizedRelationship] = []

    for e in seed_entities:
        new_entity = e.model_copy(
            deep=True,
            update={"id": str(uuid_module.uuid4()), "relationship_ids": []},
        )
        master_entities.append(new_entity)
        uf.add(new_entity)

    old_to_new = {}
    for old_e, new_e in zip(seed_entities, master_entities, strict=True):
        old_to_new[old_e.id] = new_e.id
    master_entity_lookup = {e.id: e for e in master_entities}
    for rel in seed_relationships:
        new_source = old_to_new.get(rel.source_id)
        new_target = old_to_new.get(rel.target_id)
        if new_source and new_target:
            new_rel = rel.model_copy(
                update={
                    "id": str(uuid_module.uuid4()),
                    "source_id": new_source,
                    "target_id": new_target,
                }
            )
            master_relationships.append(new_rel)
            if new_source in master_entity_lookup:
                master_entity_lookup[new_source].relationship_ids.append(new_rel.id)
            if new_target in master_entity_lookup:
                master_entity_lookup[new_target].relationship_ids.append(new_rel.id)

    t0 = time.perf_counter()
    texts = [create_entity_text(e) for e in master_entities]
    master_embeddings = np.array(embeddings_model.embed_documents(texts))
    master_entity_order = [e.id for e in master_entities]
    timings["embedding"].append(time.perf_counter() - t0)

    return master_entities, master_relationships, master_embeddings, master_entity_order


def run_cross_chapter_merge(  # noqa: PLR0912, PLR0915
    chapter_results: dict[
        int, tuple[list[NormalizedEntity], list[NormalizedRelationship]]
    ],
    config: dict,
    *,
    profile: bool = False,
    max_concurrency: int = 5,
    base_entities: list[NormalizedEntity] | None = None,
    base_relationships: list[NormalizedRelationship] | None = None,
) -> tuple[list[NormalizedEntity], list[NormalizedRelationship], list[dict]]:
    """Merge multiple chapters' KG results into a single unified graph.

    Returns (final_entities, final_relationships, llm_merge_results).
    """
    embeddings_model = OpenAIEmbeddings(model=config["embedding_model"])
    merge_chain = create_merge_chain(config)
    rule_filter_chain = (
        create_rule_filter_chain(config) if config.get("rule_filter_model") else None
    )

    master_entities: list[NormalizedEntity] = []
    master_relationships: list[NormalizedRelationship] = []
    master_embeddings: np.ndarray | None = None
    master_entity_order: list[str] = []
    uf = UnionFind()

    all_merge_decisions: list[dict] = []
    timings: dict[str, list[float]] = defaultdict(list)

    sorted_chapters = sorted(chapter_results.keys())
    has_base = base_entities is not None and base_relationships is not None

    if has_base:
        logger.info(
            "Cross-chapter merge: seeding from base graph (%d entities, %d rels), "
            "adding %d chapters %s",
            len(base_entities),
            len(base_relationships),
            len(sorted_chapters),
            sorted_chapters,
        )
    else:
        logger.info(
            "Cross-chapter merge: %d chapters %s",
            len(sorted_chapters),
            sorted_chapters,
        )

    if has_base:
        (
            master_entities,
            master_relationships,
            master_embeddings,
            master_entity_order,
        ) = _seed_master(
            base_entities, base_relationships, uf, embeddings_model, timings
        )
        logger.info(
            "  Seeded master from base graph: %d entities", len(master_entities)
        )

    for step, ch_idx in enumerate(sorted_chapters):
        ch_entities, ch_relationships = chapter_results[ch_idx]
        logger.info(
            "--- Merging chapter %d (%d entities, %d relationships) ---",
            ch_idx,
            len(ch_entities),
            len(ch_relationships),
        )

        if not has_base and step == 0:
            (
                master_entities,
                master_relationships,
                master_embeddings,
                master_entity_order,
            ) = _seed_master(
                ch_entities, ch_relationships, uf, embeddings_model, timings
            )
            logger.info("  Seeded master with %d entities", len(master_entities))
            continue

        # 1. Rule-based merge
        t0 = time.perf_counter()
        master_entities, master_relationships, newly_added, rule_decisions = merge_rule_based(
            ch_entities,
            ch_relationships,
            master_entities,
            master_relationships,
            rule_filter_chain=rule_filter_chain,
            max_concurrency=max_concurrency,
            remap_relationship_ids=True,
        )
        all_merge_decisions.extend(rule_decisions)
        timings["rule_merge"].append(time.perf_counter() - t0)
        n_rule_merged = len(ch_entities) - len(newly_added)

        # 2. Embedding merge
        n_candidates = 0
        n_llm_checked = 0
        n_llm_merged = 0

        if newly_added:
            new_embs, llm_results, n_candidates, n_llm_checked, n_llm_merged = (
                _run_embedding_merge(
                    newly_added,
                    uf,
                    embeddings_model,
                    merge_chain,
                    master_entities,
                    master_embeddings,
                    master_entity_order,
                    config,
                    timings,
                    max_concurrency,
                    result_context={"chapter_index": ch_idx},
                )
            )
            all_merge_decisions.extend(llm_results)

            if master_embeddings is None:
                master_embeddings = new_embs
            else:
                master_embeddings = np.vstack([master_embeddings, new_embs])
            master_entity_order.extend(e.id for e in newly_added)

        logger.info(
            "  Chapter %d: +%d new, %d rule-merged, %d candidates, "
            "%d LLM-checked, %d LLM-merged | master: %d entities",
            ch_idx,
            len(newly_added),
            n_rule_merged,
            n_candidates,
            n_llm_checked,
            n_llm_merged,
            len(master_entities),
        )

    # --- Finalize: apply Union-Find merges ---
    final_entities, final_relationships, _ = uf.finalize(
        master_entities, master_relationships
    )

    llm_merge_count = sum(1 for d in all_merge_decisions if d["merge_type"] == "llm")
    logger.info(
        "Cross-chapter final: %d entities, %d relationships (%d LLM merges)",
        len(final_entities),
        len(final_relationships),
        llm_merge_count,
    )

    if profile:
        _log_profile(timings, ["rule_merge", "embedding", "similarity", "llm_merge"])

    return final_entities, final_relationships, all_merge_decisions


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class KGIngestionService:
    """Orchestrates KG extraction, merging, and Weaviate storage."""

    def __init__(
        self,
        config: WeaviateConfig | None = None,
        pipeline_config: dict | None = None,
    ):
        if config is None:
            config = WeaviateConfig.from_environment()
        self._db_config = config
        self._config = {**DEFAULT_CONFIG, **(pipeline_config or {})}
        self._repo_manager: BookRepositoryManager | None = None

    @property
    def repositories(self) -> BookRepositoryManager:
        if self._repo_manager is None:
            self._repo_manager = BookRepositoryManager(self._db_config)
        return self._repo_manager

    # --- Public API ---

    def extract_chapter(
        self,
        book_index: int,
        chapter_index: int,
        *,
        force: bool = False,
        profile: bool = False,
        max_concurrency: int = 5,
    ) -> str:
        """Extract KG from a single chapter and store in Weaviate.

        Returns graph_name (e.g. 'book3_ch2').
        """
        graph_name = f"book{book_index}_ch{chapter_index}"

        # Check DB cache
        if not force:
            existing = self.repositories.kg_graphs.find_by_name(graph_name)
            if existing is not None:
                logger.info(
                    "Chapter %d: already in DB (%d entities, %d relationships). "
                    "Use force=True to regenerate.",
                    chapter_index,
                    existing.entity_count,
                    existing.relationship_count,
                )
                return graph_name

        # Load paragraphs and run extraction
        paragraphs = self._load_paragraphs(book_index, chapter_index)
        if not paragraphs:
            msg = f"No paragraphs found for book={book_index} chapter={chapter_index}"
            raise ValueError(msg)

        logger.info("Loaded %d paragraphs", len(paragraphs))

        final_entities, final_relationships, merge_decisions = run_pipeline(
            paragraphs,
            self._config,
            profile=profile,
            max_concurrency=max_concurrency,
        )

        # Store in Weaviate
        book_chapters = [f"{book_index}:{chapter_index}"]
        self._store_graph(
            graph_name, "chapter", final_entities, final_relationships, book_chapters
        )
        self._store_merge_decisions(graph_name, merge_decisions)

        return graph_name

    def merge_book(
        self,
        book_index: int,
        chapters: list[int] | None = None,
        *,
        force: bool = False,
        profile: bool = False,
        max_concurrency: int = 5,
    ) -> str:
        """Extract chapters (DB-cached) -> merge -> store as book graph.

        If chapters is None, auto-discovers all chapters from DB and uses
        name 'book{X}'. If a subset is specified, uses 'book{X}_ch{a}_{b}_{c}'.
        Returns graph_name.
        """
        # Auto-discover chapters if not specified
        is_full_book = chapters is None
        if is_full_book:
            db_chapters = self.repositories.chapters.find_by_book_index(book_index)
            if not db_chapters:
                msg = f"No chapters found for book index {book_index}"
                raise ValueError(msg)
            chapters = sorted(ch.chapter_index for ch in db_chapters)
            logger.info(
                "Auto-discovered %d chapters for book %d: %s",
                len(chapters),
                book_index,
                chapters,
            )
        else:
            chapters = sorted(chapters)

        if is_full_book:
            graph_name = f"book{book_index}"
        else:
            ch_suffix = "_".join(str(ch) for ch in chapters)
            graph_name = f"book{book_index}_ch{ch_suffix}"

        logger.info(
            "KG book merge: book=%d chapters=%s -> %s",
            book_index,
            chapters,
            graph_name,
        )

        # Phase 1: Ensure chapter graphs exist
        chapter_results: dict[
            int, tuple[list[NormalizedEntity], list[NormalizedRelationship]]
        ] = {}

        for ch_idx in chapters:
            ch_graph_name = f"book{book_index}_ch{ch_idx}"
            existing = self.repositories.kg_graphs.find_by_name(ch_graph_name)

            if existing is None or force:
                logger.info(
                    "Chapter %d: %s...",
                    ch_idx,
                    "re-extracting (force)" if existing else "extracting",
                )
                self.extract_chapter(
                    book_index,
                    ch_idx,
                    force=force,
                    profile=profile,
                    max_concurrency=max_concurrency,
                )

            # Load from DB
            result = self._load_graph_from_db(ch_graph_name)
            if result is None:
                logger.error(
                    "Failed to load chapter graph '%s' from DB, skipping",
                    ch_graph_name,
                )
                continue
            entities, relationships = result
            logger.info(
                "Chapter %d: loaded from DB (%d entities, %d relationships)",
                ch_idx,
                len(entities),
                len(relationships),
            )
            chapter_results[ch_idx] = (entities, relationships)

        if not chapter_results:
            msg = "No chapter results available for merge"
            raise ValueError(msg)

        if len(chapter_results) < 2:
            msg = f"Need at least 2 chapters for merge, got {len(chapter_results)}"
            raise ValueError(msg)

        # Phase 2: Cross-chapter merge
        logger.info("=== Phase 2: Cross-chapter merge ===")
        combined_entities, combined_rels, merge_decisions = run_cross_chapter_merge(
            chapter_results,
            self._config,
            profile=profile,
            max_concurrency=max_concurrency,
        )

        # Store in Weaviate
        book_chapters = [f"{book_index}:{ch}" for ch in chapters]
        self._store_graph(
            graph_name, "book", combined_entities, combined_rels, book_chapters
        )
        self._store_merge_decisions(graph_name, merge_decisions)

        return graph_name

    def merge_volume(
        self,
        book_indices: list[int],
        *,
        graph_name: str | None = None,
        force: bool = False,
        profile: bool = False,
        max_concurrency: int = 5,
    ) -> str:
        """Merge book graphs into a volume graph.

        Ensures book graphs exist (calls merge_book for missing ones if force,
        errors if not force and missing). Returns graph_name.
        """
        if not graph_name:
            graph_name = f"volume_{min(book_indices)}-{max(book_indices)}"

        # Ensure book graphs exist
        book_graph_names: list[str] = []
        for bi in sorted(book_indices):
            bg_name = f"book{bi}"
            existing = self.repositories.kg_graphs.find_by_name(bg_name)
            if existing is None:
                if force:
                    logger.info("Book %d graph missing, building...", bi)
                    self.merge_book(
                        bi,
                        force=force,
                        profile=profile,
                        max_concurrency=max_concurrency,
                    )
                else:
                    msg = (
                        f"Book graph '{bg_name}' not found in DB. "
                        f"Run 'book --book {bi}' first or use --force."
                    )
                    raise ValueError(msg)
            book_graph_names.append(bg_name)

        return self.merge_graphs(
            book_graph_names,
            output_name=graph_name,
            graph_type="volume",
            profile=profile,
            max_concurrency=max_concurrency,
        )

    def merge_custom(
        self,
        chapters: dict[int, list[int]],
        *,
        graph_name: str,
        force: bool = False,
        profile: bool = False,
        max_concurrency: int = 5,
    ) -> str:
        """Merge specific chapters from multiple books into a custom graph.

        Args:
            chapters: {book_index: [chapter_indices]} mapping.
            graph_name: Required name for the output graph.

        Returns graph_name.
        """
        # Ensure chapter graphs exist
        chapter_graph_names: list[str] = []
        for book_index in sorted(chapters.keys()):
            for ch_idx in sorted(chapters[book_index]):
                ch_graph_name = f"book{book_index}_ch{ch_idx}"
                existing = self.repositories.kg_graphs.find_by_name(ch_graph_name)
                if existing is None or force:
                    logger.info(
                        "Chapter book%d_ch%d: %s...",
                        book_index,
                        ch_idx,
                        "re-extracting (force)" if existing else "extracting",
                    )
                    self.extract_chapter(
                        book_index,
                        ch_idx,
                        force=force,
                        profile=profile,
                        max_concurrency=max_concurrency,
                    )
                chapter_graph_names.append(ch_graph_name)

        return self.merge_graphs(
            chapter_graph_names,
            output_name=graph_name,
            graph_type="custom",
            profile=profile,
            max_concurrency=max_concurrency,
        )

    def merge_graphs(
        self,
        graph_names: list[str],
        *,
        output_name: str,
        graph_type: str = "volume",
        profile: bool = False,
        max_concurrency: int = 5,
    ) -> str:
        """Merge N existing graphs from Weaviate into a new graph.

        Low-level merge-by-graph-name method. Returns graph_name.
        """
        graph_results: dict[
            int, tuple[list[NormalizedEntity], list[NormalizedRelationship]]
        ] = {}
        all_book_chapters: list[str] = []

        for idx, gname in enumerate(graph_names):
            result = self._load_graph_from_db(gname)
            if result is None:
                msg = f"Graph '{gname}' not found in database"
                raise ValueError(msg)
            entities, relationships = result
            graph_results[idx] = (entities, relationships)
            logger.info(
                "Loaded graph '%s': %d entities, %d relationships",
                gname,
                len(entities),
                len(relationships),
            )
            kg_graph = self.repositories.kg_graphs.find_by_name(gname)
            if kg_graph:
                all_book_chapters.extend(kg_graph.book_chapters)

        all_book_chapters = sorted(set(all_book_chapters))

        combined_entities, combined_rels, merge_decisions = run_cross_chapter_merge(
            graph_results,
            self._config,
            profile=profile,
            max_concurrency=max_concurrency,
        )

        self._store_graph(
            output_name, graph_type, combined_entities, combined_rels, all_book_chapters
        )
        self._store_merge_decisions(output_name, merge_decisions)

        logger.info("Done. Merged %d graphs into '%s'", len(graph_names), output_name)
        return output_name

    def load_graph(
        self, graph_name: str
    ) -> tuple[list[NormalizedEntity], list[NormalizedRelationship]] | None:
        """Load a graph from Weaviate. Public wrapper for standalone tools."""
        return self._load_graph_from_db(graph_name)

    def list_graphs(self) -> list[KGGraph]:
        """List all KG graphs stored in Weaviate."""
        return self.repositories.kg_graphs.list_all()

    def get_graph(self, graph_name: str) -> KGGraph | None:
        """Get a specific graph by name."""
        return self.repositories.kg_graphs.find_by_name(graph_name)

    def close(self) -> None:
        """Close all repository connections."""
        if self._repo_manager is not None:
            self._repo_manager.close_all()
            self._repo_manager = None

    # --- Private helpers ---

    def _load_paragraphs(self, book_index: int, chapter_index: int) -> list[dict]:
        """Load paragraphs from the database for a given book and chapter."""
        chapter_paragraphs = self.repositories.paragraphs.find_by_chapter_index(
            book_index=book_index, chapter_index=chapter_index
        )
        paragraphs = [
            {
                "id": str(p.id),
                "text": p.text,
                "page": p.page,
                "paragraph_index": p.paragraph_index,
                "book_index": p.book_index,
                "chapter_index": p.chapter_index,
            }
            for p in chapter_paragraphs
        ]
        paragraphs.sort(key=lambda p: (p.get("page", 0), p.get("paragraph_index", 0)))
        return paragraphs

    def _store_graph(
        self,
        graph_name: str,
        graph_type: str,
        entities: list[NormalizedEntity],
        relationships: list[NormalizedRelationship],
        book_chapters: list[str],
    ) -> None:
        """Convert pipeline entities/relationships to DB models and store in Weaviate."""
        # Delete existing graph data
        self.repositories.kg_entities.delete_by_graph(graph_name)
        self.repositories.kg_relationships.delete_by_graph(graph_name)

        # Convert entities with new DB IDs
        pipeline_id_to_db_id: dict[str, str] = {}
        kg_entities: list[KGEntity] = []
        for entity in entities:
            kg_entity = self._to_kg_entity(entity, graph_name)
            pipeline_id_to_db_id[entity.id] = kg_entity.id
            kg_entities.append(kg_entity)

        # Convert relationships with remapped IDs
        kg_relationships: list[KGRelationship] = []
        for rel in relationships:
            db_source = pipeline_id_to_db_id.get(rel.source_id)
            db_target = pipeline_id_to_db_id.get(rel.target_id)
            if db_source and db_target:
                kg_rel = KGRelationship(
                    graph_name=graph_name,
                    source_entity_id=db_source,
                    target_entity_id=db_target,
                    entity_ids=[db_source, db_target],
                    source_entity_name=rel.source_entity_name,
                    target_entity_name=rel.target_entity_name,
                    relation_type=rel.relation_type,
                    description=rel.description or "",
                    temporal_context=rel.temporal_context or "",
                    start_year=rel.start_year,
                    end_year=rel.end_year,
                    temporal_precision=rel.temporal_precision,
                    paragraph_id=rel.paragraph_id,
                    book_index=rel.book_index or 0,
                    chapter_index=rel.chapter_index or 0,
                    page=rel.page or 0,
                )
                kg_relationships.append(kg_rel)

        # Batch insert (pass None vectors — Weaviate auto-vectorizes from source properties)
        if kg_entities:
            self.repositories.kg_entities.batch_create_with_vectors(
                [(e, None) for e in kg_entities]
            )
        if kg_relationships:
            self.repositories.kg_relationships.batch_create_with_vectors(
                [(r, None) for r in kg_relationships]
            )

        # Create/update graph metadata
        existing = self.repositories.kg_graphs.find_by_name(graph_name)
        now = datetime.now(UTC)
        kg_graph = KGGraph(
            name=graph_name,
            graph_type=graph_type,
            book_chapters=book_chapters,
            entity_count=len(kg_entities),
            relationship_count=len(kg_relationships),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        if existing:
            updates = {
                "graph_type": graph_type,
                "book_chapters": book_chapters,
                "entity_count": len(kg_entities),
                "relationship_count": len(kg_relationships),
                "updated_at": now,
            }
            self.repositories.kg_graphs.update(existing.id, updates)
        else:
            self.repositories.kg_graphs.create(kg_graph)

        logger.info(
            "Stored graph '%s' in Weaviate: %d entities, %d relationships",
            graph_name,
            len(kg_entities),
            len(kg_relationships),
        )

    def _store_merge_decisions(self, graph_name: str, decisions: list[dict]) -> None:
        """Store merge audit decisions for a graph, replacing any existing records."""
        self.repositories.kg_merge_decisions.delete_by_graph(graph_name)
        if not decisions:
            return
        records = [KGMergeDecision(graph_name=graph_name, **d) for d in decisions]
        self.repositories.kg_merge_decisions.batch_create(records)
        logger.info("Stored %d merge decisions for '%s'", len(records), graph_name)

    def _load_graph_from_db(
        self, graph_name: str
    ) -> tuple[list[NormalizedEntity], list[NormalizedRelationship]] | None:
        """Load a graph from Weaviate and convert to pipeline models."""
        kg_graph = self.repositories.kg_graphs.find_by_name(graph_name)
        if kg_graph is None:
            return None

        kg_entities = self.repositories.kg_entities.find_by_graph(graph_name)
        kg_rels = self.repositories.kg_relationships.find_by_graph(graph_name)

        entities = []
        for ke in kg_entities:
            entities.append(
                NormalizedEntity(
                    id=ke.id,
                    name=ke.name,
                    type=ke.entity_type,
                    aliases=ke.aliases,
                    descriptions=[
                        SourceDescription(text=text, paragraph_id=pid)
                        for text, pid in zip(
                            ke.descriptions,
                            ke.description_paragraph_ids,
                            strict=False,
                        )
                    ],
                    source_paragraph_ids=ke.source_paragraph_ids,
                    book_indices=ke.book_indices,
                    source_book_chapters=ke.source_book_chapters,
                    source_pages=ke.source_pages,
                    occurrence_count=ke.occurrence_count,
                    merged_from_ids=[],
                    relationship_ids=[],
                    source_graph=graph_name,
                )
            )

        relationships = []
        entity_lookup = {e.id: e for e in entities}
        for kr in kg_rels:
            rel = NormalizedRelationship(
                id=kr.id,
                source_id=kr.source_entity_id,
                target_id=kr.target_entity_id,
                source_entity_name=kr.source_entity_name,
                target_entity_name=kr.target_entity_name,
                relation_type=kr.relation_type,
                description=kr.description or None,
                temporal_context=kr.temporal_context,
                start_year=kr.start_year,
                end_year=kr.end_year,
                temporal_precision=kr.temporal_precision,
                paragraph_id=kr.paragraph_id,
                book_index=kr.book_index,
                chapter_index=kr.chapter_index,
                page=kr.page,
            )
            relationships.append(rel)
            if rel.source_id in entity_lookup:
                entity_lookup[rel.source_id].relationship_ids.append(rel.id)
            if rel.target_id in entity_lookup:
                entity_lookup[rel.target_id].relationship_ids.append(rel.id)

        return entities, relationships

    @staticmethod
    def _to_kg_entity(entity: NormalizedEntity, graph_name: str) -> KGEntity:
        """Convert a pipeline NormalizedEntity to a DB KGEntity."""
        return KGEntity(
            graph_name=graph_name,
            name=entity.name,
            entity_type=entity.type,
            aliases=entity.aliases,
            descriptions=[d.text for d in entity.descriptions],
            description_paragraph_ids=[d.paragraph_id for d in entity.descriptions],
            occurrence_count=entity.occurrence_count,
            book_indices=entity.book_indices,
            source_book_chapters=entity.source_book_chapters,
            source_pages=entity.source_pages,
            source_paragraph_ids=entity.source_paragraph_ids,
            merged_from_count=len(entity.merged_from_ids),
        )
