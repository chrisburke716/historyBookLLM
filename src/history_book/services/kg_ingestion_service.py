"""KG ingestion service — extraction, merge, and Weaviate storage pipeline.

Translates the script pipeline into a production service. Processes paragraphs
from the database, extracts entities and relationships, normalizes via
rule-based + embedding + LLM merging, stores in Weaviate, and exports to files.
"""

import json
import logging
import time
import uuid as uuid_module
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity

from history_book.chains.kg_extraction_chain import (
    ExtractionResult,
    create_extraction_chain,
)
from history_book.chains.kg_merge_chain import MergedEntity, create_merge_chain
from history_book.data_models.kg_entities import KGEntity, KGGraph, KGRelationship
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
    "embedding_model": "text-embedding-3-small",
    "similarity_threshold": 0.65,
    "max_llm_candidates": 100,
    "reasoning_effort": "minimal",
}

# ---------------------------------------------------------------------------
# Pipeline-internal models
# ---------------------------------------------------------------------------


class EntityWithId(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    paragraph_id: str
    relationship_ids: list[str] = Field(default_factory=list)


class RelationshipWithId(BaseModel):
    id: str
    source_id: str
    target_id: str
    source_entity_name: str
    target_entity_name: str
    relation_type: str
    temporal_context: str | None = None
    paragraph_id: str


class NormalizedEntity(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str
    source_paragraph_ids: list[str]
    source_locations: list[dict] = Field(default_factory=list)
    occurrence_count: int
    merged_from_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)


class NormalizedRelationship(BaseModel):
    id: str
    source_id: str
    target_id: str
    source_entity_name: str
    target_entity_name: str
    relation_type: str
    temporal_context: str | None = None
    paragraph_id: str
    book_index: int | None = None
    chapter_index: int | None = None
    page: int | None = None


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
) -> tuple[list[EntityWithId], list[RelationshipWithId]]:
    """Assign UUIDs to entities and relationships from a single extraction.
    Drops orphaned entities (not referenced by any relationship)."""
    para_entities: dict[str, EntityWithId] = {}
    relationships: list[RelationshipWithId] = []
    skipped = 0

    for entity in result.entities:
        entity_id = str(uuid_module.uuid4())
        para_entities[entity.name] = EntityWithId(
            id=entity_id,
            name=entity.name,
            type=entity.type,
            aliases=entity.aliases,
            description=entity.description,
            paragraph_id=result.paragraph_id,
            relationship_ids=[],
        )

    for rel in result.relationships:
        source = para_entities.get(rel.source_entity)
        target = para_entities.get(rel.target_entity)
        if source and target:
            rel_id = str(uuid_module.uuid4())
            rel_with_id = RelationshipWithId(
                id=rel_id,
                source_id=source.id,
                target_id=target.id,
                source_entity_name=rel.source_entity,
                target_entity_name=rel.target_entity,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                paragraph_id=result.paragraph_id,
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
    """Create text representation of entity for embedding / search_text."""
    parts = [f"Name: {entity.name}", f"Type: {entity.type}"]
    if entity.description:
        parts.append(f"Description: {entity.description}")
    if entity.aliases:
        parts.append(f"Aliases: {', '.join(entity.aliases)}")
    return " | ".join(parts)


def merge_into_master_rule_based(
    new_entities: list[EntityWithId],
    new_relationships: list[RelationshipWithId],
    master_entities: list[NormalizedEntity],
    master_relationships: list[NormalizedRelationship],
    paragraph_meta: dict | None = None,
) -> tuple[
    list[NormalizedEntity], list[NormalizedRelationship], list[NormalizedEntity]
]:
    """Merge new paragraph entities into master graph using exact name + alias matching.

    Returns (updated_master_entities, updated_master_relationships, newly_added_entities).
    """
    name_to_master: dict[str, NormalizedEntity] = {}
    for me in master_entities:
        name_to_master[me.name.lower().strip()] = me
        for alias in me.aliases:
            alias_key = alias.lower().strip()
            if alias_key:
                name_to_master[alias_key] = me

    old_id_to_master_id: dict[str, str] = {}
    newly_added: list[NormalizedEntity] = []
    rule_merges = 0

    for entity in new_entities:
        key = entity.name.lower().strip()
        match = name_to_master.get(key)
        if not match:
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                match = name_to_master.get(alias_key)
                if match:
                    break

        loc = None
        if paragraph_meta:
            loc = {
                "book_index": paragraph_meta.get("book_index"),
                "chapter_index": paragraph_meta.get("chapter_index"),
                "page": paragraph_meta.get("page"),
                "paragraph_index": paragraph_meta.get("paragraph_index"),
                "paragraph_id": paragraph_meta.get("id"),
            }

        if match:
            rule_merges += 1
            logger.debug("    Rule merge: '%s' -> '%s'", entity.name, match.name)
            match.aliases = list(set(match.aliases + entity.aliases + [entity.name]))
            match.aliases = [
                a
                for a in match.aliases
                if a.lower().strip() != match.name.lower().strip()
            ]
            if entity.description:
                match.description = (
                    f"{match.description} | {entity.description}"
                    if match.description
                    else entity.description
                )
            match.source_paragraph_ids = list(
                set(match.source_paragraph_ids + [entity.paragraph_id])
            )
            if loc:
                match.source_locations.append(loc)
            match.occurrence_count += 1
            match.merged_from_ids.append(entity.id)
            old_id_to_master_id[entity.id] = match.id
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                if alias_key:
                    name_to_master[alias_key] = match
        else:
            new_master = NormalizedEntity(
                id=str(uuid_module.uuid4()),
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                description=entity.description or "",
                source_paragraph_ids=[entity.paragraph_id],
                source_locations=[loc] if loc else [],
                occurrence_count=1,
                merged_from_ids=[entity.id],
                relationship_ids=[],
            )
            master_entities.append(new_master)
            newly_added.append(new_master)
            old_id_to_master_id[entity.id] = new_master.id
            name_to_master[key] = new_master
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                if alias_key:
                    name_to_master[alias_key] = new_master

    master_entity_lookup = {e.id: e for e in master_entities}
    for rel in new_relationships:
        new_source = old_id_to_master_id.get(rel.source_id)
        new_target = old_id_to_master_id.get(rel.target_id)
        if new_source and new_target:
            norm_rel = NormalizedRelationship(
                id=rel.id,
                source_id=new_source,
                target_id=new_target,
                source_entity_name=rel.source_entity_name,
                target_entity_name=rel.target_entity_name,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                paragraph_id=rel.paragraph_id,
                book_index=paragraph_meta.get("book_index") if paragraph_meta else None,
                chapter_index=paragraph_meta.get("chapter_index")
                if paragraph_meta
                else None,
                page=paragraph_meta.get("page") if paragraph_meta else None,
            )
            master_relationships.append(norm_rel)
            if new_source in master_entity_lookup:
                master_entity_lookup[new_source].relationship_ids.append(rel.id)
            if new_target in master_entity_lookup:
                master_entity_lookup[new_target].relationship_ids.append(rel.id)

    if rule_merges:
        logger.debug("    %d rule-based merge(s)", rule_merges)

    return master_entities, master_relationships, newly_added


def merge_normalized_entities(
    new_entities: list[NormalizedEntity],
    new_relationships: list[NormalizedRelationship],
    master_entities: list[NormalizedEntity],
    master_relationships: list[NormalizedRelationship],
) -> tuple[
    list[NormalizedEntity], list[NormalizedRelationship], list[NormalizedEntity]
]:
    """Merge NormalizedEntities into master using rule-based name/alias matching.

    Works for chapter->book and book->volume merges.
    Returns (updated_master_entities, updated_master_relationships, newly_added_entities).
    """
    name_to_master: dict[str, NormalizedEntity] = {}
    for me in master_entities:
        name_to_master[me.name.lower().strip()] = me
        for alias in me.aliases:
            alias_key = alias.lower().strip()
            if alias_key:
                name_to_master[alias_key] = me

    old_id_to_master_id: dict[str, str] = {}
    newly_added: list[NormalizedEntity] = []
    rule_merges = 0

    for entity in new_entities:
        key = entity.name.lower().strip()
        match = name_to_master.get(key)
        if not match:
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                match = name_to_master.get(alias_key)
                if match:
                    break

        if match:
            rule_merges += 1
            logger.debug(
                "    Cross-chapter rule merge: '%s' -> '%s'", entity.name, match.name
            )
            match.aliases = list(set(match.aliases + entity.aliases + [entity.name]))
            match.aliases = [
                a
                for a in match.aliases
                if a.lower().strip() != match.name.lower().strip()
            ]
            if entity.description:
                match.description = (
                    f"{match.description} | {entity.description}"
                    if match.description
                    else entity.description
                )
            match.source_paragraph_ids = list(
                set(match.source_paragraph_ids + entity.source_paragraph_ids)
            )
            existing_pids = {loc.get("paragraph_id") for loc in match.source_locations}
            for loc in entity.source_locations:
                if loc.get("paragraph_id") not in existing_pids:
                    match.source_locations.append(loc)
                    existing_pids.add(loc.get("paragraph_id"))
            match.occurrence_count += entity.occurrence_count
            match.merged_from_ids = list(
                set(match.merged_from_ids + entity.merged_from_ids)
            )
            old_id_to_master_id[entity.id] = match.id
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                if alias_key:
                    name_to_master[alias_key] = match
        else:
            new_id = str(uuid_module.uuid4())
            new_master = NormalizedEntity(
                id=new_id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                description=entity.description,
                source_paragraph_ids=list(entity.source_paragraph_ids),
                source_locations=list(entity.source_locations),
                occurrence_count=entity.occurrence_count,
                merged_from_ids=list(entity.merged_from_ids),
                relationship_ids=[],
            )
            master_entities.append(new_master)
            newly_added.append(new_master)
            old_id_to_master_id[entity.id] = new_id
            name_to_master[key] = new_master
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                if alias_key:
                    name_to_master[alias_key] = new_master

    master_entity_lookup = {e.id: e for e in master_entities}
    for rel in new_relationships:
        new_source = old_id_to_master_id.get(rel.source_id)
        new_target = old_id_to_master_id.get(rel.target_id)
        if new_source and new_target:
            new_rel = NormalizedRelationship(
                id=str(uuid_module.uuid4()),
                source_id=new_source,
                target_id=new_target,
                source_entity_name=rel.source_entity_name,
                target_entity_name=rel.target_entity_name,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                paragraph_id=rel.paragraph_id,
                book_index=rel.book_index,
                chapter_index=rel.chapter_index,
                page=rel.page,
            )
            master_relationships.append(new_rel)
            if new_source in master_entity_lookup:
                master_entity_lookup[new_source].relationship_ids.append(new_rel.id)
            if new_target in master_entity_lookup:
                master_entity_lookup[new_target].relationship_ids.append(new_rel.id)

    if rule_merges:
        logger.info("    %d cross-chapter rule-based merge(s)", rule_merges)

    return master_entities, master_relationships, newly_added


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
            if sim >= threshold:
                candidates.append(
                    {
                        "new_entity": new_entities[i],
                        "master_entity": master_entities[j],
                        "similarity": float(sim),
                    }
                )

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates


def format_entity_for_prompt(entity) -> dict:
    """Format an entity for the merge prompt."""
    aliases = entity.aliases if entity.aliases else []
    return {
        "name": entity.name,
        "type": entity.type,
        "aliases": ", ".join(aliases) if aliases else "None",
        "description": entity.description or "None",
    }


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
    retryable_keywords = ["rate", "timeout", "timed out", "connection"]
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
            description=entity.description,
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
        if entity.description:
            desc = (
                entity.description[:150] + "..."
                if len(entity.description) > 150
                else entity.description
            )
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
# JSON serialization helpers
# ---------------------------------------------------------------------------


def save_entities_json(entities: list[NormalizedEntity], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.model_dump() for e in entities]
    path.write_text(json.dumps(data, indent=2))


def save_relationships_json(
    relationships: list[NormalizedRelationship], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in relationships]
    path.write_text(json.dumps(data, indent=2))


def load_entities_json(path: Path) -> list[NormalizedEntity]:
    data = json.loads(path.read_text())
    return [NormalizedEntity.model_validate(d) for d in data]


def load_relationships_json(path: Path) -> list[NormalizedRelationship]:
    data = json.loads(path.read_text())
    return [NormalizedRelationship.model_validate(d) for d in data]


def _populate_relationship_ids(
    entities: list[NormalizedEntity],
    relationships: list[NormalizedRelationship],
) -> None:
    """Populate relationship_ids on entities from relationships (in-place)."""
    entity_lookup = {e.id: e for e in entities}
    for rel in relationships:
        if rel.source_id in entity_lookup:
            entity_lookup[rel.source_id].relationship_ids.append(rel.id)
        if rel.target_id in entity_lookup:
            entity_lookup[rel.target_id].relationship_ids.append(rel.id)


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------


def export_results(
    entities: list[NormalizedEntity],
    relationships: list[NormalizedRelationship],
    llm_results: list[dict],
    output_dir: Path,
    skip_visualization: bool = False,
) -> None:
    """Export pipeline results to JSON + CSV files and optional PyVis visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (lossless)
    save_entities_json(entities, output_dir / "entities.json")
    save_relationships_json(relationships, output_dir / "relationships.json")
    logger.info("Exported JSON to %s", output_dir)

    # Entities CSV
    entity_records = []
    for e in sorted(entities, key=lambda x: x.occurrence_count, reverse=True):
        chapters = sorted(
            {
                loc["chapter_index"]
                for loc in e.source_locations
                if loc.get("chapter_index") is not None
            }
        )
        pages = sorted(
            {loc["page"] for loc in e.source_locations if loc.get("page") is not None}
        )
        entity_records.append(
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "aliases": "; ".join(e.aliases) if e.aliases else "",
                "description": e.description,
                "occurrence_count": e.occurrence_count,
                "n_relationships": len(e.relationship_ids),
                "n_merged_from": len(e.merged_from_ids),
                "source_paragraph_ids": "; ".join(e.source_paragraph_ids),
                "source_chapters": "; ".join(str(c) for c in chapters),
                "source_pages": "; ".join(str(p) for p in pages),
            }
        )
    df_entities = pd.DataFrame(entity_records)
    df_entities.to_csv(output_dir / "entities.csv", index=False)

    # Relationships CSV
    rel_records = []
    entity_lookup = {e.id: e for e in entities}
    for r in relationships:
        source_name = (
            entity_lookup[r.source_id].name
            if r.source_id in entity_lookup
            else r.source_entity_name
        )
        target_name = (
            entity_lookup[r.target_id].name
            if r.target_id in entity_lookup
            else r.target_entity_name
        )
        rel_records.append(
            {
                "id": r.id,
                "source_id": r.source_id,
                "target_id": r.target_id,
                "source_name": source_name,
                "target_name": target_name,
                "source_original": r.source_entity_name,
                "target_original": r.target_entity_name,
                "relation_type": r.relation_type,
                "temporal_context": r.temporal_context or "",
                "paragraph_id": r.paragraph_id,
                "book_index": r.book_index if r.book_index is not None else "",
                "chapter_index": r.chapter_index if r.chapter_index is not None else "",
                "page": r.page if r.page is not None else "",
            }
        )
    df_rels = pd.DataFrame(rel_records)
    df_rels.to_csv(output_dir / "relationships.csv", index=False)

    # LLM merge results CSV
    if llm_results:
        df_llm = pd.DataFrame(llm_results)
        df_llm.to_csv(output_dir / "llm_merge_results.csv", index=False)

    # PyVis visualization
    if not skip_visualization:
        G = build_knowledge_graph(entities, relationships)
        logger.info(
            "Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        viz_path = str(output_dir / "knowledge_graph.html")
        visualize_with_pyvis(G, entities, relationships, viz_path)


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
    threshold = config["similarity_threshold"]

    master_entities: list[NormalizedEntity] = []
    master_relationships: list[NormalizedRelationship] = []
    master_embeddings: np.ndarray | None = None
    master_entity_order: list[str] = []

    # Union-Find for LLM merges
    uf_parent: dict[str, str] = {}
    uf_representative: dict[str, NormalizedEntity] = {}

    def uf_find(entity_id: str) -> str:
        while uf_parent[entity_id] != entity_id:
            uf_parent[entity_id] = uf_parent[uf_parent[entity_id]]
            entity_id = uf_parent[entity_id]
        return entity_id

    def uf_union(id1: str, id2: str, merged_entity: MergedEntity | None):
        root1, root2 = uf_find(id1), uf_find(id2)
        if root1 == root2:
            return
        uf_parent[root2] = root1
        rep, other = uf_representative[root1], uf_representative[root2]
        combined_locations = rep.source_locations + other.source_locations
        if merged_entity:
            updated = NormalizedEntity(
                id=rep.id,
                name=merged_entity.name,
                type=merged_entity.type,
                aliases=list(
                    set(rep.aliases + other.aliases + (merged_entity.aliases or []))
                ),
                description=merged_entity.description or rep.description,
                source_paragraph_ids=list(
                    set(rep.source_paragraph_ids + other.source_paragraph_ids)
                ),
                source_locations=combined_locations,
                occurrence_count=rep.occurrence_count + other.occurrence_count,
                merged_from_ids=list(set(rep.merged_from_ids + other.merged_from_ids)),
                relationship_ids=list(
                    set(rep.relationship_ids + other.relationship_ids)
                ),
            )
        else:
            updated = NormalizedEntity(
                id=rep.id,
                name=rep.name,
                type=rep.type,
                aliases=list(set(rep.aliases + other.aliases)),
                description=" | ".join(
                    d for d in [rep.description, other.description] if d
                ),
                source_paragraph_ids=list(
                    set(rep.source_paragraph_ids + other.source_paragraph_ids)
                ),
                source_locations=combined_locations,
                occurrence_count=rep.occurrence_count + other.occurrence_count,
                merged_from_ids=list(set(rep.merged_from_ids + other.merged_from_ids)),
                relationship_ids=list(
                    set(rep.relationship_ids + other.relationship_ids)
                ),
            )
        uf_representative[root1] = updated

    timings: dict[str, list[float]] = defaultdict(list)
    all_llm_results: list[dict] = []
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

    # --- Phase 2: Incremental processing ---
    for i, (para, result) in enumerate(
        zip(paragraphs, extraction_results, strict=False)
    ):
        entities, relationships = assign_ids_single(result)

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
        master_entities, master_relationships, newly_added = (
            merge_into_master_rule_based(
                entities,
                relationships,
                master_entities,
                master_relationships,
                paragraph_meta=para,
            )
        )
        timings["rule_merge"].append(time.perf_counter() - t0)
        n_rule = len(entities) - len(newly_added)

        n_candidates = 0
        n_llm_checked = 0
        n_llm_merged = 0

        if newly_added:
            for ne in newly_added:
                uf_parent[ne.id] = ne.id
                uf_representative[ne.id] = ne

            t0 = time.perf_counter()
            new_texts = [create_entity_text(e) for e in newly_added]
            new_embs = np.array(embeddings_model.embed_documents(new_texts))
            timings["embedding"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            candidates = []
            if master_embeddings is not None and len(master_entity_order) > 0:
                master_lookup = {e.id: e for e in master_entities}
                existing_entities = [master_lookup[eid] for eid in master_entity_order]
                candidates = find_candidates_against_master(
                    newly_added,
                    new_embs,
                    existing_entities,
                    master_embeddings,
                    threshold,
                )
            timings["similarity"].append(time.perf_counter() - t0)
            n_candidates = len(candidates)

            # Batch LLM merge on candidates
            t0 = time.perf_counter()
            trimmed = candidates[: config["max_llm_candidates"]]
            pairs_to_check = []
            for c in trimmed:
                ne, me = c["new_entity"], c["master_entity"]
                root_new, root_master = uf_find(ne.id), uf_find(me.id)
                if root_new != root_master:
                    rep_new = uf_representative[root_new]
                    rep_master = uf_representative[root_master]
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
                    all_llm_results.append(
                        {
                            "paragraph_idx": i,
                            "page": para["page"],
                            "entity1_name": rep_new.name,
                            "entity2_name": rep_master.name,
                            "cosine_similarity": c["similarity"],
                            "should_merge": decision.should_merge,
                            "confidence": decision.confidence,
                            "reasoning": decision.reasoning,
                        }
                    )
                    if decision.should_merge:
                        ne = c["new_entity"]
                        me = c["master_entity"]
                        uf_union(ne.id, me.id, decision.merged_entity)
                        n_llm_merged += 1
                        logger.info(
                            "  ** LLM MERGE: %s + %s (cos:%.3f)",
                            rep_new.name,
                            rep_master.name,
                            c["similarity"],
                        )

            timings["llm_merge"].append(time.perf_counter() - t0)

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
    groups: dict[str, list[str]] = defaultdict(list)
    for eid in uf_parent:
        root = uf_find(eid)
        groups[root].append(eid)

    final_entities: list[NormalizedEntity] = []
    master_id_to_final_id: dict[str, str] = {}

    for root_id, member_ids in groups.items():
        rep = uf_representative[root_id]
        final_id = str(uuid_module.uuid4())
        final_entity = NormalizedEntity(
            id=final_id,
            name=rep.name,
            type=rep.type,
            aliases=rep.aliases,
            description=rep.description,
            source_paragraph_ids=rep.source_paragraph_ids,
            source_locations=rep.source_locations,
            occurrence_count=rep.occurrence_count,
            merged_from_ids=rep.merged_from_ids,
            relationship_ids=[],
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
            final_rel = NormalizedRelationship(
                id=rel.id,
                source_id=final_source,
                target_id=final_target,
                source_entity_name=rel.source_entity_name,
                target_entity_name=rel.target_entity_name,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                paragraph_id=rel.paragraph_id,
            )
            final_relationships.append(final_rel)
            if final_source in final_entity_lookup:
                final_entity_lookup[final_source].relationship_ids.append(rel.id)
            if final_target in final_entity_lookup:
                final_entity_lookup[final_target].relationship_ids.append(rel.id)

    llm_merge_count = sum(1 for r in all_llm_results if r["should_merge"])
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
        logger.info("")
        logger.info("%-16s %10s %8s %8s", "Stage", "Total(s)", "Calls", "Avg(s)")
        logger.info("-" * 46)
        total_time = 0.0
        for stage in [
            "extraction",
            "rule_merge",
            "embedding",
            "similarity",
            "llm_merge",
        ]:
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

    return final_entities, final_relationships, all_llm_results


# ---------------------------------------------------------------------------
# Pipeline: cross-chapter merge
# ---------------------------------------------------------------------------


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
) -> tuple[
    list[NormalizedEntity], list[NormalizedRelationship], list[dict], list[dict]
]:
    """Merge multiple chapters' KG results into a single unified graph.

    Returns (final_entities, final_relationships, llm_merge_results, scaling_metrics).
    """
    embeddings_model = OpenAIEmbeddings(model=config["embedding_model"])
    merge_chain = create_merge_chain(config)
    threshold = config["similarity_threshold"]

    master_entities: list[NormalizedEntity] = []
    master_relationships: list[NormalizedRelationship] = []
    master_embeddings: np.ndarray | None = None
    master_entity_order: list[str] = []

    uf_parent: dict[str, str] = {}
    uf_representative: dict[str, NormalizedEntity] = {}

    def uf_find(entity_id: str) -> str:
        while uf_parent[entity_id] != entity_id:
            uf_parent[entity_id] = uf_parent[uf_parent[entity_id]]
            entity_id = uf_parent[entity_id]
        return entity_id

    def uf_union(id1: str, id2: str, merged_entity: MergedEntity | None):
        root1, root2 = uf_find(id1), uf_find(id2)
        if root1 == root2:
            return
        uf_parent[root2] = root1
        rep, other = uf_representative[root1], uf_representative[root2]
        combined_locations = rep.source_locations + other.source_locations
        if merged_entity:
            updated = NormalizedEntity(
                id=rep.id,
                name=merged_entity.name,
                type=merged_entity.type,
                aliases=list(
                    set(rep.aliases + other.aliases + (merged_entity.aliases or []))
                ),
                description=merged_entity.description or rep.description,
                source_paragraph_ids=list(
                    set(rep.source_paragraph_ids + other.source_paragraph_ids)
                ),
                source_locations=combined_locations,
                occurrence_count=rep.occurrence_count + other.occurrence_count,
                merged_from_ids=list(set(rep.merged_from_ids + other.merged_from_ids)),
                relationship_ids=list(
                    set(rep.relationship_ids + other.relationship_ids)
                ),
            )
        else:
            updated = NormalizedEntity(
                id=rep.id,
                name=rep.name,
                type=rep.type,
                aliases=list(set(rep.aliases + other.aliases)),
                description=" | ".join(
                    d for d in [rep.description, other.description] if d
                ),
                source_paragraph_ids=list(
                    set(rep.source_paragraph_ids + other.source_paragraph_ids)
                ),
                source_locations=combined_locations,
                occurrence_count=rep.occurrence_count + other.occurrence_count,
                merged_from_ids=list(set(rep.merged_from_ids + other.merged_from_ids)),
                relationship_ids=list(
                    set(rep.relationship_ids + other.relationship_ids)
                ),
            )
        uf_representative[root1] = updated

    all_llm_results: list[dict] = []
    scaling_metrics: list[dict] = []
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

    def _seed_master(
        seed_entities: list[NormalizedEntity],
        seed_relationships: list[NormalizedRelationship],
    ) -> None:
        nonlocal \
            master_entities, \
            master_relationships, \
            master_embeddings, \
            master_entity_order
        for e in seed_entities:
            new_id = str(uuid_module.uuid4())
            new_entity = NormalizedEntity(
                id=new_id,
                name=e.name,
                type=e.type,
                aliases=e.aliases,
                description=e.description,
                source_paragraph_ids=list(e.source_paragraph_ids),
                source_locations=list(e.source_locations),
                occurrence_count=e.occurrence_count,
                merged_from_ids=list(e.merged_from_ids),
                relationship_ids=[],
            )
            master_entities.append(new_entity)
            uf_parent[new_id] = new_id
            uf_representative[new_id] = new_entity

        old_to_new = {}
        for old_e, new_e in zip(seed_entities, master_entities, strict=True):
            old_to_new[old_e.id] = new_e.id
        master_entity_lookup = {e.id: e for e in master_entities}
        for rel in seed_relationships:
            new_source = old_to_new.get(rel.source_id)
            new_target = old_to_new.get(rel.target_id)
            if new_source and new_target:
                new_rel = NormalizedRelationship(
                    id=str(uuid_module.uuid4()),
                    source_id=new_source,
                    target_id=new_target,
                    source_entity_name=rel.source_entity_name,
                    target_entity_name=rel.target_entity_name,
                    relation_type=rel.relation_type,
                    temporal_context=rel.temporal_context,
                    paragraph_id=rel.paragraph_id,
                    book_index=rel.book_index,
                    chapter_index=rel.chapter_index,
                    page=rel.page,
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

    if has_base:
        _seed_master(base_entities, base_relationships)
        scaling_metrics.append(
            {
                "step": "base",
                "chapter_index": "base",
                "master_size": len(master_entities),
                "n_new": len(master_entities),
                "n_rule_merged": 0,
                "n_candidates": 0,
                "n_llm_checked": 0,
                "n_llm_merged": 0,
            }
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
            _seed_master(ch_entities, ch_relationships)
            scaling_metrics.append(
                {
                    "step": step,
                    "chapter_index": ch_idx,
                    "master_size": len(master_entities),
                    "n_new": len(master_entities),
                    "n_rule_merged": 0,
                    "n_candidates": 0,
                    "n_llm_checked": 0,
                    "n_llm_merged": 0,
                }
            )
            logger.info("  Seeded master with %d entities", len(master_entities))
            continue

        # 1. Rule-based merge
        t0 = time.perf_counter()
        master_entities, master_relationships, newly_added = merge_normalized_entities(
            ch_entities, ch_relationships, master_entities, master_relationships
        )
        timings["rule_merge"].append(time.perf_counter() - t0)
        n_rule_merged = len(ch_entities) - len(newly_added)

        # 2. Embed newly added entities
        n_candidates = 0
        n_llm_checked = 0
        n_llm_merged = 0

        if newly_added:
            for ne in newly_added:
                uf_parent[ne.id] = ne.id
                uf_representative[ne.id] = ne

            t0 = time.perf_counter()
            new_texts = [create_entity_text(e) for e in newly_added]
            new_embs = np.array(embeddings_model.embed_documents(new_texts))
            timings["embedding"].append(time.perf_counter() - t0)

            # 3. Find candidates against existing master
            t0 = time.perf_counter()
            candidates = []
            if master_embeddings is not None and len(master_entity_order) > 0:
                master_lookup = {e.id: e for e in master_entities}
                existing_entities = [
                    master_lookup[eid]
                    for eid in master_entity_order
                    if eid in master_lookup
                ]
                existing_embs = master_embeddings[: len(existing_entities)]
                candidates = find_candidates_against_master(
                    newly_added, new_embs, existing_entities, existing_embs, threshold
                )
            timings["similarity"].append(time.perf_counter() - t0)
            n_candidates = len(candidates)

            # 4. Batch LLM merge
            t0 = time.perf_counter()
            trimmed = candidates[: config["max_llm_candidates"]]
            pairs_to_check = []
            for c in trimmed:
                ne, me = c["new_entity"], c["master_entity"]
                root_new = uf_find(ne.id)
                root_master = uf_find(me.id) if me.id in uf_parent else me.id
                if me.id not in uf_parent:
                    uf_parent[me.id] = me.id
                    uf_representative[me.id] = me
                    root_master = me.id
                if root_new != root_master:
                    rep_new = uf_representative[root_new]
                    rep_master = uf_representative[root_master]
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
                    all_llm_results.append(
                        {
                            "chapter_index": ch_idx,
                            "entity1_name": rep_new.name,
                            "entity2_name": rep_master.name,
                            "cosine_similarity": c["similarity"],
                            "should_merge": decision.should_merge,
                            "confidence": decision.confidence,
                            "reasoning": decision.reasoning,
                        }
                    )
                    if decision.should_merge:
                        uf_union(
                            c["new_entity"].id,
                            c["master_entity"].id,
                            decision.merged_entity,
                        )
                        n_llm_merged += 1
                        logger.info(
                            "  ** LLM MERGE: %s + %s (cos:%.3f)",
                            rep_new.name,
                            rep_master.name,
                            c["similarity"],
                        )

            timings["llm_merge"].append(time.perf_counter() - t0)

            # 5. Update master embeddings
            if master_embeddings is None:
                master_embeddings = new_embs
            else:
                master_embeddings = np.vstack([master_embeddings, new_embs])
            master_entity_order.extend(e.id for e in newly_added)

        scaling_metrics.append(
            {
                "step": step,
                "chapter_index": ch_idx,
                "master_size": len(master_entities),
                "n_new": len(newly_added),
                "n_rule_merged": n_rule_merged,
                "n_candidates": n_candidates,
                "n_llm_checked": n_llm_checked,
                "n_llm_merged": n_llm_merged,
            }
        )
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
    for e in master_entities:
        if e.id not in uf_parent:
            uf_parent[e.id] = e.id
            uf_representative[e.id] = e

    groups: dict[str, list[str]] = defaultdict(list)
    for eid in uf_parent:
        root = uf_find(eid)
        groups[root].append(eid)

    final_entities: list[NormalizedEntity] = []
    master_id_to_final_id: dict[str, str] = {}

    for root_id, member_ids in groups.items():
        rep = uf_representative[root_id]
        final_id = str(uuid_module.uuid4())
        final_entity = NormalizedEntity(
            id=final_id,
            name=rep.name,
            type=rep.type,
            aliases=rep.aliases,
            description=rep.description,
            source_paragraph_ids=rep.source_paragraph_ids,
            source_locations=rep.source_locations,
            occurrence_count=rep.occurrence_count,
            merged_from_ids=rep.merged_from_ids,
            relationship_ids=[],
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
            final_rel = NormalizedRelationship(
                id=rel.id,
                source_id=final_source,
                target_id=final_target,
                source_entity_name=rel.source_entity_name,
                target_entity_name=rel.target_entity_name,
                relation_type=rel.relation_type,
                temporal_context=rel.temporal_context,
                paragraph_id=rel.paragraph_id,
                book_index=rel.book_index,
                chapter_index=rel.chapter_index,
                page=rel.page,
            )
            final_relationships.append(final_rel)
            if final_source in final_entity_lookup:
                final_entity_lookup[final_source].relationship_ids.append(rel.id)
            if final_target in final_entity_lookup:
                final_entity_lookup[final_target].relationship_ids.append(rel.id)

    llm_merge_count = sum(1 for r in all_llm_results if r["should_merge"])
    logger.info(
        "Cross-chapter final: %d entities, %d relationships (%d LLM merges)",
        len(final_entities),
        len(final_relationships),
        llm_merge_count,
    )

    if profile:
        logger.info("")
        logger.info("%-16s %10s %8s %8s", "Stage", "Total(s)", "Calls", "Avg(s)")
        logger.info("-" * 46)
        total_time = 0.0
        for stage in ["rule_merge", "embedding", "similarity", "llm_merge"]:
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

    return final_entities, final_relationships, all_llm_results, scaling_metrics


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class KGIngestionService:
    """Orchestrates KG extraction, merging, file export, and Weaviate storage."""

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
        skip_visualization: bool = False,
    ) -> str:
        """Extract KG from a single chapter.

        Stores as chapter graph in Weaviate + writes to file cache.
        Returns graph_name (e.g. 'book3_ch2').
        """
        graph_name = f"book{book_index}_ch{chapter_index}"
        cache_dir = self._chapter_cache_dir(book_index, chapter_index)

        # Check file cache
        if not force:
            cached = self._load_chapter_cache(cache_dir)
            if cached is not None:
                entities, relationships = cached
                logger.info(
                    "Chapter %d: already cached (%d entities, %d relationships). "
                    "Use force=True to regenerate.",
                    chapter_index,
                    len(entities),
                    len(relationships),
                )
                # Ensure stored in DB even if from cache
                book_chapters = [f"{book_index}:{chapter_index}"]
                self._store_graph(
                    graph_name, "chapter", entities, relationships, book_chapters
                )
                return graph_name

        # Load paragraphs and run extraction
        paragraphs = self._load_paragraphs(book_index, chapter_index)
        if not paragraphs:
            msg = f"No paragraphs found for book={book_index} chapter={chapter_index}"
            raise ValueError(msg)

        logger.info("Loaded %d paragraphs", len(paragraphs))

        final_entities, final_relationships, llm_results = run_pipeline(
            paragraphs,
            self._config,
            profile=profile,
            max_concurrency=max_concurrency,
        )

        # Export to file cache
        export_results(
            final_entities,
            final_relationships,
            llm_results,
            cache_dir,
            skip_visualization=skip_visualization,
        )
        logger.info("Done. Output written to %s", cache_dir)

        # Store in Weaviate
        book_chapters = [f"{book_index}:{chapter_index}"]
        self._store_graph(
            graph_name, "chapter", final_entities, final_relationships, book_chapters
        )

        return graph_name

    def merge_chapters(  # noqa: PLR0912
        self,
        book_index: int,
        chapters: list[int],
        *,
        graph_name: str | None = None,
        base_graph: str | None = None,
        force: bool = False,
        profile: bool = False,
        max_concurrency: int = 5,
        skip_visualization: bool = False,
    ) -> str:
        """Extract chapters (cached) -> merge -> store as book graph.

        Returns graph_name (e.g. 'book3').
        """
        chapters = sorted(chapters)

        # Load base graph if provided
        base_entities = None
        base_relationships = None
        base_metadata: dict = {}
        base_chapters: list[int] = []

        if base_graph:
            base_graph_dir = Path(base_graph)
            has_json = (base_graph_dir / "entities.json").exists()
            has_csv = (base_graph_dir / "entities.csv").exists()
            if not has_json and not has_csv:
                msg = f"Base graph not found at {base_graph_dir}"
                raise FileNotFoundError(msg)
            base_entities, base_relationships, base_metadata = self._load_graph_files(
                base_graph_dir
            )
            base_chapters = base_metadata.get("chapters_included", [])
            logger.info(
                "Loaded base graph from %s: %d entities, %d relationships (chapters %s)",
                base_graph_dir,
                len(base_entities),
                len(base_relationships),
                base_chapters,
            )
            # Skip overlapping chapters
            overlap = set(chapters) & set(base_chapters)
            if overlap:
                logger.warning(
                    "Chapters %s already in base graph — they will be skipped",
                    sorted(overlap),
                )
                chapters = [ch for ch in chapters if ch not in overlap]
                if not chapters:
                    msg = "No new chapters to merge after removing overlaps"
                    raise ValueError(msg)

        all_chapters = sorted(set(base_chapters + chapters))

        # Determine graph output name
        if not graph_name:
            ch_range = f"{min(all_chapters)}-{max(all_chapters)}"
            graph_name = f"book{book_index}_ch{ch_range}"

        graph_dir = self._graph_output_dir(graph_name)
        logger.info(
            "KG extraction: book=%d chapters=%s -> %s",
            book_index,
            chapters,
            graph_dir,
        )

        # Phase 1: Per-chapter extraction (with caching)
        chapter_results: dict[
            int, tuple[list[NormalizedEntity], list[NormalizedRelationship]]
        ] = {}

        for ch_idx in chapters:
            ch_cache_dir = self._chapter_cache_dir(book_index, ch_idx)

            if not force:
                cached = self._load_chapter_cache(ch_cache_dir)
                if cached is not None:
                    entities, relationships = cached
                    logger.info(
                        "Chapter %d: loaded from cache (%d entities, %d relationships)",
                        ch_idx,
                        len(entities),
                        len(relationships),
                    )
                    chapter_results[ch_idx] = (entities, relationships)
                    continue

            # Run extraction for this chapter
            paragraphs = self._load_paragraphs(book_index, ch_idx)
            if not paragraphs:
                logger.error(
                    "No paragraphs found for book=%d chapter=%d, skipping",
                    book_index,
                    ch_idx,
                )
                continue

            logger.info(
                "Chapter %d: extracting from %d paragraphs...",
                ch_idx,
                len(paragraphs),
            )
            entities, relationships, llm_results = run_pipeline(
                paragraphs,
                self._config,
                profile=profile,
                max_concurrency=max_concurrency,
            )

            # Export to chapter cache
            export_results(
                entities,
                relationships,
                llm_results,
                ch_cache_dir,
                skip_visualization=skip_visualization,
            )
            chapter_results[ch_idx] = (entities, relationships)

        if not chapter_results:
            msg = "No chapter results available for merge"
            raise ValueError(msg)

        total_chapters = len(base_chapters) + len(chapter_results)
        if total_chapters < 2:
            msg = f"Need at least 2 chapters for merge, got {total_chapters}"
            raise ValueError(msg)

        # Phase 2: Cross-chapter merge
        logger.info("=== Phase 2: Cross-chapter merge ===")
        combined_entities, combined_rels, combined_llm, scaling_metrics = (
            run_cross_chapter_merge(
                chapter_results,
                self._config,
                profile=profile,
                max_concurrency=max_concurrency,
                base_entities=base_entities,
                base_relationships=base_relationships,
            )
        )

        # Phase 3: Export to graph directory
        export_results(
            combined_entities,
            combined_rels,
            combined_llm,
            graph_dir,
            skip_visualization=skip_visualization,
        )
        if scaling_metrics:
            df_metrics = pd.DataFrame(scaling_metrics)
            df_metrics.to_csv(graph_dir / "scaling_metrics.csv", index=False)

        # Save file metadata
        now = datetime.now(UTC).isoformat()
        metadata = {
            "book_index": book_index,
            "chapters_included": all_chapters,
            "entity_count": len(combined_entities),
            "relationship_count": len(combined_rels),
            "created_at": base_metadata.get("created_at", now),
            "updated_at": now,
            "config": {
                k: v
                for k, v in self._config.items()
                if k in ("similarity_threshold", "extraction_model", "merge_model")
            },
        }
        if base_graph:
            metadata["base_graph"] = str(base_graph)
        self._save_graph_metadata(graph_dir, metadata)

        # Store in Weaviate
        book_chapters = [f"{book_index}:{ch}" for ch in all_chapters]
        self._store_graph(
            graph_name, "book", combined_entities, combined_rels, book_chapters
        )

        logger.info("Done. Output written to %s", graph_dir)
        return graph_name

    def merge_graphs(
        self,
        graph_names: list[str],
        *,
        output_name: str,
        graph_type: str = "volume",
        profile: bool = False,
        max_concurrency: int = 5,
        skip_visualization: bool = False,
    ) -> str:
        """Merge N existing graphs from Weaviate into a new graph.

        Works for book->volume or any arbitrary merge. Returns graph_name.
        """
        # Load existing graphs from Weaviate
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
            # Gather book_chapters from graph metadata
            kg_graph = self.repositories.kg_graphs.find_by_name(gname)
            if kg_graph:
                all_book_chapters.extend(kg_graph.book_chapters)

        all_book_chapters = sorted(set(all_book_chapters))

        # Run cross-chapter merge (treating each graph as a "chapter")
        combined_entities, combined_rels, combined_llm, scaling_metrics = (
            run_cross_chapter_merge(
                graph_results,
                self._config,
                profile=profile,
                max_concurrency=max_concurrency,
            )
        )

        # Export to file
        graph_dir = self._graph_output_dir(output_name)
        export_results(
            combined_entities,
            combined_rels,
            combined_llm,
            graph_dir,
            skip_visualization=skip_visualization,
        )

        # Store in Weaviate
        self._store_graph(
            output_name, graph_type, combined_entities, combined_rels, all_book_chapters
        )

        logger.info("Done. Merged %d graphs into '%s'", len(graph_names), output_name)
        return output_name

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

    @staticmethod
    def _chapter_cache_dir(book_index: int, chapter_index: int) -> Path:
        return Path(f"output/kg/chapters/book{book_index}_ch{chapter_index}")

    @staticmethod
    def _graph_output_dir(graph_name: str) -> Path:
        return Path(f"output/kg/graphs/{graph_name}")

    @staticmethod
    def _save_graph_metadata(graph_dir: Path, metadata: dict) -> None:
        graph_dir.mkdir(parents=True, exist_ok=True)
        (graph_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

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

    @staticmethod
    def _load_chapter_cache(
        cache_dir: Path,
    ) -> tuple[list[NormalizedEntity], list[NormalizedRelationship]] | None:
        """Load cached chapter results from JSON (preferred) or CSV."""
        # Prefer JSON
        entities_json = cache_dir / "entities.json"
        relationships_json = cache_dir / "relationships.json"
        if entities_json.exists() and relationships_json.exists():
            entities = load_entities_json(entities_json)
            relationships = load_relationships_json(relationships_json)
            _populate_relationship_ids(entities, relationships)
            return entities, relationships

        # Fall back to CSV
        entities_csv = cache_dir / "entities.csv"
        relationships_csv = cache_dir / "relationships.csv"
        if not entities_csv.exists() or not relationships_csv.exists():
            return None

        df_entities = pd.read_csv(entities_csv)
        df_rels = pd.read_csv(relationships_csv)

        def _parse_list(val, sep=";"):
            if pd.notna(val) and str(val).strip():
                return [item.strip() for item in str(val).split(sep) if item.strip()]
            return []

        entities = []
        for _, row in df_entities.iterrows():
            aliases = _parse_list(row.get("aliases"))
            para_ids = _parse_list(row.get("source_paragraph_ids"))
            source_locations = [{"paragraph_id": pid} for pid in para_ids]

            chapters_str = row.get("source_chapters", "")
            if pd.notna(chapters_str) and str(chapters_str).strip():
                chapter_indices = [
                    int(c.strip()) for c in str(chapters_str).split(";") if c.strip()
                ]
                for loc in source_locations:
                    if chapter_indices:
                        loc["chapter_index"] = chapter_indices[0]

            entities.append(
                NormalizedEntity(
                    id=row["id"],
                    name=row["name"],
                    type=row["type"],
                    aliases=aliases,
                    description=str(row.get("description", ""))
                    if pd.notna(row.get("description"))
                    else "",
                    source_paragraph_ids=para_ids,
                    source_locations=source_locations,
                    occurrence_count=int(row.get("occurrence_count", 1)),
                    merged_from_ids=[],
                    relationship_ids=[],
                )
            )

        entity_lookup = {e.id: e for e in entities}
        name_to_entity = {e.name: e for e in entities}

        relationships = []
        for _, row in df_rels.iterrows():
            source_id = row.get("source_id", "")
            target_id = row.get("target_id", "")

            # Resolve from names if IDs missing
            if not source_id or (pd.notna(source_id) is False):
                source_entity = name_to_entity.get(row.get("source_name", ""))
                target_entity = name_to_entity.get(row.get("target_name", ""))
                if not source_entity or not target_entity:
                    continue
                source_id = source_entity.id
                target_id = target_entity.id

            rel = NormalizedRelationship(
                id=row["id"],
                source_id=str(source_id),
                target_id=str(target_id),
                source_entity_name=row.get(
                    "source_original", row.get("source_name", "")
                ),
                target_entity_name=row.get(
                    "target_original", row.get("target_name", "")
                ),
                relation_type=row["relation_type"],
                temporal_context=row.get("temporal_context")
                if pd.notna(row.get("temporal_context"))
                else None,
                paragraph_id=str(row.get("paragraph_id", "")),
                book_index=int(row["book_index"])
                if pd.notna(row.get("book_index")) and row.get("book_index") != ""
                else None,
                chapter_index=int(row["chapter_index"])
                if pd.notna(row.get("chapter_index")) and row.get("chapter_index") != ""
                else None,
                page=int(row["page"])
                if pd.notna(row.get("page")) and row.get("page") != ""
                else None,
            )
            relationships.append(rel)
            if rel.source_id in entity_lookup:
                entity_lookup[rel.source_id].relationship_ids.append(rel.id)
            if rel.target_id in entity_lookup:
                entity_lookup[rel.target_id].relationship_ids.append(rel.id)

        return entities, relationships

    @staticmethod
    def _load_graph_files(
        graph_dir: Path,
    ) -> tuple[list[NormalizedEntity], list[NormalizedRelationship], dict]:
        """Load a saved graph from files (JSON preferred, CSV fallback)."""
        if (graph_dir / "entities.json").exists():
            entities = load_entities_json(graph_dir / "entities.json")
            relationships = load_relationships_json(graph_dir / "relationships.json")
            _populate_relationship_ids(entities, relationships)
        else:
            result = KGIngestionService._load_chapter_cache(graph_dir)
            if result is None:
                msg = f"No entities.json or entities.csv found in {graph_dir}"
                raise FileNotFoundError(msg)
            entities, relationships = result

        metadata_path = graph_dir / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        return entities, relationships, metadata

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
                    temporal_context=rel.temporal_context or "",
                    paragraph_id=rel.paragraph_id,
                    book_index=rel.book_index or 0,
                    chapter_index=rel.chapter_index or 0,
                    page=rel.page or 0,
                )
                kg_relationships.append(kg_rel)

        # Batch insert (pass None vectors — Weaviate auto-vectorizes search_text)
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
            source_locations = [
                {"paragraph_id": pid} for pid in ke.source_paragraph_ids
            ]
            entities.append(
                NormalizedEntity(
                    id=ke.id,
                    name=ke.name,
                    type=ke.entity_type,
                    aliases=ke.aliases,
                    description=ke.description,
                    source_paragraph_ids=ke.source_paragraph_ids,
                    source_locations=source_locations,
                    occurrence_count=ke.occurrence_count,
                    merged_from_ids=[],
                    relationship_ids=[],
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
                temporal_context=kr.temporal_context,
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
        book_indices = set()
        source_book_chapters = set()
        source_pages = set()
        for loc in entity.source_locations:
            bi = loc.get("book_index")
            ci = loc.get("chapter_index")
            page = loc.get("page")
            if bi is not None:
                book_indices.add(bi)
            if bi is not None and ci is not None:
                source_book_chapters.add(f"{bi}:{ci}")
            if page is not None:
                source_pages.add(page)

        return KGEntity(
            graph_name=graph_name,
            name=entity.name,
            entity_type=entity.type,
            aliases=entity.aliases,
            description=entity.description,
            occurrence_count=entity.occurrence_count,
            book_indices=sorted(book_indices),
            source_book_chapters=sorted(source_book_chapters),
            source_pages=sorted(source_pages),
            source_paragraph_ids=entity.source_paragraph_ids,
            merged_from_count=len(entity.merged_from_ids),
            search_text=create_entity_text(entity),
        )
