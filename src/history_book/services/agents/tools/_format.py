"""Shared formatters for KG-aware tool outputs."""

from history_book.api.models.kg_models import RelationshipSummary
from history_book.data_models.kg_entities import KGRelationship


def format_year_range(start_year: int | None, end_year: int | None) -> str:
    """Render a year range as `[YYYY–YYYY]`, `[YYYY]`, or `""` when unknown."""
    if start_year is None and end_year is None:
        return ""
    if start_year is not None and end_year is not None:
        if start_year == end_year:
            return f"[{start_year}]"
        return f"[{start_year}–{end_year}]"
    if start_year is not None:
        return f"[{start_year}–?]"
    return f"[?–{end_year}]"


def format_relationship_summary(r: RelationshipSummary) -> str:
    """Render a 1-hop RelationshipSummary as a single LLM-readable line."""
    arrow = "→" if r.direction == "outgoing" else "←"
    desc = f" — {r.description}" if r.description else ""
    cite = f" (Ch {r.chapter_index}, p. ?)" if r.chapter_index is not None else ""
    return (
        f"{arrow} [{r.relation_type}] {r.other_entity_name} "
        f"(id={r.other_entity_id}){desc}{cite}"
    )


def format_relationship(r: KGRelationship) -> str:
    """Render a full KGRelationship row (used by neighborhood + period queries)."""
    years = format_year_range(r.start_year, r.end_year)
    years_part = f"{years} " if years else ""
    desc = f" — {r.description}" if r.description else ""
    cite = f" (Ch {r.chapter_index}, p. {r.page})"
    return (
        f"{years_part}{r.source_entity_name} (id={r.source_entity_id}) "
        f"—[{r.relation_type}]→ "
        f"{r.target_entity_name} (id={r.target_entity_id}){desc}{cite}"
    )
