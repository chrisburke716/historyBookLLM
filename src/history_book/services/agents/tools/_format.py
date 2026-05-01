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
    """Render a 1-hop RelationshipSummary as a compact two-line block."""
    arrow = "→" if r.direction == "outgoing" else "←"
    head = f"{arrow} [{r.relation_type}] {r.other_entity_name}"
    meta = (
        f"id={r.other_entity_id} · Bk {r.book_index}, Ch {r.chapter_index}, p. {r.page}"
    )
    if r.description:
        return f"{head}\n    {r.description}\n    ({meta})"
    return f"{head}\n    ({meta})"


def format_relationship(r: KGRelationship) -> str:
    """Render a full KGRelationship row as a compact two-line block."""
    years = format_year_range(r.start_year, r.end_year)
    years_part = f"{years} " if years else ""
    head = (
        f"{years_part}{r.source_entity_name} —[{r.relation_type}]→ "
        f"{r.target_entity_name}"
    )
    meta = (
        f"src={r.source_entity_id} · tgt={r.target_entity_id} · "
        f"Bk {r.book_index}, Ch {r.chapter_index}, p. {r.page}"
    )
    if r.description:
        return f"{head}\n    {r.description}\n    ({meta})"
    return f"{head}\n    ({meta})"
