"""Shared FastAPI dependency factories for KG routes."""

from functools import lru_cache

from history_book.services.kg_metrics_service import KGMetricsService
from history_book.services.kg_service import KGService


@lru_cache
def get_kg_service() -> KGService:
    """Get a cached KGService instance (shared across kg and kg_metrics routes)."""
    return KGService()


@lru_cache
def get_kg_metrics_service() -> KGMetricsService:
    """Get a cached KGMetricsService instance sharing the same KGService."""
    return KGMetricsService(get_kg_service())
