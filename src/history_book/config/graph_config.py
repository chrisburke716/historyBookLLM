"""Configuration for LangGraph-based agent functionality."""

from pydantic_settings import BaseSettings


class GraphConfig(BaseSettings):
    """
    Configuration for LangGraph-based agent.

    This configuration controls the behavior of the graph RAG agent,
    including checkpointing strategy, feature flags, and retrieval parameters.
    """

    # Checkpointer settings
    checkpointer_type: str = "memory"  # "memory" | "sqlite" (future)

    # Feature flags
    enable_streaming: bool = True
    enable_tracing: bool = True
    enable_graph_visualization: bool = True

    # Retrieval configuration
    max_retrieval_results: int = 40
    min_retrieval_results: int = 5
    similarity_threshold: float = 0.4

    # Future: tool calling configuration
    # enabled_tools: list[str] = []

    # Future: reasoning configuration
    # enable_planning: bool = False
    # enable_reflection: bool = False
    # max_reasoning_iterations: int = 3
