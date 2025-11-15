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

    # Tool calling configuration
    enable_tools: bool = True
    enabled_tools: list[str] = ["book_search"]  # List of tool names to enable
    max_tool_iterations: int = 3  # Maximum tool call loops to prevent infinite iterations

    # Future: reasoning configuration
    # enable_planning: bool = False
    # enable_reflection: bool = False
    # max_reasoning_iterations: int = 3
