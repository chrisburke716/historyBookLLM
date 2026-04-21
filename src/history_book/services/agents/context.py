"""Runtime context for the RAG agent graph."""

from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """
    Ephemeral runtime config injected at invocation time via context=.

    Flows through nodes and tools via Runtime[AgentContext] / ToolRuntime[AgentContext].
    Not persisted in checkpoints — reconstruct each call from service-level config.
    """

    llm_config: object  # LLMConfig — typed as object to avoid circular import
    repository_manager: object  # BookRepositoryManager
    max_tool_iterations: int = 3
    tool_max_results: int = 40
    tool_min_similarity: float = 0.4
    extra_tool_kwargs: dict = field(default_factory=dict)
