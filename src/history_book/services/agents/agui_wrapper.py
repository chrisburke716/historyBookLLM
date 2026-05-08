"""AG-UI adapter around the compiled RAG graph.

`LangGraphAgent` from `ag-ui-langgraph` wraps a `CompiledStateGraph` and emits
AG-UI events. Two subclass overrides are needed for our graph:

1. `get_stream_kwargs` — force-pass `AgentContext` to `astream_events`. The
   base implementation drops `context` because LangGraph doesn't expose it as
   a named parameter (it accepts it via **kwargs), so the base's signature
   check returns False.
2. `get_schema_keys` — skip the base's `config_schema()` / `context_schema()`
   introspection. Both raise on our graph because `AgentContext` carries
   non-Pydantic services (BookRepositoryManager, KGService) that can't be
   JSON-schema-serialized. The base's outer try/except catches the failure
   but discards the successfully-computed input/output keys, which strips
   `retrieved_paragraphs` from STATE_SNAPSHOT events.
"""

from typing import Any

from ag_ui_langgraph import LangGraphAgent
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from history_book.services.agents.context import AgentContext


class HistoryBookLangGraphAgent(LangGraphAgent):
    def __init__(
        self,
        *,
        name: str,
        graph: CompiledStateGraph,
        agent_context: AgentContext,
        description: str | None = None,
        config: RunnableConfig | dict | None = None,
    ):
        super().__init__(name=name, graph=graph, description=description, config=config)
        self._agent_context = agent_context

    def clone(self):
        return type(self)(
            name=self.name,
            graph=self.graph,
            agent_context=self._agent_context,
            description=self.description,
            config=dict(self.config) if self.config else None,
        )

    def get_stream_kwargs(
        self,
        input: Any,
        subgraphs: bool = False,
        version: str = "v2",
        config: RunnableConfig | None = None,
        context: dict[str, Any] | None = None,
        fork: Any = None,
    ) -> dict[str, Any]:
        kwargs = dict(input=input, subgraphs=subgraphs, version=version)
        if config:
            kwargs["config"] = config
        if fork:
            kwargs.update(fork)
        kwargs["context"] = self._agent_context
        return kwargs

    def get_schema_keys(self, config: RunnableConfig):
        def _keys(fn):
            try:
                schema = fn()
                return (
                    list(schema.get("properties", {}).keys())
                    if isinstance(schema, dict)
                    else []
                )
            except Exception:
                return []

        input_keys = _keys(self.graph.get_input_jsonschema)
        output_keys = _keys(self.graph.get_output_jsonschema)
        return {
            "input": [*input_keys, *self.constant_schema_keys],
            "output": [*output_keys, *self.constant_schema_keys],
            "config": [],
            "context": [],
        }


def build_agui_wrapper(
    *,
    name: str,
    graph: CompiledStateGraph,
    agent_context: AgentContext,
    config: RunnableConfig | dict | None = None,
) -> HistoryBookLangGraphAgent:
    return HistoryBookLangGraphAgent(
        name=name, graph=graph, agent_context=agent_context, config=config
    )
