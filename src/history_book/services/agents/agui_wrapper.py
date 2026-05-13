"""AG-UI adapter around the compiled RAG graph.

`LangGraphAgent` from `ag-ui-langgraph` translates a `CompiledStateGraph`'s
`astream_events` output into AG-UI events over SSE. We subclass it to fix
three gaps between the stock adapter and our graph:

1. `get_stream_kwargs` — base detects context support via signature
   inspection; LangGraph 1.1.x exposes `context` only via **kwargs, so the
   check fails and our `AgentContext` is silently dropped. We force-pass it.
2. `langgraph_default_merge_state` — base spreads client-posted state into
   the graph input. We strip backend-owned keys (see GRAPH_OWNED_STATE_KEYS)
   so the frontend can't echo them back.
3. `get_schema_keys` — base introspects context/config schemas, which raise
   on our `AgentContext` (live service handles aren't JSON-schemable). The
   base's catch-all fallback then discards input/output keys too. We compute
   input/output independently and skip config/context entirely.
"""

from typing import Any

from ag_ui_langgraph import LangGraphAgent
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from history_book.services.agents.context import AgentContext

# State keys the graph owns exclusively (backend-write, frontend-read).
#
# AG-UI's state model assumes state is a shared, bidirectional workspace —
# both the agent and the frontend can read and write it, and CopilotKit
# round-trips it on every turn (STATE_SNAPSHOT out, RunAgentInput.state back
# in). For accumulators like `retrieved_paragraphs` that the frontend only
# observes (citation chips), the round-trip is pure echo: the client posts
# back JSON-serialized copies of paragraphs we already have in the
# checkpointer. Letting them merge in pollutes graph state with dict-shaped
# entries that break the typed reducer.
GRAPH_OWNED_STATE_KEYS = ("retrieved_paragraphs",)


class HistoryBookLangGraphAgent(LangGraphAgent):
    """Adapts `LangGraphAgent` to our graph; see module docstring for details.

    Clone per request — the parent stores active-run state on the instance,
    so sharing one wrapper across concurrent runs corrupts that state.
    """

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
        # Stashed for get_stream_kwargs to inject into each run.
        self._agent_context = agent_context

    def clone(self):
        # Parent's clone() doesn't know about agent_context; reimplement so
        # it carries through.
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
        # Replaces parent entirely. `astream_events` forwards `context=` via
        # **kwargs to the Pregel runtime, which constructs Runtime[T] for
        # our nodes — that's how tools get `runtime.context.repository_manager`
        # etc.
        kwargs = dict(input=input, subgraphs=subgraphs, version=version)
        if config:
            kwargs["config"] = config
        if fork:
            kwargs.update(fork)
        kwargs["context"] = self._agent_context
        return kwargs

    def langgraph_default_merge_state(
        self, state: dict, messages: list[BaseMessage], input: Any
    ) -> dict:
        # Let the parent do its work (message dedup, tool-call repair,
        # orphan ToolMessage handling), then surgically drop our keys.
        merged = super().langgraph_default_merge_state(state, messages, input)
        for key in GRAPH_OWNED_STATE_KEYS:
            merged.pop(key, None)
        return merged

    def get_schema_keys(self, config: RunnableConfig):
        # `_keys` swallows per-call failures so a broken context schema
        # doesn't void successfully-computed input/output keys — which is
        # what the parent's outer try/except does.
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
    """Construct a wrapper. Keeps the service-layer import surface to one symbol."""
    return HistoryBookLangGraphAgent(
        name=name, graph=graph, agent_context=agent_context, config=config
    )
