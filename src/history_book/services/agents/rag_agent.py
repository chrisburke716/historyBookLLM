"""RAG agent graph — builds and exposes the compiled LangGraph agent."""

import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command

from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError
from history_book.services.agents.context import AgentContext
from history_book.services.agents.prompts import (
    AGENT_SYSTEM_PROMPT,
    FINAL_ITERATION_SUFFIX,
)
from history_book.services.agents.state import AgentState
from history_book.services.agents.tools import TOOLS

logger = logging.getLogger(__name__)


def _build_llm(llm_config: LLMConfig) -> BaseChatModel:
    """Construct a chat model from LLMConfig using init_chat_model."""
    model_id = f"{llm_config.provider}:{llm_config.model_name}"
    kwargs = dict(temperature=llm_config.temperature)
    if llm_config.api_key:
        kwargs["api_key"] = llm_config.api_key
    if llm_config.max_tokens:
        kwargs["max_tokens"] = llm_config.max_tokens
    if llm_config.api_base:
        kwargs["base_url"] = llm_config.api_base
    kwargs.update(llm_config.provider_kwargs)
    return init_chat_model(model_id, **kwargs)


def _count_tool_iterations(messages: list[BaseMessage]) -> int:
    """Count how many times the LLM has called tools so far in this turn."""
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


def _previous_queries(messages: list[BaseMessage]) -> list[str]:
    """Extract search queries from previous tool calls to avoid repeats."""
    queries = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args = (
                    tc.get("args", {})
                    if isinstance(tc, dict)
                    else getattr(tc, "args", {})
                )
                if "query" in args:
                    queries.append(args["query"])
    return queries


def _build_system_message(messages: list[BaseMessage], is_final: bool) -> SystemMessage:
    content = AGENT_SYSTEM_PROMPT
    prev = _previous_queries(messages)
    if prev:
        content += (
            "\n\nPrevious search queries (do not repeat these exactly):\n"
            + "\n".join(f'- "{q}"' for q in prev)
        )
    if is_final:
        content += FINAL_ITERATION_SUFFIX
    return SystemMessage(content=content)


async def agent_node(
    state: AgentState, runtime: Runtime[AgentContext]
) -> Command[Literal["tools", "__end__"]]:
    """
    Core agent node: invoke LLM, route to tools or END.

    On each call:
    - Count prior tool iterations from message history
    - If at the iteration cap, strip tools so LLM must give a final answer
    - Return Command combining the new AI message + routing decision
    """
    ctx = runtime.context
    messages = state["messages"]

    iterations = _count_tool_iterations(messages)
    is_final = iterations >= ctx.max_tool_iterations

    llm = _build_llm(ctx.llm_config)
    llm_to_use = llm if is_final else llm.bind_tools(TOOLS)

    system = _build_system_message(messages, is_final)
    prompt = [system] + list(messages)

    try:
        response = await llm_to_use.ainvoke(prompt)
    except Exception as e:
        raise LLMError(f"Agent LLM call failed: {e}") from e

    if response.tool_calls and not is_final:
        logger.info(
            f"Agent calling {len(response.tool_calls)} tool(s) "
            f"(iteration {iterations + 1}/{ctx.max_tool_iterations})"
        )
        goto = "tools"
    else:
        logger.info(
            f"Agent providing final answer after {iterations} tool iteration(s)"
        )
        goto = END

    return Command(update={"messages": [response]}, goto=goto)


def build_rag_agent() -> CompiledStateGraph:
    """
    Compile and return the RAG agent graph.

    Graph structure:
        START → agent → tools → agent → ... → END
                      ↘ END (when no tool calls or iteration cap reached)

    No checkpointer — history is passed explicitly from Weaviate on each invocation.
    MemorySaver would duplicate messages since lc_history already contains the full
    conversation; Weaviate is the durable store.
    """
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_edge("tools", "agent")
    return g.compile()
