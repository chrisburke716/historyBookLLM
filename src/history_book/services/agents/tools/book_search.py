"""Book search tool for the RAG agent."""

import logging
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from history_book.services.agents.context import AgentContext
from history_book.services.agents.prompts import format_excerpts_for_llm

logger = logging.getLogger(__name__)


@tool
def search_book(
    query: Annotated[str, "Search query to find relevant passages in the book"],
    runtime: ToolRuntime[AgentContext],
) -> Command:
    """Search 'The Penguin History of the World' for relevant passages.

    Use this tool when you need to retrieve specific historical information from
    the book to answer the user's question. Returns relevant passages with
    chapter and page metadata for inline citations.
    """
    logger.info(f"Book search: {query[:100]}")

    if not query or len(query.strip()) < 3:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Please provide a more specific search query.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    try:
        repo = runtime.context.repository_manager
        ctx = runtime.context

        results = repo.paragraphs.similarity_search_by_text(
            query_text=query,
            limit=ctx.tool_max_results,
            threshold=ctx.tool_min_similarity,
        )

        if not results:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="No relevant passages found in the book for this query. Try a different search term.",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        paragraphs = [p for p, _ in results]
        formatted = format_excerpts_for_llm(paragraphs)
        logger.info(f"Book search returned {len(paragraphs)} passages")

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Found {len(paragraphs)} relevant passages:\n\n{formatted}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "retrieved_paragraphs": paragraphs,
            }
        )

    except Exception as e:
        logger.error(f"Book search failed: {e}", exc_info=True)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="An error occurred while searching the book. Please try again.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
