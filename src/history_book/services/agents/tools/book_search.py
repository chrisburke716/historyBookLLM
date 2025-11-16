"""Book search tool for retrieving information from 'The Penguin History of the World'.

This tool performs retrieval from the book's vector database and generates
a formatted response with proper citations.
"""

import logging
from typing import Annotated

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from history_book.config.graph_config import GraphConfig
from history_book.database.repositories import BookRepositoryManager
from history_book.llm import create_chat_model

from .prompts import BOOK_SEARCH_PROMPT

logger = logging.getLogger(__name__)


@tool
def search_book(
    query: Annotated[str, "The search query to find relevant passages in the book"],
) -> str:
    """Search 'The Penguin History of the World' for relevant information.

    Use this tool when you need to retrieve specific information from the book
    to answer the user's question. The tool will search the book's content and
    return relevant passages with proper chapter and page citations.

    This tool should be your primary source for historical information when
    answering questions about topics covered in the book.

    Args:
        query: The search query describing what information to find in the book.

    Returns:
        A formatted response with relevant information and citations from the book.
    """
    logger.info(f"Book search tool called with query: {query[:100]}")

    try:
        # 1. Validate query
        if not query or len(query.strip()) < 3:
            return "Please provide a more specific question."

        # 2. Initialize configuration and services
        graph_config = GraphConfig()
        repository_manager = BookRepositoryManager()

        # 3. Determine retrieval parameters (use book-specific overrides if set)
        max_results = (
            graph_config.book_tool_max_results or graph_config.tool_max_results
        )
        min_similarity = (
            graph_config.book_tool_min_similarity or graph_config.tool_min_similarity
        )

        logger.info(
            f"Retrieving with max_results={max_results}, min_similarity={min_similarity}"
        )

        # 4. Retrieve relevant paragraphs
        paragraphs = repository_manager.paragraphs.similarity_search_by_text(
            query_text=query, limit=max_results, threshold=min_similarity
        )

        logger.info(f"Retrieved {len(paragraphs)} paragraphs")

        # 5. Handle no results
        if not paragraphs:
            return "I could not find relevant information about this topic in 'The Penguin History of the World'."

        # 6. Handle low quality results
        if len(paragraphs) < 3:
            logger.warning(
                f"Only retrieved {len(paragraphs)} paragraphs, results may be incomplete"
            )

        # 7. Format context for prompt
        context_str = "\n\n".join(
            [
                f"[Chapter {p.chapter_number}, Page {p.page_number}]\n{p.content}"
                for p in paragraphs
            ]
        )

        logger.debug(f"Context length: {len(context_str)} characters")

        # 8. Create LLM using factory (provider-agnostic)
        llm = create_chat_model()

        # 9. Generate answer with citations
        prompt = PromptTemplate.from_template(BOOK_SEARCH_PROMPT)
        chain = prompt | llm

        logger.info("Invoking LLM to generate answer with citations")

        response = chain.invoke({"context": context_str, "query": query})

        # 10. Extract text from response
        answer = response.content if hasattr(response, "content") else str(response)

        # 11. Validate answer is not empty
        if not answer or answer.strip() == "":
            logger.warning("LLM returned empty answer")
            return "I was unable to formulate an answer based on the retrieved content."

        logger.info(f"Generated answer with {len(answer)} characters")
        logger.debug(f"Answer preview: {answer[:200]}...")

        return answer

    except ImportError as e:
        logger.error(f"Missing dependency: {e}", exc_info=True)
        return "The book search service is currently unavailable due to a missing dependency."
    except Exception as e:
        logger.error(f"Book search failed: {e}", exc_info=True)
        return "I encountered an error searching the book. Please try again."
