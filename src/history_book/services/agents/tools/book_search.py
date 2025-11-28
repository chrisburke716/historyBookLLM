"""Book search tool for retrieving information from 'The Penguin History of the World'.

This tool performs retrieval from the book's vector database and returns
relevant paragraphs as structured data for the agent to synthesize.
"""

import json
import logging
from typing import Annotated

from langchain_core.tools import tool

from history_book.config.graph_config import GraphConfig
from history_book.database.config.database_config import WeaviateConfig
from history_book.database.repositories import BookRepositoryManager

logger = logging.getLogger(__name__)


@tool
def search_book(
    query: Annotated[str, "The search query to find relevant passages in the book"],
) -> str:
    """Search 'The Penguin History of the World' for relevant information.

    Use this tool when you need to retrieve specific information from the book
    to answer the user's question. The tool searches the book's content and
    returns relevant passages with chapter and page metadata.

    This tool should be your primary source for historical information when
    answering questions about topics covered in the book.

    Args:
        query: The search query describing what information to find in the book.

    Returns:
        JSON string containing relevant excerpts from the book with metadata.
        Format: {"excerpts": [{"text": "...", "chapter": X, "page": Y, ...}], ...}
    """
    logger.info(f"Book search tool called with query: {query[:100]}")

    try:
        # 1. Validate query
        if not query or len(query.strip()) < 3:
            return json.dumps(
                {
                    "error": "Please provide a more specific question.",
                    "excerpts": [],
                    "query": query,
                    "num_results": 0,
                }
            )

        # 2. Initialize configuration and services
        graph_config = GraphConfig()
        weaviate_config = WeaviateConfig.from_environment()
        repository_manager = BookRepositoryManager(weaviate_config)

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

        # 4. Retrieve relevant paragraphs (returns list of tuples: (Paragraph, score))
        search_results = repository_manager.paragraphs.similarity_search_by_text(
            query_text=query, limit=max_results, threshold=min_similarity
        )

        logger.info(f"Retrieved {len(search_results)} paragraphs")

        # 5. Handle no results
        if not search_results:
            return json.dumps(
                {
                    "message": "No relevant information found in the book for this query.",
                    "excerpts": [],
                    "query": query,
                    "num_results": 0,
                }
            )

        # 6. Handle low quality results
        if len(search_results) < 3:
            logger.warning(
                f"Only retrieved {len(search_results)} paragraphs, results may be incomplete"
            )

        # 7. Format results as structured data
        excerpts = []
        for paragraph, score in search_results:
            excerpt = {
                "id": paragraph.id,
                "text": paragraph.text,
                "chapter": paragraph.chapter_index,
                "page": paragraph.page,
                "book": paragraph.book_index,
                "similarity_score": float(score) if score is not None else None,
            }
            excerpts.append(excerpt)

        logger.info(f"Returning {len(excerpts)} excerpts from the book")

        return json.dumps(
            {
                "excerpts": excerpts,
                "query": query,
                "num_results": len(excerpts),
            }
        )

    except ImportError as e:
        logger.error(f"Missing dependency: {e}", exc_info=True)
        return json.dumps(
            {
                "error": "The book search service is currently unavailable due to a missing dependency.",
                "excerpts": [],
                "query": query,
                "num_results": 0,
            }
        )
    except Exception as e:
        logger.error(f"Book search failed: {e}", exc_info=True)
        return json.dumps(
            {
                "error": "An error occurred while searching the book. Please try again.",
                "excerpts": [],
                "query": query,
                "num_results": 0,
            }
        )
