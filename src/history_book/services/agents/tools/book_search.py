"""Book search tool for retrieving information from 'The Penguin History of the World'.

This tool performs retrieval from the book's vector database and generates
a formatted response with proper citations.
"""

from typing import Annotated

from langchain_core.tools import tool


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
    # STUB IMPLEMENTATION - Returns mock data for testing tool infrastructure
    # Full implementation with actual retrieval + generation will be added later

    return f"""Based on 'The Penguin History of the World':

The query "{query}" relates to important historical developments covered in the book.

[Chapter 12, p. 234-236] This period saw significant changes in political structures
and trade relationships across the Mediterranean region.

[Chapter 15, p. 289] Archaeological evidence suggests complex social hierarchies
and sophisticated administrative systems.

[Chapter 18, p. 345] Historical records indicate extensive cultural exchange and
the spread of technological innovations during this era.

Note: This is mock data for testing. Full implementation pending."""
