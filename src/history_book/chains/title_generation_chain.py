"""
Title generation chain for chat sessions.

Uses LCEL to create concise, topic-focused titles from conversation history.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

TITLE_GENERATION_PROMPT = """You are an expert at creating concise, informative chat session titles.

Review the FULL conversation history below and generate a clear, descriptive title that captures the main topic.

CRITICAL INSTRUCTIONS:
- **Weight recent messages MORE HEAVILY** - the most recent exchanges are more important for determining the current topic
- Focus on the general topic, not specific details from individual messages
- Maximum 100 characters
- Use title case (e.g., "The Fall of the Roman Empire")
- Be specific but not overly detailed
- Avoid phrases like "Chat about" or "Discussion on"

EXAMPLES:
Good: "Ancient Egyptian Pyramids and Construction Methods"
Bad: "User asks about pyramids"
Good: "French Revolution: Causes and Key Events"
Bad: "Discussion on various historical topics"

CONVERSATION HISTORY (most recent at bottom):
{conversation}

Generate only the title, nothing else."""


def create_title_generation_chain(chat_model: BaseChatModel):
    """
    Create LCEL chain for title generation.

    Args:
        chat_model: LangChain chat model to use

    Returns:
        Runnable chain that takes {"conversation": str} and returns title string
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TITLE_GENERATION_PROMPT),
        ]
    )

    return prompt | chat_model | StrOutputParser()
