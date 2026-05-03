"""Prompts and context formatting for the RAG agent."""

from history_book.data_models.entities import Paragraph

PROMPT_PREAMBLE = """You are a history expert assistant with access to "The Penguin History of the World" by J.M. Roberts and Odd Arne Westad.

You have a set of tools that retrieve passages and structured knowledge graph data from the book. Each tool's docstring explains when to use it."""

PROMPT_WORKFLOW = """WORKFLOW:
1. Use the available tools to retrieve relevant passages and/or knowledge graph entities
2. Review the results — if they are insufficient, call additional tools with refined queries
3. Once you have sufficient context, synthesize a comprehensive answer with inline citations in the format [Ch X, p. Y]"""

PROMPT_RULES = """IMPORTANT INSTRUCTIONS:
- Base your answer entirely on retrieved text excerpts and knowledge graph data
- Do NOT use any information from your training data or other sources
- Include inline citations [Ch X, p. Y] for every claim or piece of information drawn from passages
- Provide historical context and explanation where appropriate
- Write as much as needed to fully answer the question — there are no length limits
- If the book doesn't contain relevant information after searching, clearly state: "I could not find information about this topic in 'The Penguin History of the World'.\""""

FINAL_ITERATION_SUFFIX = """

**You have reached the maximum number of search iterations. You must provide your final answer now.**
If the excerpts retrieved so far are sufficient, answer with inline citations.
If the excerpts are insufficient or irrelevant, state: "I could not find information about this topic in 'The Penguin History of the World'.\""""


# Cross-tool guidance — only included when both tools in the key set are enabled.
CROSS_TOOL_NOTES: list[tuple[set[str], str]] = [
    (
        {"search_entities", "search_book"},
        "- Prefer search_entities when the user names a specific person, place, event, "
        "polity, or concept; use search_book for thematic or descriptive queries.",
    ),
    (
        {"get_entity_detail", "get_paragraphs"},
        "- After get_entity_detail, pass source_paragraph_ids to get_paragraphs to "
        "ground claims in the source text.",
    ),
    (
        {"search_entities", "get_entity_detail"},
        "- search_entities returns IDs only; call get_entity_detail to see "
        "descriptions and relationships for an entity of interest.",
    ),
    (
        {"query_relationships_by_period", "get_entity_detail"},
        "- Use query_relationships_by_period for era-scoped questions; follow up "
        "with get_entity_detail on entities that look interesting.",
    ),
]


def build_system_prompt(enabled_tool_names: set[str]) -> str:
    """Assemble the system prompt for the currently-enabled tool set."""
    notes = [text for names, text in CROSS_TOOL_NOTES if names <= enabled_tool_names]
    sections = [PROMPT_PREAMBLE]
    if notes:
        sections.append("TOOL SELECTION NOTES:\n" + "\n".join(notes))
    sections.append(PROMPT_WORKFLOW)
    sections.append(PROMPT_RULES)
    return "\n\n".join(sections)


# Backwards-compatible default prompt (full toolset). Kept for tests/legacy callers.
AGENT_SYSTEM_PROMPT = build_system_prompt(
    {
        "search_book",
        "search_entities",
        "get_entity_detail",
        "get_paragraphs",
        "query_relationships_by_period",
    }
)


def format_excerpts_for_llm(paragraphs: list[Paragraph]) -> str:
    """Format paragraphs with chapter/page headers for LLM-readable citations."""
    if not paragraphs:
        return ""
    parts = []
    for para in paragraphs:
        parts.append(f"[Chapter {para.chapter_index}, Page {para.page}]\n{para.text}")
    return "\n\n".join(parts)
