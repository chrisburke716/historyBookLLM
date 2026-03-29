"""KG temporal context parsing chain.

Uses a lightweight LLM to parse free-text temporal expressions (e.g., "753 BC",
"sixth century BC") into structured year data for timeline construction.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------


class TemporalParsed(BaseModel):
    start_year: int | None = Field(
        default=None,
        description="Start year as integer. Negative for BC (e.g., -753 for 753 BC). Null if not parseable.",
    )
    end_year: int | None = Field(
        default=None,
        description="End year for ranges. Null for point-in-time events.",
    )
    precision: str = Field(
        default="approximate",
        description="One of: year, decade, century, approximate",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TEMPORAL_PARSE_PROMPT = """Parse this temporal expression into structured year data.
Return start_year (negative for BC), end_year (null if point-in-time), and precision.

Relationship: {source_entity} --[{relation_type}]--> {target_entity}
Temporal expression: {temporal_context}

Examples:
- "753 BC" → start_year=-753, end_year=null, precision="year"
- "264-241 BC" → start_year=-264, end_year=-241, precision="year"
- "sixth century BC" → start_year=-600, end_year=-501, precision="century"
- "early second century AD" → start_year=100, end_year=130, precision="approximate"
- "after 387" → start_year=387, end_year=null, precision="approximate"
- "around 750 BC" → start_year=-750, end_year=null, precision="approximate"
- "after the Roman victory" → start_year=null, end_year=null, precision="approximate"

If the expression contains no discernible date or time period, return start_year=null."""


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------


def create_temporal_chain(config: dict):
    """Create a temporal parsing chain: relationship fields -> TemporalParsed.

    Args:
        config: Pipeline config with keys temporal_model, temporal_temperature.

    Returns:
        Runnable that takes {source_entity, relation_type, target_entity,
        temporal_context} and returns TemporalParsed.
    """
    llm = ChatOpenAI(
        model=config["temporal_model"],
        temperature=config.get("temporal_temperature", 0.0),
        request_timeout=30,
    )
    prompt = ChatPromptTemplate.from_template(TEMPORAL_PARSE_PROMPT)
    return prompt | llm.with_structured_output(TemporalParsed)
