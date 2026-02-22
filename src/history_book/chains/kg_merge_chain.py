"""KG entity merge decision chain.

Uses structured outputs to determine if two entities refer to the same
historical entity and, if so, produce a merged representation.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class MergedEntity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class EntityMergeDecision(BaseModel):
    reasoning: str = Field(description="Brief explanation of the decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    should_merge: bool = Field(
        description="True if entities refer to the same historical entity"
    )
    merged_entity: MergedEntity | None = Field(
        default=None,
        description="The merged entity if should_merge=True, otherwise None",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ENTITY_MERGE_PROMPT = """You are an expert historian analyzing entity mentions from "The Penguin History of the World".

Given two entities extracted from different paragraphs, determine if they refer to the SAME historical entity.
This is an entity normalization task as part of knowledge graph construction. The goal is to merge duplicate entities while maintaining distinct but related entities separately.

**Entity 1:**
Name: {entity1_name}
Type: {entity1_type}
Aliases: {entity1_aliases}
Description: {entity1_description}

**Entity 2:**
Name: {entity2_name}
Type: {entity2_type}
Aliases: {entity2_aliases}
Description: {entity2_description}

**Entity types**: person, polity, place, event

**Instructions:**
1. Determine if these refer to the SAME historical entity
    - Same here means strictly identical entities, not just similar or related.
    - Mergeable examples:
        - "Octavian" and "Augustus" (same person, different names)
        - "Roman Legions" and "Roman Army" (same organization)
        - "Roman Republic" and "Rome" (same political entity)
    - Non-mergeable examples:
        - Different people with same last name (e.g., "Julius Caesar" vs "Augustus Caesar")
        - Same place in different contexts (e.g., "Rome" the city vs "Rome" the empire)
        - Related political and geographical entities (e.g., "Roman Empire" vs "Italy")
        - Different entity types (e.g., "Punic Wars" event vs "Carthage" polity)
2. If they should be merged:
   - Choose the most canonical/common name
   - Write a consolidated description (combine key information, ~2-3 sentences)
   - Merge aliases (include both original names if not already aliases)
"""


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------


def create_merge_chain(config: dict):
    """Create a merge decision chain: entity pair fields -> EntityMergeDecision.

    Args:
        config: Pipeline config with keys merge_model, merge_temperature,
                and optional reasoning_effort.

    Returns:
        Runnable that takes {entity1_name, entity1_type, entity1_aliases,
        entity1_description, entity2_name, entity2_type, entity2_aliases,
        entity2_description} and returns EntityMergeDecision.
    """
    llm = ChatOpenAI(
        model=config["merge_model"],
        temperature=config["merge_temperature"],
        reasoning_effort=config.get("reasoning_effort"),
        request_timeout=60,
    )
    prompt = ChatPromptTemplate.from_template(ENTITY_MERGE_PROMPT)
    return prompt | llm.with_structured_output(EntityMergeDecision)
