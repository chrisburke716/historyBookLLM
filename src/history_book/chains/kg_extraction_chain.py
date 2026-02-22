"""KG entity/relationship extraction chain.

Uses structured outputs to extract entities and relationships from paragraph text.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class Relationship(BaseModel):
    source_entity: str
    relation_type: str
    target_entity: str
    temporal_context: str | None = None


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]
    paragraph_id: str


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are analyzing text from "The Penguin History of the World".

Extract only the most historically significant entities and relationships from the provided paragraph. Focus on major states, leaders, regions, and pivotal events. Typically extract 3-6 entities per paragraph.

**ENTITY TYPES** (use exactly these):
- person: Major historical figures — rulers, generals, political leaders (e.g., Augustus, Caesar, Hannibal)
- polity: States, empires, peoples, organizations, political bodies (e.g., Rome, Etruscans, Senate, Roman Republic)
- place: Major cities, regions, bodies of water (e.g., Italy, Carthage, Mediterranean)
- event: Wars, revolts, reforms, conquests, pivotal moments (e.g., Punic Wars, revolt of the Latin cities)

**RELATIONSHIP TYPES** (use exactly one of these):
- ruled: A person or polity governed a place or polity
- conquered: Military takeover of a place or polity
- fought: Armed conflict without outright conquest
- allied_with: Formal alliance or cooperation
- succeeded: One leader/polity followed another in power
- revolted_against: Rebellion or uprising against authority
- influenced: Cultural, political, or intellectual impact
- part_of: Geographic or organizational membership (e.g., Sicily part_of Roman Republic)
- founded: Established or created
- evolved_into: Political transformation (e.g., Roman Republic evolved_into Roman Empire)
- participated_in: Connects actors to event entities (e.g., Rome participated_in Punic Wars)

**IMPORTANT GUIDELINES**:
1. Extract entities FROM THIS PARAGRAPH ONLY — do not use external knowledge
2. Be highly selective — only major historical actors, places, and events
3. Extract relationships that are EXPLICITLY STATED in the text
4. Include aliases if the entity is referred to by multiple names (e.g., "Octavian" also called "Augustus")
5. Do NOT extract dates or time periods as entities — instead, include them as temporal_context on relationships
6. Relationships MUST reference exact entity names from your entities list
7. Only extract entities that participate in at least one relationship

**DO NOT EXTRACT**:
- Unnamed individuals or groups ("an astronomer", "his great-uncle", "money-lenders")
- Abstract concepts ("Roman power", "political authority", "civil war" as a concept)
- Generic descriptions ("sea-going vessels", "land and water routes", "frontier provinces")
- Infrastructure or objects ("roads", "aqueducts", "temples")
- Cultural traditions or practices ("European tradition", "Greek mythology")
- Minor geographic features unless historically pivotal
- Entities mentioned only in passing or as comparisons

Extract entities and relationships from this paragraph:

{paragraph_text}
"""


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------


def create_extraction_chain(config: dict):
    """Create an extraction chain: paragraph_text -> ExtractionResult.

    Args:
        config: Pipeline config with keys extraction_model, extraction_temperature.

    Returns:
        Runnable that takes {"paragraph_text": str} and returns ExtractionResult.
    """
    llm = ChatOpenAI(
        model=config["extraction_model"],
        temperature=config["extraction_temperature"],
        request_timeout=60,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting structured historical entities and relationships from text.",
            ),
            ("human", EXTRACTION_PROMPT),
        ]
    )
    return prompt | llm.with_structured_output(ExtractionResult)
