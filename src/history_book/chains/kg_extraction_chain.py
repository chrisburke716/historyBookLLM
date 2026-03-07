"""KG entity/relationship extraction chain.

Uses structured outputs to extract entities and relationships from paragraph text.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from history_book.data_models.kg_entities import EntityType

# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class Entity(BaseModel):
    name: str = Field(description="The canonical name of the entity")
    type: EntityType = Field(
        description="Entity type: person, polity, place, event, or concept"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names for this entity"
    )
    description: str | None = Field(
        default=None, description="Brief description based only on this paragraph"
    )


class Relationship(BaseModel):
    source_entity: str = Field(description="Exact name from entities list")
    relation_type: str = Field(
        description="One of: ruled, conquered, fought, allied_with, succeeded, "
        "revolted_against, influenced, part_of, founded, evolved_into, participated_in"
    )
    target_entity: str = Field(description="Exact name from entities list")
    description: str | None = Field(
        default=None,
        description="Brief description capturing key context about this relationship "
        "(1 sentence). Null if the relation_type alone is sufficient.",
    )
    temporal_context: str | None = Field(
        default=None,
        description="Date, year, year range, or century when this relationship held "
        "(e.g., '753 BC', '264-241 BC', 'sixth century BC'). "
        "Null if no specific time is stated.",
    )
    start_year: int | None = Field(default=None, exclude=True)
    end_year: int | None = Field(default=None, exclude=True)
    temporal_precision: str | None = Field(default=None, exclude=True)


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
- concept: Religions, belief systems, philosophies, ideologies (e.g., Christianity, Judaism, Stoicism, democracy)

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
5. Do NOT extract dates or time periods as entities — capture them as temporal_context on relationships. temporal_context should be a specific date, year, year range, or century (e.g., "753 BC", "264-241 BC", "sixth century BC"). Leave it null if no time is mentioned in the text for that relationship.
6. For each relationship, include a brief description capturing key context (1 sentence). Leave null if the relation_type alone is sufficient.
7. Relationships MUST reference exact entity names from your entities list
8. Only extract entities that participate in at least one relationship

**DO NOT EXTRACT**:
- Unnamed individuals or groups ("an astronomer", "his great-uncle", "money-lenders")
- Vague abstractions ("Roman power", "political authority", "civil war" as a concept)
- Generic descriptions ("sea-going vessels", "land and water routes", "frontier provinces")
- Infrastructure or objects ("roads", "aqueducts", "temples")
- Cultural traditions or practices ("European tradition", "Greek mythology") — but DO extract named religions/philosophies as concept
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
