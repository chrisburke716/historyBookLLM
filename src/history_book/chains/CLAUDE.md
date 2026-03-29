# CLAUDE.md - Chains

LangChain LCEL chain factories for LLM-powered KG operations. Each module exposes a factory function returning a `Runnable` (prompt | llm.with_structured_output).

## Files

| File | Purpose | Model key in config |
|------|---------|---------------------|
| `kg_extraction_chain.py` | Extract entities + relationships from paragraph text | `extraction_model` |
| `kg_merge_chain.py` | Decide whether two entities should merge (LLM similarity merge) | `merge_model` |
| `kg_merge_chain.py` | Filter rule-based merge candidates (name/alias match) | `rule_filter_model` |
| `kg_temporal_chain.py` | Parse free-text temporal expressions into year integers | `temporal_model` |
| `title_generation_chain.py` | Generate descriptive session titles from chat history | (hardcoded) |

## KG Extraction (`kg_extraction_chain.py`)

**Input**: `{paragraph_text, paragraph_id}`
**Output**: `ExtractionResult(entities: list[Entity], relationships: list[Relationship], paragraph_id)`

**Entity** fields: `name`, `type` (EntityType enum), `aliases`, `description`
**Relationship** fields: `source_entity`, `relation_type` (RelationType enum), `target_entity`, `description`, `temporal_context`

```python
chain = create_extraction_chain(config)
result: ExtractionResult = await chain.ainvoke({"paragraph_text": "...", "paragraph_id": "uuid"})
```

## KG Merge (`kg_merge_chain.py`)

Two chains, same 8-field input: `entity1_name/type/aliases/description` + `entity2_name/type/aliases/description`.

**`create_merge_chain`** — LLM similarity merge decisions:
- Output: `EntityMergeDecision(reasoning, confidence, should_merge, canonical_name)`
- Used in batches: similarity candidates (cosine > threshold) sent for LLM judgment

**`create_rule_filter_chain`** — Validates rule-based (name/alias) matches:
- Output: `RuleMergeDecision(should_merge, canonical_name)`
- Lighter model (`gpt-4.1-mini`); "default to merging, only reject if clearly distinct"
- Run before committing exact-name matches to catch e.g. same surname ≠ same person

```python
filter_chain = create_rule_filter_chain(config)  # rule_filter_model = "gpt-4.1-mini"
merge_chain = create_merge_chain(config)           # merge_model = "o4-mini" or similar
```

## KG Temporal (`kg_temporal_chain.py`)

Parses free-text temporal expressions extracted during entity merging.

**Input**: `{temporal_context: str}`
**Output**: `TemporalParsed(start_year: int | None, end_year: int | None, precision: str)`

- Years are integers; BC dates are negative (753 BC → `-753`)
- `precision`: `"year"` | `"decade"` | `"century"` | `"approximate"`

## Title Generation (`title_generation_chain.py`)

Used by `GraphChatService` to generate session titles from conversation history. Not part of the KG pipeline.

## Config Keys (from `KGIngestionService.DEFAULT_CONFIG`)

```python
"extraction_model": "gpt-4.1"
"merge_model": "o4-mini"
"rule_filter_model": "gpt-4.1-mini"
"rule_filter_temperature": 0.0
"merge_temperature": 1.0  # required for o-series reasoning models
```
