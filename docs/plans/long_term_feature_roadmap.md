# Long-Term Feature Roadmap for History Book Chat

## Context
This document captures high-level ideas and future enhancements for the History Book RAG chat application. These are not immediate priorities but serve as a reference for future development.

**User Goal:** Interact with and understand "The Penguin History of the World" - not just answer history questions, but help track themes, see big-picture trends, and navigate a long, detailed book with breaks.

---

## Future Tools & Capabilities

### Smart Routing & Planning (HIGH PRIORITY)
**What:** Multi-step query decomposition and sub-query synthesis
- Break complex queries into focused sub-queries
- Execute sub-queries separately (in parallel or sequence)
- Synthesize results into coherent response
- Related to compare/contrast functionality

**Use Cases:**
- "Compare Rome and China, then explain how they influenced each other"
- "Give me a timeline of Mediterranean trade, highlighting conflicts"

**Implementation Notes:**
- Add planning node to LangGraph agent
- Query complexity analysis
- Sub-query generation and routing
- Result aggregation/synthesis step

---

### Source Attribution & Citations (HIGH PRIORITY)
**What:** Explicit tracking and display of information sources

**Preferred Format:** Footnote style with structured backend
- Inline footnote markers in text: `[1]`, `[2]`
- Citation list at end of response
- Structured response format for frontend formatting

**Source Types:**
- Book: Chapter, paragraph, page number
- Web: URL, access date
- Tool: Which tool provided the information

**Implementation Notes:**
- Track sources at state level or in tool outputs
- Frontend citation/footnote formatter
- Consider expandable citations or sidebar display

---

### Generate Summaries
**What:** Pre-generated summaries available for reading or as retrievable context

**Approach:**
- One-time process (not on-demand tool)
- Summaries stored for later retrieval
- Can be queried by the agent when needed

**Future Enhancement:** Integrate with topic breaks detection (v2)
- Summaries organized by detected topics within chapters
- Hierarchical structure: chapter → topics → summary

**Implementation Notes:**
- Part of ingestion pipeline or separate batch process
- Store in Weaviate for retrieval
- Consider different granularities: chapter-level, topic-level

---

### Detect Topic Breaks
**What:** Identify topic transitions within chapters

**Approach:**
- Semantic clustering/grouping of paragraphs
- Iterative process: identify clusters → summarize topic → re-cluster with summary vectors
- Second pass with LLM for refinement

**Architecture:**
- Not a callable agent tool
- Functionality in Summary/Text Analysis Service
- Pre-compute during ingestion or batch analysis

**Use Cases:**
- Understanding chapter structure
- Creating more granular summaries
- Better context retrieval (topic-level vs paragraph-level)

**Implementation Notes:**
- Experiment by hand first to find best approach
- Embedding-based clustering (cosine similarity thresholds)
- Balance between automation and quality

---

### Compare & Contrast
**What:** Structured comparison of entities, regions, or concepts

**Approach:**
- Related to smart routing/planning
- Break comparison into sub-queries:
  1. Retrieve context for entity 1
  2. Retrieve context for entity 2
  3. Synthesize comparison
- Execute sub-queries separately, feed results back for synthesis

**Use Cases:**
- "Compare the Roman and Han Chinese empires"
- "How did democracy develop differently in Athens vs Rome?"
- "Contrast the book's view with modern scholarship"

**Implementation Notes:**
- Depends on planning/routing infrastructure
- Consider pre-defined dimensions (political, economic, cultural) vs free-form
- Output formats: narrative, table, bullet points

---

### Search Conversation History
**What:** RAG over previous conversation messages (LOWER PRIORITY)

**Current Status:** Haven't fully thought through yet

**Related Enhancement:** Thread titles based on thread summaries
- Auto-generate descriptive titles from conversation content
- Help organize and find past conversations
- Could be displayed in session list

**Implementation Notes:**
- Index ChatMessage entities with embeddings
- Search within session or across all sessions
- Privacy considerations

---

### Chapter-Aware Retrieval
**What:** Optional filtering to avoid spoilers or focus on specific sections

**Approach:**
- Simple UI checkbox: "Only query chapters before: ___"
- User enters chapter number
- Filter applied to book retrieval tool

**Priority:** Low but easy to implement

**Implementation Notes:**
- Chapter metadata must be indexed in Weaviate
- Session-level setting stored in ChatSession.metadata
- Clear indication when results are filtered

---

## Standalone Features (Future Integrations)

### Knowledge Graph
**Purpose:**
- Visualization of entity relationships
- Enhance retrieved context for agent (graph-based retrieval)

**Initial Implementation:**
- Standalone tool with own page in app
- NER (Named Entity Recognition) to extract entities
- Relationship extraction from text

**Future Integration:**
- Agent can query knowledge graph as a tool
- "Show me how X relates to Y"
- Graph-based retrieval augments vector search

---

### Timeline Builder
**Purpose:**
- Track rise and fall of empires, dynasties, civilizations
- See parallel timelines (what happened in different regions simultaneously)
- Visualize temporal relationships

**Initial Implementation:**
- Standalone tool with own page in app
- Manual or semi-automated timeline creation
- Extract temporal information from text

**Future Integration:**
- Agent can call timeline builder as a tool
- "Generate a timeline of Mediterranean empires 500 BCE - 500 CE"
- Results can be saved and referenced in conversations

**Extensibility:**
- Compare timelines across regions
- Filter by entity type (political, cultural, technological events)
- Zoom levels (millennia → centuries → decades)

---

## Implementation Philosophy

**Incremental Development:**
- Build one feature at a time
- Run evaluations after each change
- Iterate based on results

**Separation of Concerns:**
- Analysis/processing services (topic detection, summarization) separate from agent tools
- Agent tools are interfaces to underlying capabilities
- Standalone features can later become agent-callable tools

**User Control:**
- Features should enhance, not replace, reading experience
- Provide options and controls (chapter filtering, citation styles)
- Transparent about sources and limitations

---

## Dependencies & Prerequisites

**For Smart Routing/Planning:**
- Requires tool infrastructure (Phase 1)
- Complex query detection logic
- Sub-query generation capabilities

**For Source Attribution:**
- Structured response format from tools
- Frontend rendering for citations
- Tool outputs must include source metadata

**For Knowledge Graph:**
- NER pipeline (spaCy, Flair, or LLM-based)
- Graph database or structure (Neo4j, or in-memory graph)
- Relationship extraction logic

**For Timeline Builder:**
- Temporal information extraction
- Date normalization (various formats)
- Timeline visualization library (frontend)

---

This is a living document. Ideas will evolve as we build and learn from using the system.
