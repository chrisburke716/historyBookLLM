# Short-Term Plan: Tool-Enabled LangGraph Agent

## Overview
Transform the current 2-node linear agent (retrieve → generate) into a tool-enabled agent that can call specialized tools and synthesize results.

**Timeline:** 3-4 weeks
**Goal:** Enable agent to use book search and web search tools with proper orchestration and refinement capabilities.

---

## Architecture Decision: Tool Design Pattern

### Chosen Approach: Retrieval + Generation in Tool

**Pattern:** Tools produce complete, specialized outputs. Agent orchestrates and synthesizes.

**Why:**
- Each tool has domain-specific prompting (book-specific vs web-specific)
- Tools are self-contained and testable
- Agent focuses on: which tools to call, in what order, how to combine results
- Better token efficiency (condensed outputs vs raw paragraphs)
- Cleaner citation handling per domain

**Graph Structure:**
```
START → generate_node → [conditional routing] → tools_node → generate_node → END
                         ↓
                        END (if no tools needed)
```

**Key Change:** Remove automatic retrieve_node. All book queries are explicit tool calls.

---

## Phase 1: Tool Infrastructure

### Goal
Enable LangGraph agent to call tools with conditional routing and iteration limits.

### Core Components

1. **Update AgentState Schema**
   - Uncomment `tool_calls` and `tool_results` fields
   - Add `tool_iterations` counter to prevent infinite loops
   - Track tool metadata for observability

2. **Create Tools Module Structure**
   - New directory: `src/history_book/services/agents/tools/`
   - Tools: `book_search.py`, `web_search.py`
   - Base exports in `__init__.py`

3. **Implement Conditional Routing**
   - `should_continue(state)` function
   - Returns "tools" if LLM made tool calls, "end" otherwise
   - Enforce max_iterations limit (start with 3)

4. **Add Tools Node**
   - Use LangGraph's `ToolNode` (handles execution automatically)
   - Receives tool_calls from LLM, executes, returns ToolMessage results

5. **Update Graph Construction**
   - Add conditional edge from generate_node
   - Add loop from tools_node back to generate_node
   - Remove automatic retrieve_node

6. **Bind Tools to LLM**
   - Use `llm.bind_tools([...])` in generate_node
   - LLM can now decide to call tools or answer directly

7. **Configuration Updates**
   - Add tool-related settings to GraphConfig
   - `max_tool_iterations`, `enable_tools`, `tools_enabled` list
   - Toggle tools per environment

### Testing Focus
- Routing logic with various states
- Tool iteration counting and limits
- Graph visualization shows conditional routing
- LangSmith traces show tool calls

---

## Phase 2: Tool #1 - Book Info Retrieval

### Goal
Create a tool that retrieves from the book AND generates a formatted answer with citations.

### Tool Behavior

**What it does:**
1. Takes query as input
2. Retrieves relevant paragraphs from Weaviate (reuses RagService)
3. Generates answer using book-specific prompt
4. Returns formatted answer with citations + source metadata

**Tool Output Structure:**
- **Formatted answer:** Complete response with inline citations `[Ch 12, p. 234]`
- **Source metadata:** Structured data about retrieved paragraphs (for evals/debugging)
- Both components returned together

### Book-Specific Prompt Requirements

**Key principles:**
- Base all responses on retrieved text only (no memorized/outside info)
- Include context about project: "The Penguin History of the World"
- Proper citation format with chapter and page numbers
- Historical context and explanation

**Details to be refined during implementation**

### Integration Points

- Reuse existing `RagService.retrieve_context()` for retrieval
- Add new generation step with book-specific prompt
- Tool registered in GraphRagService tools list
- Configurable parameters: max_results, similarity threshold

### Testing Focus
- Tool produces complete answers with citations
- Source metadata is accurate and complete
- Evals comparing old auto-retrieval vs new tool approach
- Citation format is consistent

---

## Phase 3: Tool #2 - Web Search

### Goal
Enable agent to search external sources to supplement or contrast with book content.

### Tool Behavior

**Use cases:**
- User explicitly requests external sources
- Agent detects book answer would benefit from additional context
- Comparing book's perspective with modern scholarship
- Finding information beyond book's scope

**Implementation:**
- Use OpenAI's `web_search_preview` capability via LangChain
- Tool generates formatted answer from web results
- Returns answer with source URLs and access dates

### Agent Decision Logic

**When to use web search:**
- Explicit user request for external sources
- Modern perspectives or recent events
- Cross-referencing controversial claims

**Strategy:**
- Rely on system prompt guidance + tool descriptions
- Agent learns appropriate usage through prompting
- Iterate based on observed behavior

**System prompt should emphasize:**
- Prefer book content when available
- Use web search as supplement, not replacement
- Be explicit about which sources are being used

### Testing Focus
- Agent correctly chooses when to use web vs book tool
- Web results properly cited with URLs
- Multi-tool synthesis (book + web) works smoothly

---

## Phase 4: Agent Orchestration & Refinement

### Goal
Agent effectively combines tool results and refines outputs as needed.

### Agent Responsibilities

**Decision Making:**
- Which tools to call for a given query
- Whether to call multiple tools
- When tool output is sufficient vs needs refinement

**Refinement Scenarios:**
1. **Multi-tool synthesis:** Combining book + web search results
2. **Follow-up questions:** Building on previous tool results in context
3. **Rephrasing for clarity:** Adjusting tool output for better user experience

**Simple Scenario (initial implementation):**
- With only book tool available initially, agent should refine when:
  - Follow-up questions need context integration
  - User asks for rephrasing or clarification

### Agent Prompt Design

**Key elements:**
- Clear guidance on tool usage
- Instructions for when to refine vs return tool output directly
- Synthesis patterns for multi-tool results

**To be refined during implementation**

---

## Key Requirements Summary

### Tool #1: Book Search
- ✓ Retrieval + generation in one tool
- ✓ Book-specific prompt (base on retrieved text, no outside info)
- ✓ Returns: formatted answer with citations + source metadata
- ✓ Output length: as long as needed
- ✓ Citation format: enough info to link to source text

### Tool #2: Web Search
- ✓ Use OpenAI web_search_preview
- ✓ Triggered when appropriate (explicit request or detected need)
- ✓ Returns: answer with URLs and dates
- ✓ Clear source attribution

### Agent Behavior
- ✓ Orchestrates tool calls
- ✓ Synthesizes multi-tool results
- ✓ Refines when needed: multi-tool, follow-ups, rephrasing
- ✓ Returns tool output directly when sufficient

### Architecture
- ✓ Max iterations: 3 (configurable)
- ✓ Tools have full state access
- ✓ No automatic retrieve_node (all queries are explicit tool calls)
- ✓ Conditional routing after generate_node
- ✓ Tool loop back to generate_node

---

## Configuration Decisions

### Max Tool Iterations
- Start with 3
- Configurable via GraphConfig
- Prevents infinite loops

### Tool Access
- Tools receive full state (not just current message)
- Enables context-aware tool behavior

### Auto-Retrieval
- Remove automatic retrieve_node
- All book queries via explicit search_book tool calls
- Cleaner, more traceable

---

## Implementation Sequence

### Week 1: Infrastructure
- Update state schema and graph structure
- Implement routing and tool execution
- Bind tools to LLM
- Configuration updates
- Basic testing

### Week 2: Book Search Tool
- Implement tool with retrieval + generation
- Define book-specific prompt
- Integrate with GraphRagService
- End-to-end testing
- Run evals for baseline

### Week 3: Web Search Tool
- Implement OpenAI web_search_preview integration
- Add system prompt guidance
- Test tool selection behavior
- Multi-tool integration testing

### Week 4: Agent Refinement
- Optimize agent prompt for orchestration
- Test refinement scenarios
- Comprehensive evals (old vs new)
- Document behavior and usage patterns

---

## Future Enhancements (Not in Scope)

These are noted for future reference but not part of this implementation phase:

- Hybrid search for book tool
- Advanced filtering in book tool
- Additional prompt refinements based on evals
- More sophisticated tool routing logic
- Tool result caching
- Streaming improvements for tool calls

---

## Success Criteria

**Phase 1 Complete:**
- Agent can call tools and route conditionally
- Tool iterations limited correctly
- Graph visualization shows new structure

**Phase 2 Complete:**
- Book search tool produces complete answers with citations
- Source metadata captured for evals
- Tool output quality meets or exceeds current RAG

**Phase 3 Complete:**
- Web search tool integrated
- Agent uses both tools appropriately
- Multi-tool synthesis works

**Phase 4 Complete:**
- Agent refines outputs when needed
- Evals show improvement over baseline
- Tool usage patterns are sensible and traceable

---

## Open Items for Later Discussion

These will be addressed during implementation of each phase:

- Exact book-specific prompt wording
- Citation format details (inline structure)
- Source metadata schema
- Agent prompt refinement patterns
- Evaluation metrics and test cases
- Streaming behavior with tools
- Error handling strategies

---

This is a living document. Details will be refined as each phase is implemented.
