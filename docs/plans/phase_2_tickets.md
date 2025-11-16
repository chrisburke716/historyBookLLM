# Phase 2 Implementation Tickets: Book Search Tool

**Created:** November 15, 2025
**Status:** 📋 Ready for Implementation
**Depends On:** Phase 1 (Tool Infrastructure) - ✅ Complete

---

## Overview

Phase 2 replaces the stub `book_search` tool with a complete implementation that performs retrieval + generation in a single tool call. The tool will retrieve relevant paragraphs from the vector database and generate a formatted answer with inline citations, all using a book-specific prompt.

**Total Tickets:** 11 (8 core implementation + 3 testing/evaluation)

---

## Core Implementation Tickets (8)

### Ticket 1: Book-Specific LLM Prompt Creation

**File:** `src/history_book/services/agents/tools/prompts.py` (new) or inline in `book_search.py`

**Description:**
Create a specialized prompt template that enforces book-only responses with proper citations.

**Requirements:**
- Base all responses ONLY on retrieved text (no memorized/outside information)
- Include context: "The Penguin History of the World"
- Instruct to use citation format: `[Ch X, p. Y]`
- Provide historical context and explanation
- Encourage complete answers (no artificial length limits)
- Instruct LLM to provide a definitive answer when sufficient information is retrieved (prevents repeated tool calls)
- Handle cases where retrieved content is insufficient

**Prompt Structure:**
```python
BOOK_SEARCH_PROMPT = """You are a history expert assistant with access to "The Penguin History of the World" by J.M. Roberts and Odd Arne Westad.

You have been provided with relevant excerpts from the book below. Your task is to answer the user's question based ONLY on these excerpts.

IMPORTANT INSTRUCTIONS:
- Base your answer entirely on the provided text
- Do NOT use any information from your training or other sources
- Include inline citations in the format [Ch X, p. Y] for every claim
- Provide historical context and explanation where appropriate
- If the excerpts provide sufficient information, give a complete answer
- If the excerpts are insufficient, clearly state what information is missing

Retrieved Excerpts:
{context}

User Question: {query}

Your Answer (with citations):"""
```

**Acceptance Criteria:**
- [ ] Prompt template created and documented
- [ ] Template enforces book-only responses
- [ ] Citation format is clearly specified
- [ ] Handles insufficient information gracefully
- [ ] Tested with sample context and queries

**Notes:**
- Consider using LangChain's `PromptTemplate` for consistency
- May need iteration based on testing results
- Balance between strict adherence and natural language flow

---

### Ticket 2: Tool Output Schema Definition

**File:** `src/history_book/services/agents/tools/book_search.py`

**Description:**
Define the structured output format that the tool returns, including both the formatted answer and source metadata.

**Requirements:**
- Tool must return a string (LangChain tool requirement)
- String should contain formatted answer with citations
- Source metadata must be captured separately for evaluation/debugging
- Consider using JSON string for structured output if needed

**Output Schema Options:**

**Option A: String with embedded metadata**
```python
# Tool returns a string directly
"Julius Caesar (100-44 BCE) was a Roman general... [Ch 8, p. 156]"

# Metadata stored in state separately via tool_results
```

**Option B: JSON string with structure**
```python
# Tool returns JSON string
{
    "answer": "Julius Caesar (100-44 BCE) was... [Ch 8, p. 156]",
    "sources": [
        {"chapter": 8, "page": 156, "content": "..."},
        {"chapter": 8, "page": 158, "content": "..."}
    ],
    "retrieval_count": 3,
    "confidence": "high"
}
```

**Recommendation:** Start with Option A (simple string) since LLM can parse the answer directly. Metadata is captured via ToolMessage attributes or state updates.

**Acceptance Criteria:**
- [ ] Output schema defined and documented
- [ ] Schema supports both formatted answer and metadata
- [ ] Compatible with LangChain `@tool` decorator requirements
- [ ] Easy for LLM to parse and use in synthesis
- [ ] Supports evaluation needs

---

### Ticket 3: Book Search Tool Core Implementation

**File:** `src/history_book/services/agents/tools/book_search.py`

**Description:**
Implement the main tool logic that performs retrieval + generation in a single call.

**Current State (Stub):**
```python
@tool
def search_book(query: str) -> str:
    """Search The Penguin History of the World for information about a topic."""
    return "Mock result: Julius Caesar was a Roman general. [Ch 8, p. 156]. Note: This is mock data for testing."
```

**New Implementation:**
```python
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import Annotated

from history_book.services.rag_service import RagService
from history_book.llm.chat_model_factory import ChatModelFactory
from history_book.config.graph_config import GraphConfig

# Import or define BOOK_SEARCH_PROMPT

@tool
def search_book(
    query: Annotated[str, "The question or topic to search for in the book"]
) -> str:
    """
    Search 'The Penguin History of the World' for information about a historical topic.

    This tool retrieves relevant passages from the book and generates a comprehensive
    answer with inline citations in the format [Ch X, p. Y].

    Use this tool when the user asks about historical events, figures, periods, or
    concepts that would be covered in a world history book.

    Args:
        query: The question or topic to search for

    Returns:
        A formatted answer with inline citations, based solely on book content
    """
    # 1. Initialize services
    config = GraphConfig()  # Or inject via dependency
    rag_service = RagService()
    llm = ChatModelFactory.create_chat_model()

    # 2. Retrieve relevant paragraphs
    paragraphs = rag_service.retrieve_context(
        query=query,
        max_results=config.tool_max_results,  # New config field
        min_similarity=config.tool_min_similarity,  # New config field
    )

    # 3. Check if sufficient context retrieved
    if not paragraphs:
        return "I could not find relevant information about this topic in 'The Penguin History of the World'."

    # 4. Format context for prompt
    context_str = "\n\n".join([
        f"[Chapter {p.chapter_number}, Page {p.page_number}]\n{p.content}"
        for p in paragraphs
    ])

    # 5. Generate answer with citations
    prompt = PromptTemplate.from_template(BOOK_SEARCH_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "context": context_str,
        "query": query
    })

    # 6. Extract text from response
    answer = response.content if hasattr(response, 'content') else str(response)

    # 7. Return formatted answer
    return answer
```

**Acceptance Criteria:**
- [ ] Tool replaces stub implementation
- [ ] Successfully retrieves paragraphs via RagService
- [ ] Generates formatted answer with LLM
- [ ] Returns proper string output
- [ ] Handles empty retrieval results
- [ ] Works with LangGraph ToolNode integration
- [ ] Properly decorated with `@tool`

**Dependencies:**
- Ticket 1 (prompt template)
- Ticket 4 (RagService integration)
- Ticket 7 (config parameters)

---

### Ticket 4: RagService Integration

**File:** `src/history_book/services/agents/tools/book_search.py`

**Description:**
Integrate the existing `RagService.retrieve_context()` method into the book search tool.

**Current RagService Interface:**
```python
# From src/history_book/services/rag_service.py
def retrieve_context(
    self,
    query: str,
    max_results: int = 5,
    min_similarity: float = 0.7
) -> list[Paragraph]:
    """
    Retrieve relevant paragraphs from the vector database.

    Args:
        query: The search query
        max_results: Maximum number of paragraphs to retrieve
        min_similarity: Minimum similarity score threshold

    Returns:
        List of Paragraph entities with content, chapter, page, etc.
    """
```

**Integration Requirements:**
- Import `RagService` properly in tool module
- Handle service initialization (singleton or per-call)
- Pass configurable `max_results` and `min_similarity`
- Use returned `Paragraph` entities to build context
- Extract necessary fields: `content`, `chapter_number`, `page_number`

**Service Initialization Options:**

**Option A: Initialize per call**
```python
def search_book(query: str) -> str:
    rag_service = RagService()  # New instance each call
    paragraphs = rag_service.retrieve_context(query)
```

**Option B: Use dependency injection**
```python
# Tool receives service as parameter (requires ToolNode modification)
def search_book(query: str, rag_service: RagService = None) -> str:
    if rag_service is None:
        rag_service = RagService()
```

**Option C: Module-level singleton**
```python
# At module level
_rag_service = RagService()

def search_book(query: str) -> str:
    paragraphs = _rag_service.retrieve_context(query)
```

**Recommendation:** Start with Option A (simplest). Optimize later if performance issues arise.

**Acceptance Criteria:**
- [ ] RagService properly imported and initialized
- [ ] `retrieve_context()` called with correct parameters
- [ ] Retrieved paragraphs are properly typed (list[Paragraph])
- [ ] Paragraph fields are accessible (content, chapter_number, page_number)
- [ ] Handles empty results gracefully
- [ ] No circular import issues

**Testing:**
- Verify retrieval returns expected paragraph count
- Check paragraph content is non-empty
- Validate chapter and page numbers are present

---

### Ticket 5: Citation Formatting Logic

**File:** `src/history_book/services/agents/tools/book_search.py`

**Description:**
Implement logic to ensure citations are consistently formatted as `[Ch X, p. Y]` in generated answers.

**Approach Options:**

**Option A: LLM-Only (Rely on Prompt)**
- Prompt instructs LLM to use citation format
- No post-processing required
- Simplest implementation
- Risk: LLM may deviate from format

**Option B: Hybrid (LLM + Post-Processing)**
- LLM generates citations in any format
- Post-processing normalizes to `[Ch X, p. Y]`
- More robust but complex
- Requires regex/parsing

**Option C: Structured Output (LLM with Schema)**
- Use structured output to separate claims from citations
- Reassemble with consistent formatting
- Most robust but highest complexity

**Recommendation:** Start with Option A. The prompt should be sufficient for consistent formatting. Add post-processing in Ticket 8 if needed.

**Prompt Instructions (from Ticket 1):**
```
- Include inline citations in the format [Ch X, p. Y] for every claim
- Example: "Julius Caesar crossed the Rubicon in 49 BCE [Ch 8, p. 156]."
```

**Context Formatting (in Tool):**
```python
# Provide citations in the context itself to guide LLM
context_str = "\n\n".join([
    f"[Chapter {p.chapter_number}, Page {p.page_number}]\n{p.content}"
    for p in paragraphs
])
```

**Validation Logic:**
```python
# Optional: Check if response contains citations
import re

def validate_citations(answer: str) -> bool:
    """Check if answer contains citations in expected format."""
    pattern = r'\[Ch \d+, p\. \d+\]'
    citations = re.findall(pattern, answer)
    return len(citations) > 0
```

**Acceptance Criteria:**
- [ ] Citations appear in consistent format: `[Ch X, p. Y]`
- [ ] Citations are inline with claims (not just at end)
- [ ] Multiple citations can appear in one answer
- [ ] Format matches across different queries
- [ ] LLM reliably follows citation instructions

**Testing:**
- Generate answers for 5-10 test queries
- Manually verify citation format consistency
- Check citation accuracy (chapter/page match context)

---

### Ticket 6: Source Metadata Extraction

**File:** `src/history_book/services/agents/tools/book_search.py`

**Description:**
Capture and structure metadata about retrieved paragraphs for debugging and evaluation purposes.

**Metadata to Capture:**
- Number of paragraphs retrieved
- Similarity scores (if available from RagService)
- Chapter and page numbers of sources
- Retrieved content (for evaluation comparison)
- Retrieval timestamp (optional)

**Implementation Approach:**

Since tools return strings to the LLM, metadata cannot be in the tool's return value. Instead, we have options:

**Option A: Store in AgentState**
```python
# Tool updates state directly (requires state parameter)
# NOT POSSIBLE - tools can't access state in LangGraph ToolNode
```

**Option B: Store in ToolMessage Attributes**
```python
# ToolNode automatically creates ToolMessage with tool output
# LangGraph 0.2+ supports artifact parameter in tools
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def search_book(query: str) -> tuple[str, dict]:
    """Search the book..."""
    # ... retrieval logic ...

    answer = "Julius Caesar was..."
    metadata = {
        "retrieval_count": len(paragraphs),
        "sources": [
            {"chapter": p.chapter_number, "page": p.page_number, "content": p.content[:200]}
            for p in paragraphs
        ]
    }

    return answer, metadata
```

**Option C: Log to LangSmith**
```python
# Use LangSmith run metadata
from langsmith import trace

@tool
def search_book(query: str) -> str:
    with trace(metadata={
        "retrieval_count": len(paragraphs),
        "chapters": [p.chapter_number for p in paragraphs]
    }):
        # ... tool logic ...
```

**Option D: Return JSON String**
```python
# Tool returns JSON that LLM can parse
import json

@tool
def search_book(query: str) -> str:
    # ... retrieval logic ...

    result = {
        "answer": "Julius Caesar was...",
        "metadata": {
            "sources": [{"chapter": 8, "page": 156}],
            "count": 3
        }
    }

    return json.dumps(result)
```

**Recommendation:** Use Option B (`response_format="content_and_artifact"`) if available in current LangGraph version. Otherwise use Option D (JSON string) for Phase 2, then upgrade to artifacts later.

**Acceptance Criteria:**
- [ ] Metadata includes retrieval count
- [ ] Metadata includes source chapter/page numbers
- [ ] Metadata includes snippet of retrieved content
- [ ] Metadata is accessible for evaluation scripts
- [ ] Metadata doesn't interfere with LLM's use of answer

**Testing:**
- Verify metadata is captured in LangSmith traces
- Verify metadata can be extracted programmatically
- Verify metadata includes all required fields

---

### Ticket 7: Tool Configuration Parameters

**Files:**
- `src/history_book/config/graph_config.py`
- `src/history_book/services/agents/tools/book_search.py`

**Description:**
Add configurable parameters for the book search tool to GraphConfig, including max_results and similarity_threshold.

**Current GraphConfig (Relevant Section):**
```python
# src/history_book/config/graph_config.py
class GraphConfig:
    enable_tools: bool = True
    enabled_tools: list[str] = ["book_search"]
    max_tool_iterations: int = 3
```

**New Configuration Fields:**
```python
class GraphConfig:
    # Existing fields
    enable_tools: bool = True
    enabled_tools: list[str] = ["book_search"]
    max_tool_iterations: int = 3

    # NEW: Tool-specific parameters
    tool_max_results: int = 5  # Max paragraphs to retrieve per tool call
    tool_min_similarity: float = 0.7  # Min similarity score for retrieval

    # Optional: Book-specific overrides
    book_tool_max_results: int | None = None  # If set, overrides tool_max_results
    book_tool_min_similarity: float | None = None  # If set, overrides tool_min_similarity
```

**Tool Usage:**
```python
# In book_search.py
def search_book(query: str) -> str:
    config = GraphConfig()

    max_results = config.book_tool_max_results or config.tool_max_results
    min_similarity = config.book_tool_min_similarity or config.tool_min_similarity

    paragraphs = rag_service.retrieve_context(
        query=query,
        max_results=max_results,
        min_similarity=min_similarity
    )
```

**Alternative: Dependency Injection**
```python
# More testable but requires ToolNode customization
def create_book_search_tool(config: GraphConfig):
    @tool
    def search_book(query: str) -> str:
        # Use config from closure
        paragraphs = rag_service.retrieve_context(
            query=query,
            max_results=config.tool_max_results,
            min_similarity=config.tool_min_similarity
        )
    return search_book
```

**Acceptance Criteria:**
- [ ] `tool_max_results` added to GraphConfig
- [ ] `tool_min_similarity` added to GraphConfig
- [ ] Default values are reasonable (5 and 0.7)
- [ ] Tool reads configuration values
- [ ] Configuration is documented with comments
- [ ] Configuration can be overridden for testing

**Testing:**
- Verify default values work as expected
- Test with different max_results (1, 3, 5, 10)
- Test with different min_similarity (0.5, 0.7, 0.9)
- Verify configuration changes affect retrieval

---

### Ticket 8: Error Handling & Edge Cases

**File:** `src/history_book/services/agents/tools/book_search.py`

**Description:**
Add robust error handling for all failure modes and edge cases in the book search tool.

**Edge Cases to Handle:**

1. **No Results Retrieved**
   ```python
   if not paragraphs:
       return "I could not find relevant information about this topic in 'The Penguin History of the World'."
   ```

2. **Low Quality Results (Below Threshold)**
   ```python
   if all(p.similarity_score < 0.6 for p in paragraphs):
       return "I found some related content, but it may not directly answer your question. Please try rephrasing."
   ```

3. **LLM Generation Failure**
   ```python
   try:
       response = chain.invoke({"context": context_str, "query": query})
   except Exception as e:
       logger.error(f"LLM generation failed: {e}")
       return "I encountered an error generating an answer. Please try again."
   ```

4. **Empty LLM Response**
   ```python
   answer = response.content if hasattr(response, 'content') else str(response)
   if not answer or answer.strip() == "":
       return "I was unable to formulate an answer based on the retrieved content."
   ```

5. **Service Initialization Failure**
   ```python
   try:
       rag_service = RagService()
   except Exception as e:
       logger.error(f"RagService initialization failed: {e}")
       return "The book search service is currently unavailable."
   ```

6. **Missing Required Fields**
   ```python
   # Validate Paragraph objects
   for p in paragraphs:
       if not hasattr(p, 'chapter_number') or not hasattr(p, 'page_number'):
           logger.warning(f"Paragraph missing required fields: {p}")
   ```

7. **Query Too Short/Empty**
   ```python
   if not query or len(query.strip()) < 3:
       return "Please provide a more specific question."
   ```

**Error Response Guidelines:**
- Be informative but not technical (user-facing)
- Suggest actions when appropriate (rephrase, try again)
- Log technical details for debugging
- Never expose internal errors to LLM/user

**Logging Strategy:**
```python
import logging

logger = logging.getLogger(__name__)

@tool
def search_book(query: str) -> str:
    logger.info(f"Book search tool called with query: {query[:100]}")

    try:
        # ... tool logic ...
        logger.info(f"Retrieved {len(paragraphs)} paragraphs")
        logger.debug(f"Paragraph sources: {[p.chapter_number for p in paragraphs]}")

        return answer

    except Exception as e:
        logger.error(f"Book search failed: {e}", exc_info=True)
        return "I encountered an error searching the book. Please try again."
```

**Acceptance Criteria:**
- [ ] All edge cases have explicit handling
- [ ] Error messages are user-friendly
- [ ] Technical errors are logged appropriately
- [ ] Tool never raises unhandled exceptions
- [ ] Empty/invalid queries are handled gracefully
- [ ] Service failures don't crash the graph
- [ ] All error paths are tested

**Testing:**
- Test with empty query
- Test with very short query (1-2 chars)
- Test with query that returns no results
- Test with service unavailable (mock)
- Test with LLM failure (mock)
- Verify logging output

---

## Testing & Evaluation Tickets (3)

### Ticket 9: Functional Testing

**Files:**
- `scripts/verify/verify_book_search_tool.py` (new)
- `tests/test_book_search_tool.py` (new, optional)

**Description:**
Create comprehensive tests to verify the book search tool works correctly in isolation and integrated with the graph.

**Test Categories:**

**1. Unit Tests (Tool in Isolation)**
```python
# tests/test_book_search_tool.py
import pytest
from history_book.services.agents.tools.book_search import search_book

def test_search_book_returns_string():
    result = search_book.invoke({"query": "Julius Caesar"})
    assert isinstance(result, str)
    assert len(result) > 0

def test_search_book_includes_citations():
    result = search_book.invoke({"query": "Roman Empire"})
    assert "[Ch" in result and "p." in result

def test_search_book_handles_empty_query():
    result = search_book.invoke({"query": ""})
    assert "specific question" in result.lower()

def test_search_book_handles_no_results():
    # Mock RagService to return empty list
    result = search_book.invoke({"query": "nonexistent topic xyz123"})
    assert "could not find" in result.lower()
```

**2. Integration Tests (Tool with Graph)**
```python
# scripts/verify/verify_book_search_tool.py
from history_book.services.graph_rag_service import GraphRagService
from history_book.services.graph_chat_service import GraphChatService

def test_tool_in_graph():
    """Test that the tool is called correctly by the graph."""
    chat_service = GraphChatService()
    session_id = chat_service.create_session(title="Test Session")

    result = chat_service.send_message(
        session_id=session_id,
        user_message="Who was Julius Caesar?"
    )

    # Verify tool was called
    assert result.metadata["tool_iterations"] > 0

    # Verify answer includes citations
    assert "[Ch" in result.content
    assert "p." in result.content

    # Verify LangSmith trace shows tool execution
    # (Manual verification via LangSmith UI)
    print(f"Trace ID: {result.trace_id}")

def test_multiple_tool_iterations():
    """Test that the graph handles multiple tool iterations correctly."""
    # Create scenario where LLM might call tool multiple times
    # Verify max iterations is respected
    pass

def test_tool_synthesis():
    """Test that LLM synthesizes tool results correctly."""
    # Verify final answer incorporates tool output
    pass
```

**3. Manual Test Cases**
```bash
# scripts/verify/verify_book_search_tool.py (CLI script)

# Test Case 1: Simple factual query
Query: "Who was Julius Caesar?"
Expected: Answer with 2-3 citations, covers key facts

# Test Case 2: Complex query requiring multiple sources
Query: "What led to the fall of the Roman Empire?"
Expected: Multi-paragraph answer with multiple citations

# Test Case 3: Specific event
Query: "What happened at the Battle of Actium?"
Expected: Focused answer with citations

# Test Case 4: Broad topic
Query: "Tell me about Ancient Egypt"
Expected: Comprehensive answer or appropriate scoping

# Test Case 5: Edge case - not in book
Query: "Who is Elon Musk?"
Expected: "Could not find information" message
```

**Verification Script:**
```python
#!/usr/bin/env python3
"""Verify book search tool functionality."""

import sys
from history_book.services.graph_chat_service import GraphChatService

TEST_QUERIES = [
    "Who was Julius Caesar?",
    "What led to the fall of the Roman Empire?",
    "What happened at the Battle of Actium?",
    "Tell me about Ancient Egypt",
    "Who is Elon Musk?",  # Should say "not found"
]

def main():
    chat_service = GraphChatService()
    session_id = chat_service.create_session(title="Tool Verification")

    print("=" * 80)
    print("BOOK SEARCH TOOL VERIFICATION")
    print("=" * 80)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 80)

        result = chat_service.send_message(session_id, query)

        print(f"Answer: {result.content[:200]}...")
        print(f"Tool Iterations: {result.metadata.get('tool_iterations', 0)}")
        print(f"Citations Found: {'[Ch' in result.content and 'p.' in result.content}")
        print(f"Trace ID: {result.trace_id}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE - Review results above and LangSmith traces")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

**Acceptance Criteria:**
- [ ] Tool returns correct output structure (string with citations)
- [ ] Tool integrates with graph (no errors)
- [ ] Tool is called by LLM when appropriate
- [ ] Citations are present and correctly formatted
- [ ] Edge cases are handled gracefully
- [ ] All manual test cases pass
- [ ] Verification script runs successfully

**Deliverables:**
- Verification script in `scripts/verify/`
- Test results documented
- LangSmith trace IDs for key test cases
- Optional: pytest unit tests

---

### Ticket 10: LangSmith Trace Verification

**Description:**
Verify that the book search tool execution is properly traced and observable in LangSmith.

**What to Verify:**

1. **Tool Call Visibility**
   - Tool call appears in trace tree
   - Tool name is correct (`search_book`)
   - Tool inputs are captured (query)
   - Tool outputs are captured (answer)

2. **Execution Timing**
   - Retrieval step is visible (if logged)
   - Generation step is visible
   - Total tool execution time is reasonable

3. **Metadata Capture**
   - Number of retrieved paragraphs (if logged)
   - Similarity scores (if logged)
   - Source chapters/pages (if logged)

4. **Error Traces**
   - Errors are properly captured
   - Error messages are informative
   - Stack traces are available

**Verification Process:**

1. Run test query through graph
2. Get trace ID from result
3. Open trace in LangSmith UI
4. Verify all elements are present
5. Screenshot key views for documentation

**LangSmith Checklist:**
- [ ] Tool appears in trace tree under ToolNode
- [ ] Tool input (query) is visible
- [ ] Tool output (answer) is visible
- [ ] Tool execution time is tracked
- [ ] Tool run has metadata (if implemented)
- [ ] Can drill down into LLM call within tool
- [ ] Citations are visible in output
- [ ] Errors are captured (if any)

**Documentation:**
- Capture screenshot of successful tool execution trace
- Document trace structure in completion doc
- Note any LangSmith limitations encountered

**Acceptance Criteria:**
- [ ] Tool execution is fully traced
- [ ] All relevant data is captured
- [ ] Traces are useful for debugging
- [ ] Trace structure is documented

---

### Ticket 11: Evaluation Comparison

**Files:**
- `src/history_book/evals/run_comparison.py` (may need updates)
- `docs/plans/phase_2_evaluation_results.md` (new)

**Description:**
Run evaluations comparing Phase 2 tool-based approach against Phase 0 baseline RAG to measure quality impact.

**Evaluation Setup:**

1. **Baseline Configuration (Phase 0)**
   - Linear graph: `retrieve_node → generate_node`
   - No tools, automatic retrieval
   - Generic RAG prompt

2. **Phase 2 Configuration**
   - Tool-enabled graph: `generate_node → tools_node`
   - Book search tool with specialized prompt
   - LLM decides when to retrieve

**Metrics to Compare:**

**Quality Metrics (LLM-based evaluators):**
- Helpfulness
- Hallucination rate
- Factual accuracy
- Coherence
- IDK handling (when book doesn't have info)
- Relevance

**Structural Metrics (function evaluators):**
- Document count (avg paragraphs retrieved)
- Response length
- Citation count
- Tool iteration count

**Performance Metrics:**
- Latency (avg response time)
- Token usage (if tracked)
- Success rate (non-empty responses)

**Evaluation Process:**

```python
# Pseudo-code for comparison
from history_book.evals import run_experiment

# Run Phase 0 baseline
baseline_results = run_experiment(
    experiment_name="Phase 0 Baseline (Auto-Retrieve)",
    dataset="eval_dataset_100",
    graph_config={"enable_tools": False}  # Use old graph
)

# Run Phase 2 tool-based
phase2_results = run_experiment(
    experiment_name="Phase 2 Tool-Based RAG",
    dataset="eval_dataset_100",
    graph_config={"enable_tools": True}  # Use new graph
)

# Compare results
comparison = compare_experiments(baseline_results, phase2_results)
```

**Success Criteria:**

Phase 2 should meet or exceed baseline on:
- Helpfulness score (target: ≥ baseline)
- Hallucination rate (target: ≤ baseline)
- Factual accuracy (target: ≥ baseline)
- Citation presence (target: 100% where applicable)

**Analysis Questions:**
- When does LLM choose to call tool vs answer directly?
- How often does LLM need multiple tool iterations?
- Are citations more consistent in Phase 2?
- Is response quality higher with specialized prompt?
- Are there any regressions?

**Acceptance Criteria:**
- [ ] Evaluation runs on same dataset as baseline
- [ ] All metrics are comparable
- [ ] Results are documented with analysis
- [ ] LangSmith experiment links are captured
- [ ] Phase 2 meets or exceeds baseline quality
- [ ] Any regressions are explained and justified
- [ ] Tool usage patterns are analyzed

**Deliverables:**
- Evaluation results document (markdown)
- LangSmith experiment URLs
- Comparison table (baseline vs Phase 2)
- Analysis of tool usage patterns
- Recommendations for Phase 3

---

## Implementation Order

**Recommended sequence:**

1. **Ticket 1** - Create prompt template (foundation for everything)
2. **Ticket 2** - Define output schema (clarifies what tool returns)
3. **Ticket 7** - Add configuration (needed by implementation)
4. **Ticket 4** - Integrate RagService (retrieval component)
5. **Ticket 3** - Core implementation (brings it all together)
6. **Ticket 5** - Citation formatting (refine output)
7. **Ticket 6** - Metadata extraction (observability)
8. **Ticket 8** - Error handling (robustness)
9. **Ticket 9** - Functional testing (verify works)
10. **Ticket 10** - LangSmith verification (observability check)
11. **Ticket 11** - Evaluation comparison (quality check)

**Alternative (Parallel Tracks):**

**Track A (Core Implementation):** 1 → 2 → 7 → 4 → 3
**Track B (Quality & Robustness):** 5 → 6 → 8
**Track C (Testing):** 9 → 10 → 11

Can work on Track A first, then B, then C. Or have parallel development.

---

## Success Criteria (Phase 2 Complete)

Phase 2 is complete when:

✅ **Implementation Complete**
- [ ] All 8 core tickets implemented
- [ ] Stub tool replaced with full implementation
- [ ] Tool produces formatted answers with citations
- [ ] Tool integrates with graph successfully

✅ **Testing Passed**
- [ ] Functional tests pass
- [ ] Manual test cases verified
- [ ] LangSmith traces show correct execution
- [ ] No regressions in server startup or basic functionality

✅ **Quality Verified**
- [ ] Citations are present and consistently formatted
- [ ] Answers are based solely on book content
- [ ] Edge cases are handled gracefully
- [ ] Error handling is robust

✅ **Evaluation Complete**
- [ ] Comparison with baseline complete
- [ ] Phase 2 meets or exceeds quality metrics
- [ ] Tool usage patterns documented
- [ ] Results published to LangSmith

✅ **Documentation Updated**
- [ ] Implementation completion doc created
- [ ] Evaluation results documented
- [ ] Code is commented and clear
- [ ] CLAUDE.md updated if needed

---

## Known Considerations

### Configuration Management
- Consider making config injectable for testability
- May need environment-specific overrides later
- Tool parameters might need tuning based on evaluation

### Prompt Engineering
- Expect iteration on prompt wording
- May need A/B testing different prompt versions
- Balance between strict adherence and natural flow

### Citation Accuracy
- LLM may hallucinate citations even with instruction
- Consider post-processing validation if needed
- Track citation accuracy in evaluations

### Metadata Handling
- LangGraph artifact support may vary by version
- May need to upgrade LangGraph for best metadata handling
- JSON string fallback is acceptable for Phase 2

### Performance
- Tool call adds latency (retrieval + generation in tool)
- Monitor total response time vs baseline
- May need caching strategy later

---

## Phase 3 Preview

After Phase 2 is complete and evaluated, Phase 3 will add:

1. **Web Search Tool** - Supplement book with external sources
2. **Multi-Tool Synthesis** - Combine book + web results
3. **Advanced Prompt Engineering** - Guide when to use each tool
4. **Streaming Support** - Stream responses with tools

---

## References

**Phase 1 Documentation:**
- `/docs/plans/phase_1_implementation_complete.md` - Tool infrastructure

**Planning Documents:**
- `/docs/plans/short_term_agent_tools_plan.md` - Original Phase 2 plan
- `/docs/plans/long_term_feature_roadmap.md` - Future vision

**Key Code:**
- Stub tool: `/src/history_book/services/agents/tools/book_search.py`
- Graph service: `/src/history_book/services/graph_rag_service.py`
- RAG service: `/src/history_book/services/rag_service.py`
- Config: `/src/history_book/config/graph_config.py`

**Evaluation Framework:**
- Evals directory: `/src/history_book/evals/`
- CLAUDE.md: `/src/history_book/evals/CLAUDE.md`

---

**Status:** 📋 Ready for Implementation
**Total Tickets:** 11 (8 core + 3 testing/eval)
**Estimated Effort:** 1-2 weeks
**Dependencies:** Phase 1 complete ✅
