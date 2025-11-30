# Pairwise Evaluation Implementation Plan

**Date:** November 28, 2025
**Status:** ✅ Ready for Implementation
**Principles:** DRY, YAGNI, KISS

---

## Executive Summary

This plan implements LangSmith pairwise evaluations to compare two RAG system experiments (legacy LCEL vs agent-based). Following DRY/YAGNI/KISS principles, we'll extend `run_evals.py` with a `--pairwise` flag rather than creating a separate script, and use simple functions instead of complex base classes.

**Key Decision:** Simple function-based approach with minimal new code (~250 lines across 4 files).

---

## 1. Architecture Decision: Extend run_evals.py (Option A)

### Rationale

**Option A: Extend `run_evals.py`** ✅ CHOSEN
- **DRY**: Reuses argument parsing, LangSmith client setup, LLM initialization
- **YAGNI**: Don't create new script when existing one can be extended
- **KISS**: Single entry point for all evaluations (standard + pairwise)
- **User experience**: `poetry run python scripts/run_evals.py --pairwise --experiments exp1 exp2`

**Option B: New `run_pairwise_evals.py`** ❌ REJECTED
- Duplicates ~50% of run_evals.py code (argparse, client setup, evaluator creation)
- Two scripts to maintain vs one
- User must remember two different commands
- Violates DRY principle

### Implementation

Add to `run_evals.py`:
```python
parser.add_argument(
    "--pairwise",
    action="store_true",
    help="Run pairwise comparison between two experiments"
)
parser.add_argument(
    "--experiments",
    nargs=2,
    metavar=("EXP1", "EXP2"),
    help="Two experiment IDs or names to compare (required with --pairwise)"
)
```

---

## 2. Base Class Design: Simple Functions (No PairwiseEvaluator)

### Rationale

**Simple Functions** ✅ CHOSEN
- **YAGNI**: We only need 3-4 pairwise evaluators initially
- **KISS**: Functions are simpler than class hierarchies
- **LangSmith API**: `evaluate_comparative()` accepts plain functions
- **Precedent**: `FunctionEvaluator` already uses function-based pattern

**PairwiseEvaluator Base Class** ❌ REJECTED
- Only saves ~5 lines per evaluator
- Adds complexity to `base.py`
- Over-engineering for 3-4 evaluators
- Can always refactor later if we add 10+ pairwise evaluators (YAGNI)

### Implementation

Pairwise evaluator signature:
```python
def pairwise_helpfulness(runs: list, example) -> dict:
    """
    Compare two runs for helpfulness.

    Args:
        runs: List of 2 Run objects from different experiments
        example: The evaluation example

    Returns:
        {
            "key": "pairwise_helpfulness",
            "scores": {run_a.id: 1.0, run_b.id: 0.0},
            "comment": "Explanation of winner"
        }
    """
    run_a, run_b = runs
    # Comparison logic
    return result
```

---

## 3. Evaluator Implementation: 4 High-Value Comparisons

### Selected Evaluators

**1. Pairwise Helpfulness** (Highest Priority)
- Compares which response better answers the user's question
- Uses LLM judge with comparison prompt
- Returns 1.0 for winner, 0.0 for loser, 0.5 for tie

**2. Pairwise Hallucination** (Critical for RAG)
- Compares which response has fewer hallucinations **relative to its own retrieved context**
- Each response judged against what it had access to
- Fair comparison: doesn't penalize different retrieval results

**3. Pairwise Coherence** (Quality Metric)
- Compares clarity, structure, and logical flow
- Useful for comparing agent (multi-step) vs legacy (single-step) outputs

**4. Pairwise Completeness** (Coverage Metric)
- Compares which response more thoroughly addresses the question
- Catches cases where one system provides more comprehensive answers

### Why These 4?

- **Helpfulness**: Primary user-centric metric
- **Hallucination**: RAG-specific quality control
- **Coherence**: Differentiates agent's multi-step reasoning
- **Completeness**: Ensures we don't sacrifice coverage for brevity

### Out of Scope (YAGNI)

- ❌ Pairwise factual accuracy (use absolute evaluator instead)
- ❌ Pairwise IDK (binary metric, not comparative)
- ❌ Pairwise relevance (evaluates retrieval, not generation)
- ❌ Custom weighting schemes (premature optimization)

---

## 4. Integration with Registry: Separate Simple Registry

### Rationale

**Separate Registry** ✅ CHOSEN
- **KISS**: Simple dict mapping names to functions
- **YAGNI**: Don't complicate existing class-based registry
- **Clear separation**: Pairwise evaluators have different signature than standard evaluators
- **Easy discovery**: `get_all_pairwise_evaluators()` returns all available

**Integrate with Main Registry** ❌ REJECTED
- Would require changing registry to handle both classes and functions
- Standard evaluators use `(run, example)` signature; pairwise use `(runs, example)`
- Unnecessary complexity for 4 evaluators
- Violates KISS principle

### Implementation

In `pairwise_evaluators.py`:
```python
# Simple registry
_PAIRWISE_REGISTRY = {}

def register_pairwise(name: str):
    """Decorator to register pairwise evaluator."""
    def decorator(func):
        _PAIRWISE_REGISTRY[name] = func
        return func
    return decorator

@register_pairwise("pairwise_helpfulness")
def pairwise_helpfulness(runs, example):
    ...

def get_all_pairwise_evaluators():
    """Get all registered pairwise evaluators."""
    return list(_PAIRWISE_REGISTRY.values())
```

---

## 5. File Structure

### New Files (2)

**`src/history_book/evals/pairwise_prompts.py`** (~80 lines)
- 4 comparison prompts adapted from existing criteria prompts
- Reuses structure and wording from `criteria_prompts.py`
- Format: Takes {question}, {response_a}, {response_b}, {context_a}, {context_b}

**`src/history_book/evals/pairwise_evaluators.py`** (~120 lines)
- 4 pairwise evaluator functions
- Simple registry with decorator
- Helper functions for LLM judge invocation

### Modified Files (2)

**`scripts/run_evals.py`** (~50 new lines)
- Add `--pairwise` and `--experiments` arguments
- Add conditional branch for pairwise evaluation
- Import and use `evaluate_comparative` from LangSmith

**`src/history_book/evals/__init__.py`** (~5 new lines)
- Export `get_all_pairwise_evaluators` function
- Export individual pairwise evaluator functions

### Total New Code

- **~250 lines** total
- **4 files** modified/created
- **No changes** to base classes or existing evaluators

---

## 6. Detailed Implementation

### Step 1: Create Pairwise Prompts

**File:** `src/history_book/evals/pairwise_prompts.py`

```python
"""
Prompt templates for pairwise comparison evaluations.
"""

from langchain.prompts import PromptTemplate

PAIRWISE_HELPFULNESS_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which is more helpful.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Instructions: Compare these responses on helpfulness. Consider:
- Which response better answers the user's question?
- Which provides more relevant and useful information?
- Which is clearer and easier to understand?
- Which would be more helpful to the user?

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, state your judgment:
- "A" if Response A is more helpful
- "B" if Response B is more helpful
- "TIE" if both are equally helpful

On the final line, write only A, B, or TIE.

Reasoning:""")

PAIRWISE_HALLUCINATION_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which has fewer hallucinations.

User Question: {question}

Response A:
{response_a}

Context Available to Response A:
{context_a}

Response B:
{response_b}

Context Available to Response B:
{context_b}

Instructions: Compare these responses for hallucinations. For each response:
- Check if it makes claims not supported by ITS OWN context
- Look for invented facts, dates, names, or events
- Identify any contradictions with the provided context

NOTE: Each response should only be judged against its own retrieved context.
A response is NOT hallucinating just because it didn't retrieve the same documents.

First, provide your step-by-step reasoning.
Then, on the final line, state which response has FEWER hallucinations:
- "A" if Response A has fewer hallucinations (is better grounded)
- "B" if Response B has fewer hallucinations (is better grounded)
- "TIE" if both are equally well-grounded or equally hallucinated

On the final line, write only A, B, or TIE.

Reasoning:""")

PAIRWISE_COHERENCE_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which is more coherent.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Instructions: Compare these responses on coherence and structure. Consider:
- Which has clearer logical flow from one idea to the next?
- Which is better organized and structured?
- Which avoids contradictions within the response?
- Which presents ideas in a more understandable sequence?

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, state your judgment:
- "A" if Response A is more coherent
- "B" if Response B is more coherent
- "TIE" if both are equally coherent

On the final line, write only A, B, or TIE.

Reasoning:""")

PAIRWISE_COMPLETENESS_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which more completely answers the question.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Instructions: Compare these responses on completeness. Consider:
- Which addresses more aspects of the question?
- Which provides more thorough coverage of the topic?
- Which gives more comprehensive information?
- If the question has multiple parts, which addresses more parts?

Note: More complete doesn't necessarily mean longer. Focus on coverage of the question's scope.

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, state your judgment:
- "A" if Response A is more complete
- "B" if Response B is more complete
- "TIE" if both are equally complete

On the final line, write only A, B, or TIE.

Reasoning:""")
```

### Step 2: Create Pairwise Evaluators

**File:** `src/history_book/evals/pairwise_evaluators.py`

```python
"""
Pairwise evaluator implementations for comparing two experiment runs.
"""

from langchain_openai import ChatOpenAI

from history_book.evals.pairwise_prompts import (
    PAIRWISE_COHERENCE_PROMPT,
    PAIRWISE_COMPLETENESS_PROMPT,
    PAIRWISE_HALLUCINATION_PROMPT,
    PAIRWISE_HELPFULNESS_PROMPT,
)

# Simple registry for pairwise evaluators
_PAIRWISE_REGISTRY = {}


def register_pairwise(name: str):
    """Decorator to register a pairwise evaluator function."""

    def decorator(func):
        _PAIRWISE_REGISTRY[name] = func
        # Store name as attribute for LangSmith
        func.evaluator_name = name
        return func

    return decorator


def get_all_pairwise_evaluators():
    """Get all registered pairwise evaluator functions."""
    return list(_PAIRWISE_REGISTRY.values())


def get_pairwise_evaluator(name: str):
    """Get a specific pairwise evaluator by name."""
    return _PAIRWISE_REGISTRY.get(name)


# Shared LLM instance for all pairwise evaluators
_llm = None


def get_llm():
    """Get or create the LLM instance for pairwise evaluation."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1.0)
    return _llm


def _extract_winner(response_text: str) -> str:
    """
    Extract winner from LLM response.

    Args:
        response_text: LLM's reasoning + final judgment

    Returns:
        "A", "B", or "TIE"
    """
    # Check last few lines for the judgment
    lines = response_text.strip().split("\n")
    for line in reversed(lines[-3:]):  # Check last 3 lines
        cleaned = line.strip().upper()
        if cleaned in ("A", "B", "TIE"):
            return cleaned
    # Default to TIE if unclear
    return "TIE"


def _calculate_scores(winner: str, run_a_id: str, run_b_id: str) -> dict:
    """
    Convert winner to score dict.

    Args:
        winner: "A", "B", or "TIE"
        run_a_id: ID of run A
        run_b_id: ID of run B

    Returns:
        Dict mapping run IDs to scores (1.0 = winner, 0.0 = loser, 0.5 = tie)
    """
    if winner == "A":
        return {run_a_id: 1.0, run_b_id: 0.0}
    elif winner == "B":
        return {run_a_id: 0.0, run_b_id: 1.0}
    else:  # TIE
        return {run_a_id: 0.5, run_b_id: 0.5}


@register_pairwise("pairwise_helpfulness")
def pairwise_helpfulness(runs: list, example) -> dict:
    """
    Compare two responses for helpfulness.

    Args:
        runs: List of 2 Run objects from different experiments
        example: The evaluation example

    Returns:
        Evaluation result with scores for each run
    """
    run_a, run_b = runs

    # Extract data
    question = example.inputs.get("question", "")
    response_a = run_a.outputs.get("content", "")
    response_b = run_b.outputs.get("content", "")

    # Format prompt
    prompt_text = PAIRWISE_HELPFULNESS_PROMPT.format(
        question=question, response_a=response_a, response_b=response_b
    )

    # Get LLM judgment
    llm = get_llm()
    response = llm.invoke(prompt_text)
    winner = _extract_winner(response.content)

    # Calculate scores
    scores = _calculate_scores(winner, run_a.id, run_b.id)

    return {
        "key": "pairwise_helpfulness",
        "scores": scores,
        "comment": f"Winner: {winner} | {response.content[:200]}...",
    }


@register_pairwise("pairwise_hallucination")
def pairwise_hallucination(runs: list, example) -> dict:
    """
    Compare two responses for hallucinations (fewer is better).

    Each response is judged against its own retrieved context.

    Args:
        runs: List of 2 Run objects from different experiments
        example: The evaluation example

    Returns:
        Evaluation result with scores for each run
    """
    run_a, run_b = runs

    # Extract data
    question = example.inputs.get("question", "")
    response_a = run_a.outputs.get("content", "")
    response_b = run_b.outputs.get("content", "")

    # Get contexts (each run has its own retrieved context)
    context_a_list = run_a.outputs.get("retrieved_context", [])
    context_b_list = run_b.outputs.get("retrieved_context", [])

    # Format contexts as strings
    context_a = "\n\n".join(context_a_list) if context_a_list else "(No context retrieved)"
    context_b = "\n\n".join(context_b_list) if context_b_list else "(No context retrieved)"

    # Format prompt
    prompt_text = PAIRWISE_HALLUCINATION_PROMPT.format(
        question=question,
        response_a=response_a,
        context_a=context_a,
        response_b=response_b,
        context_b=context_b,
    )

    # Get LLM judgment
    llm = get_llm()
    response = llm.invoke(prompt_text)
    winner = _extract_winner(response.content)

    # Calculate scores
    scores = _calculate_scores(winner, run_a.id, run_b.id)

    return {
        "key": "pairwise_hallucination",
        "scores": scores,
        "comment": f"Fewer hallucinations: {winner} | {response.content[:200]}...",
    }


@register_pairwise("pairwise_coherence")
def pairwise_coherence(runs: list, example) -> dict:
    """
    Compare two responses for coherence and logical structure.

    Args:
        runs: List of 2 Run objects from different experiments
        example: The evaluation example

    Returns:
        Evaluation result with scores for each run
    """
    run_a, run_b = runs

    # Extract data
    question = example.inputs.get("question", "")
    response_a = run_a.outputs.get("content", "")
    response_b = run_b.outputs.get("content", "")

    # Format prompt
    prompt_text = PAIRWISE_COHERENCE_PROMPT.format(
        question=question, response_a=response_a, response_b=response_b
    )

    # Get LLM judgment
    llm = get_llm()
    response = llm.invoke(prompt_text)
    winner = _extract_winner(response.content)

    # Calculate scores
    scores = _calculate_scores(winner, run_a.id, run_b.id)

    return {
        "key": "pairwise_coherence",
        "scores": scores,
        "comment": f"More coherent: {winner} | {response.content[:200]}...",
    }


@register_pairwise("pairwise_completeness")
def pairwise_completeness(runs: list, example) -> dict:
    """
    Compare two responses for completeness in answering the question.

    Args:
        runs: List of 2 Run objects from different experiments
        example: The evaluation example

    Returns:
        Evaluation result with scores for each run
    """
    run_a, run_b = runs

    # Extract data
    question = example.inputs.get("question", "")
    response_a = run_a.outputs.get("content", "")
    response_b = run_b.outputs.get("content", "")

    # Format prompt
    prompt_text = PAIRWISE_COMPLETENESS_PROMPT.format(
        question=question, response_a=response_a, response_b=response_b
    )

    # Get LLM judgment
    llm = get_llm()
    response = llm.invoke(prompt_text)
    winner = _extract_winner(response.content)

    # Calculate scores
    scores = _calculate_scores(winner, run_a.id, run_b.id)

    return {
        "key": "pairwise_completeness",
        "scores": scores,
        "comment": f"More complete: {winner} | {response.content[:200]}...",
    }
```

### Step 3: Update run_evals.py

**File:** `scripts/run_evals.py`

Add imports:
```python
from langsmith.evaluation import evaluate_comparative
from history_book.evals import get_all_pairwise_evaluators
```

Add arguments (after line 29):
```python
parser.add_argument(
    "--pairwise",
    action="store_true",
    help="Run pairwise comparison between two experiments (requires --experiments)",
)
parser.add_argument(
    "--experiments",
    nargs=2,
    metavar=("EXP1", "EXP2"),
    help="Two experiment IDs or names to compare (required with --pairwise)",
)
```

Add validation (after line 34):
```python
# Validate pairwise arguments
if args.pairwise and not args.experiments:
    parser.error("--pairwise requires --experiments with 2 experiment IDs/names")
if args.experiments and not args.pairwise:
    parser.error("--experiments can only be used with --pairwise")
```

Add pairwise execution branch (after line 53):
```python
if args.pairwise:
    # Pairwise evaluation mode
    exp1, exp2 = args.experiments

    print(f"Running pairwise evaluation")
    print(f"Experiment 1: {exp1}")
    print(f"Experiment 2: {exp2}")

    ls_client = Client()

    # Get pairwise evaluators
    pairwise_evaluators = get_all_pairwise_evaluators()
    evaluator_names = [e.evaluator_name for e in pairwise_evaluators]

    print(f"Pairwise evaluators: {evaluator_names}")

    description = f"Pairwise comparison: {exp1} vs {exp2}"
    metadata = {
        "comparison_type": "pairwise",
        "experiment_1": exp1,
        "experiment_2": exp2,
        "evaluator_count": len(pairwise_evaluators),
    }

    results = evaluate_comparative(
        experiments=(exp1, exp2),
        evaluators=pairwise_evaluators,
        description=description,
        metadata=metadata,
        max_concurrency=5,
        client=ls_client,
    )

    print("\n✅ Pairwise evaluation complete!")
    print(f"Compared: {exp1} vs {exp2}")
    print("View results in LangSmith")

    return
```

### Step 4: Update __init__.py

**File:** `src/history_book/evals/__init__.py`

Add exports:
```python
from history_book.evals.pairwise_evaluators import (
    get_all_pairwise_evaluators,
    get_pairwise_evaluator,
    pairwise_coherence,
    pairwise_completeness,
    pairwise_hallucination,
    pairwise_helpfulness,
)

__all__ = [
    # ... existing exports ...
    "get_all_pairwise_evaluators",
    "get_pairwise_evaluator",
    "pairwise_helpfulness",
    "pairwise_hallucination",
    "pairwise_coherence",
    "pairwise_completeness",
]
```

---

## 7. Usage Examples

### Running Standard Evaluations (Unchanged)

```bash
# Agent system on full dataset
poetry run python scripts/run_evals.py --mode agent --full

# Legacy system on subset
poetry run python scripts/run_evals.py --mode legacy --subset
```

### Running Pairwise Evaluations (New)

```bash
# Compare two experiment runs
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments "agent-run-abc123" "legacy-run-def456"

# Using experiment names instead of IDs
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments "LangGraph agent - full dataset" "Legacy RAG - full dataset"
```

### Workflow for Comparison

```bash
# Step 1: Run both systems on same dataset
poetry run python scripts/run_evals.py --mode agent --full
# Note the experiment name/ID from LangSmith

poetry run python scripts/run_evals.py --mode legacy --full
# Note the experiment name/ID from LangSmith

# Step 2: Run pairwise comparison
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments "exp-id-1" "exp-id-2"

# Step 3: View results in LangSmith dashboard
# - Win rates per evaluator
# - Individual comparison feedback
# - LLM judge reasoning in comments
```

---

## 8. Testing Strategy

### Phase 1: Unit Testing (Manual)

Test each pairwise evaluator individually:

```python
# In Python REPL or notebook
from history_book.evals.pairwise_evaluators import pairwise_helpfulness
from langsmith.schemas import Run, Example

# Create mock runs
run_a = Run(id="a", outputs={"content": "Response A text"})
run_b = Run(id="b", outputs={"content": "Response B text"})
example = Example(inputs={"question": "Test question?"})

# Test evaluator
result = pairwise_helpfulness([run_a, run_b], example)
print(result)  # Should return scores dict
```

### Phase 2: Integration Testing

```bash
# Test with 2-query subset from existing experiments
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments "test-exp-1" "test-exp-2"
```

Verify:
- ✅ Evaluators run without errors
- ✅ Scores are valid (0.0, 0.5, or 1.0)
- ✅ Comments contain reasoning
- ✅ Results appear in LangSmith

### Phase 3: Full Comparison

```bash
# Run on full 100-query dataset
poetry run python scripts/run_evals.py --mode agent --full
poetry run python scripts/run_evals.py --mode legacy --full

# Compare the full experiments
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments <agent-exp-id> <legacy-exp-id>
```

Analyze:
- Win rate for each evaluator
- Agreement between pairwise and absolute metrics
- Edge cases where systems tie
- LLM judge reasoning quality

---

## 9. Analysis and Interpretation

### Win Rate Calculation

In LangSmith dashboard, for each evaluator:
- **System A wins**: Count of scores where A=1.0, B=0.0
- **System B wins**: Count of scores where A=0.0, B=1.0
- **Ties**: Count of scores where A=0.5, B=0.5
- **Win rate**: `wins_A / (wins_A + wins_B)` (excluding ties)

### Expected Insights

**Helpfulness**: Which system provides more useful answers?
**Hallucination**: Which system stays better grounded in context?
**Coherence**: Does agent's multi-step reasoning improve clarity?
**Completeness**: Does one system provide more thorough coverage?

### Cross-Reference with Absolute Metrics

Compare pairwise results with existing evaluators:
- If System A wins pairwise helpfulness but has lower absolute helpfulness scores → investigate discrepancy
- If hallucination pairwise matches absolute hallucination rates → confirms consistency
- Look for cases where metrics disagree to understand nuances

---

## 10. Future Enhancements (Out of Scope)

**Explicitly marked as YAGNI for this implementation:**

1. ❌ Statistical significance testing (Binomial test, McNemar's test)
2. ❌ Multi-way comparisons (3+ systems at once)
3. ❌ Custom evaluator weighting schemes
4. ❌ Automated win rate calculation scripts
5. ❌ PairwiseEvaluator base class (only needed if we add 10+ evaluators)
6. ❌ Integration with main evaluator registry (different signatures)
7. ❌ Pairwise document count comparison (not meaningful for pairwise)
8. ❌ Randomized order to reduce positional bias (LangSmith may handle this)

Can be added later if needed, but not required for initial implementation.

---

## 11. Success Criteria

### Implementation Complete When:

✅ All 4 files created/modified
✅ 4 pairwise evaluators implemented
✅ `run_evals.py` accepts `--pairwise --experiments` flags
✅ Registry exports pairwise evaluators
✅ Code passes `ruff check` and `ruff format`
✅ Manual testing confirms evaluators run

### Validation Complete When:

✅ Pairwise evaluation runs on 2 real experiment IDs
✅ Results appear in LangSmith dashboard
✅ Win rates can be calculated from scores
✅ LLM judge reasoning is sensible and helpful
✅ No errors during execution

---

## 12. Dependencies and Requirements

### Existing Dependencies (Already Installed)
- ✅ `langsmith` - For `evaluate_comparative()` method
- ✅ `langchain-openai` - For LLM judge (gpt-5-mini)
- ✅ `langchain` - For prompt templates

### Environment Variables (Already Configured)
- ✅ `LANGSMITH_API_KEY` - For LangSmith API access
- ✅ `OPENAI_API_KEY` - For LLM judge

### No New Dependencies Required

---

## 13. Rollout Plan

### Step 1: Implementation (1-2 hours)
1. Create `pairwise_prompts.py` with 4 prompts
2. Create `pairwise_evaluators.py` with 4 evaluators + registry
3. Modify `run_evals.py` with pairwise branch
4. Update `__init__.py` exports
5. Run `ruff check` and `ruff format`

### Step 2: Testing (30 mins)
1. Unit test each evaluator function
2. Integration test with existing experiment IDs
3. Verify LangSmith dashboard shows results

### Step 3: Full Comparison (Overnight)
1. Run `--mode agent --full` experiment
2. Run `--mode legacy --full` experiment
3. Run pairwise comparison between them
4. Analyze results and win rates

### Step 4: Documentation (15 mins)
1. Update `/src/history_book/evals/CLAUDE.md` with pairwise section
2. Add usage examples to README or docs

---

## 14. Key Design Principles Applied

### DRY (Don't Repeat Yourself)
- ✅ Reuses `run_evals.py` instead of new script
- ✅ Adapts existing prompt patterns from `criteria_prompts.py`
- ✅ Shares LLM instance across evaluators
- ✅ Extracts common logic (`_extract_winner`, `_calculate_scores`)

### YAGNI (You Aren't Gonna Need It)
- ✅ No `PairwiseEvaluator` base class (only 4 evaluators)
- ✅ No statistical significance testing (can add later if needed)
- ✅ No custom weighting (simple scores sufficient)
- ✅ No multi-way comparisons (only need 2-way for now)
- ✅ Separate registry instead of complex integration

### KISS (Keep It Simple, Stupid)
- ✅ Plain functions instead of class hierarchies
- ✅ Simple decorator-based registry
- ✅ Clear, explicit prompts without complex templating
- ✅ Straightforward winner extraction logic
- ✅ Single entry point with flag instead of multiple scripts

---

## 15. Risk Mitigation

### Potential Issues and Solutions

**Issue**: LLM judge gives inconsistent results
- **Mitigation**: Use temperature=1.0 for gpt-5-mini (as per existing evaluators)
- **Fallback**: Can experiment with temperature or try different models

**Issue**: Experiment IDs are hard to find
- **Mitigation**: LangSmith dashboard shows experiment names and IDs
- **Enhancement**: Could add `--list-experiments` flag in future (YAGNI for now)

**Issue**: Context formatting differs between systems
- **Mitigation**: Both systems use same `retrieved_context` format from `target_wrapper`
- **Verification**: Already tested in Phase 2b refactor

**Issue**: Pairwise results contradict absolute metrics
- **Expected**: Different evaluation approaches may disagree
- **Response**: Use both types of evaluation for comprehensive view

---

## 16. Conclusion

This plan implements LangSmith pairwise evaluations following DRY, YAGNI, and KISS principles:

- **~250 lines of new code** across 4 files
- **4 high-value pairwise evaluators** (helpfulness, hallucination, coherence, completeness)
- **Extends existing script** instead of creating new one
- **Simple functions** instead of complex class hierarchies
- **Clear separation** from existing evaluator infrastructure

The implementation enables direct comparison of legacy LCEL RAG vs agent-based RAG systems through LLM-judged pairwise evaluations, complementing the existing absolute evaluation metrics.

Ready for implementation when approved.
