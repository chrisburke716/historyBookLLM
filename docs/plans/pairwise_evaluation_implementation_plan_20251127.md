# LangSmith Pairwise Evaluation Implementation Plan

**Date:** 2025-11-27  
**Author:** Claude Code  
**Status:** Planning Phase

## Executive Summary

This plan implements pairwise evaluations for comparing two RAG system experiments in LangSmith. Following DRY/YAGNI/KISS principles, we extend the existing `run_evals.py` script with `--pairwise` mode and create 3-4 simple pairwise evaluator functions.

**Key Decision:** Different experiments may retrieve different context for the same query. Pairwise evaluators will compare each response against its **own** retrieved context, then determine which performed better.

---

## Table of Contents

1. [Architecture Decision](#architecture-decision)
2. [Handling Retrieved Context](#handling-retrieved-context)
3. [File Structure](#file-structure)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Critical Files](#critical-files)

---

## Architecture Decision

### Recommendation: Extend `run_evals.py` (Option A)

**Rationale (DRY/YAGNI/KISS):**

1. **DRY**: Reuses 95% of existing code (CLI parsing, LLM setup, dataset access, metadata extraction)
2. **YAGNI**: No need for separate script when one flag handles the difference
3. **KISS**: Single entrypoint for all evaluations - users remember one command

**Implementation:**
- Add `--pairwise` flag + `--experiments EXP_A EXP_B` args
- Conditional branch: if pairwise, call `evaluate()` with experiments list
- Mutually exclusive with `--mode agent/legacy`

---

## Handling Retrieved Context

### The Challenge

When comparing two experiments (e.g., legacy LCEL vs agent RAG), the same query may retrieve **different contexts**:

```python
# Experiment A (LCEL) output for "What caused WWI?"
{
    "content": "WWI was caused by...",
    "retrieved_context": [
        "[Book 1, Chapter 5, Page 42] Text about assassination of Archduke...",
        "[Book 1, Chapter 5, Page 43] Text about alliance systems..."
    ]
}

# Experiment B (Agent) output for same query
{
    "content": "The causes of WWI included...",
    "retrieved_context": [
        "[Book 1, Chapter 6, Page 50] Text about militarism...",
        "[Book 1, Chapter 5, Page 42] Text about assassination..." # Overlap possible
    ]
}
```

###

 Solution: Compare Each Response Against Its Own Context

**Approach for Hallucination Evaluator:**

1. Evaluate Response A hallucinations against Context A
2. Evaluate Response B hallucinations against Context B  
3. Compare: which response had fewer/no hallucinations relative to its context?
4. Return winner

**Prompt Pattern:**
```
Response A: {answer_a}
Context A (what Response A had access to): {context_a}

Response B: {answer_b}
Context B (what Response B had access to): {context_b}

Which response is better grounded in its available context?
- Response A: hallucinations relative to Context A?
- Response B: hallucinations relative to Context B?

Return [1, 0] if A is better grounded, [0, 1] if B is better grounded, [0.5, 0.5] if equal.
```

### Why This Approach is Fair

- Each system is judged by what **it** retrieved and had available
- Doesn't penalize System A for not using documents it never retrieved
- Evaluates "did the system hallucinate given its context?" not "did it retrieve the same docs?"
- Aligns with real-world usage: users care about accuracy given what was found, not theoretical best retrieval

---

## File Structure

### Files to Create

#### 1. `src/history_book/evals/pairwise_evaluators.py` (NEW)
Pairwise evaluator functions with simple registry.

#### 2. `src/history_book/evals/pairwise_prompts.py` (NEW)  
Comparative prompts adapted from existing criteria_prompts.py.

### Files to Modify

#### 3. `scripts/run_evals.py`
Add CLI args and pairwise execution branch.

#### 4. `src/history_book/evals/__init__.py`
Export `get_pairwise_evaluators()`.

---

## Implementation Details

### Step 1: Create Pairwise Prompts

**File:** `/Users/chris/Desktop/historyBook/history_book/src/history_book/evals/pairwise_prompts.py`

```python
"""
Prompt templates for pairwise evaluations.

These prompts compare two responses to determine which is better.
"""

from langchain.prompts import PromptTemplate

PAIRWISE_HELPFULNESS_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which is more helpful to the user.

User Question: {question}

Response A: {answer_a}
Response B: {answer_b}

Compare the two responses based on helpfulness:
1. Which response better addresses the user's question?
2. Which provides more actionable, useful, or complete information?
3. Which is more relevant and directly answers what was asked?

Provide your step-by-step reasoning, then return scores as a JSON object:
{{"scores": [score_a, score_b], "reasoning": "your reasoning here"}}

Scoring:
- [1, 0] if Response A is more helpful
- [0, 1] if Response B is more helpful  
- [0.5, 0.5] if equally helpful

Your response:
""")

PAIRWISE_HALLUCINATION_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which has fewer hallucinations relative to their retrieved contexts.

User Question: {question}

Response A: {answer_a}
Retrieved Context A (what Response A had access to): {context_a}

Response B: {answer_b}
Retrieved Context B (what Response B had access to): {context_b}

For each response, check for hallucinations - factual claims not supported by its own retrieved context:
1. Does Response A make claims not in Context A?
2. Does Response B make claims not in Context B?
3. Which response is better grounded in what it had available?

Provide your step-by-step reasoning, then return scores as a JSON object:
{{"scores": [score_a, score_b], "reasoning": "your reasoning here"}}

Scoring:
- [1, 0] if Response A has fewer hallucinations (better grounded in Context A)
- [0, 1] if Response B has fewer hallucinations (better grounded in Context B)
- [0.5, 0.5] if both equally grounded or equally hallucinated

Your response:
""")

PAIRWISE_COHERENCE_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which is more coherent and well-structured.

User Question: {question}

Response A: {answer_a}
Response B: {answer_b}

Compare the two responses based on coherence:
1. Which has clearer logical flow from one idea to the next?
2. Which is better organized and structured?
3. Which has fewer contradictions or confusing transitions?

Provide your step-by-step reasoning, then return scores as a JSON object:
{{"scores": [score_a, score_b], "reasoning": "your reasoning here"}}

Scoring:
- [1, 0] if Response A is more coherent
- [0, 1] if Response B is more coherent
- [0.5, 0.5] if equally coherent

Your response:
""")

PAIRWISE_COMPLETENESS_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which provides a more complete answer.

User Question: {question}

Response A: {answer_a}
Response B: {answer_b}

Compare the two responses based on completeness:
1. Which addresses more aspects of the question?
2. Which provides more comprehensive coverage of the topic?
3. Which leaves fewer important points unaddressed?

Provide your step-by-step reasoning, then return scores as a JSON object:
{{"scores": [score_a, score_b], "reasoning": "your reasoning here"}}

Scoring:
- [1, 0] if Response A is more complete
- [0, 1] if Response B is more complete
- [0.5, 0.5] if equally complete

Your response:
""")
```

### Step 2: Create Pairwise Evaluators

**File:** `/Users/chris/Desktop/historyBook/history_book/src/history_book/evals/pairwise_evaluators.py`

```python
"""
Pairwise evaluator functions for comparing two experiment outputs.

These evaluators receive inputs (query) and outputs (list of 2 dicts) and return
comparative scores indicating which output is better.
"""

import json
from typing import Any

from langchain_openai import ChatOpenAI

from history_book.evals.pairwise_prompts import (
    PAIRWISE_COHERENCE_PROMPT,
    PAIRWISE_COMPLETENESS_PROMPT,
    PAIRWISE_HALLUCINATION_PROMPT,
    PAIRWISE_HELPFULNESS_PROMPT,
)

# Simple registry for pairwise evaluators
_PAIRWISE_EVALUATORS = {}


def pairwise_evaluator(name: str):
    """
    Decorator to register pairwise evaluator functions.
    
    Usage:
        @pairwise_evaluator("pairwise_helpfulness")
        def create_helpfulness_evaluator(llm):
            ...
    """

    def decorator(func):
        _PAIRWISE_EVALUATORS[name] = func
        func.eval_name = name
        return func

    return decorator


def get_pairwise_evaluators(llm: ChatOpenAI) -> list:
    """
    Get all registered pairwise evaluator functions with LLM bound.
    
    Args:
        llm: ChatOpenAI instance to use for evaluation
        
    Returns:
        List of evaluator functions ready to use with evaluate()
    """
    return [create_func(llm) for create_func in _PAIRWISE_EVALUATORS.values()]


def list_pairwise_evaluators() -> list[str]:
    """List names of all registered pairwise evaluators."""
    return list(_PAIRWISE_EVALUATORS.keys())


@pairwise_evaluator("pairwise_helpfulness")
def create_helpfulness_evaluator(llm: ChatOpenAI):
    """Create a pairwise helpfulness evaluator."""
    chain = PAIRWISE_HELPFULNESS_PROMPT | llm

    def evaluator(inputs: dict, outputs: list[dict]) -> dict[str, Any]:
        """
        Compare two responses for helpfulness.
        
        Args:
            inputs: Dict with "question" key
            outputs: List of 2 dicts, each with "content" and "retrieved_context"
            
        Returns:
            Dict with "key", "scores" list, and "comment"
        """
        try:
            result = chain.invoke({
                "question": inputs.get("question", ""),
                "answer_a": outputs[0].get("content", ""),
                "answer_b": outputs[1].get("content", ""),
            })
            
            # Parse LLM JSON response
            content = result.content if hasattr(result, 'content') else str(result)
            parsed = json.loads(content)
            scores = parsed.get("scores", [0.5, 0.5])
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            return {
                "key": "pairwise_helpfulness",
                "scores": scores,
                "comment": reasoning,
            }
        except Exception as e:
            # Fallback on error
            return {
                "key": "pairwise_helpfulness",
                "scores": [0.5, 0.5],
                "comment": f"Error in evaluation: {str(e)}",
            }

    evaluator.name = "pairwise_helpfulness"
    return evaluator


@pairwise_evaluator("pairwise_hallucination")
def create_hallucination_evaluator(llm: ChatOpenAI):
    """Create a pairwise hallucination evaluator."""
    chain = PAIRWISE_HALLUCINATION_PROMPT | llm

    def evaluator(inputs: dict, outputs: list[dict]) -> dict[str, Any]:
        """
        Compare two responses for hallucinations against their own retrieved contexts.
        
        Args:
            inputs: Dict with "question" key
            outputs: List of 2 dicts, each with "content" and "retrieved_context"
            
        Returns:
            Dict with "key", "scores" list, and "comment"
        """
        try:
            # Format contexts as strings
            context_a = "\n\n".join(outputs[0].get("retrieved_context", []))
            context_b = "\n\n".join(outputs[1].get("retrieved_context", []))
            
            result = chain.invoke({
                "question": inputs.get("question", ""),
                "answer_a": outputs[0].get("content", ""),
                "context_a": context_a,
                "answer_b": outputs[1].get("content", ""),
                "context_b": context_b,
            })
            
            # Parse LLM JSON response
            content = result.content if hasattr(result, 'content') else str(result)
            parsed = json.loads(content)
            scores = parsed.get("scores", [0.5, 0.5])
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            return {
                "key": "pairwise_hallucination",
                "scores": scores,
                "comment": reasoning,
            }
        except Exception as e:
            return {
                "key": "pairwise_hallucination",
                "scores": [0.5, 0.5],
                "comment": f"Error in evaluation: {str(e)}",
            }

    evaluator.name = "pairwise_hallucination"
    return evaluator


@pairwise_evaluator("pairwise_coherence")
def create_coherence_evaluator(llm: ChatOpenAI):
    """Create a pairwise coherence evaluator."""
    chain = PAIRWISE_COHERENCE_PROMPT | llm

    def evaluator(inputs: dict, outputs: list[dict]) -> dict[str, Any]:
        """Compare two responses for coherence and logical structure."""
        try:
            result = chain.invoke({
                "question": inputs.get("question", ""),
                "answer_a": outputs[0].get("content", ""),
                "answer_b": outputs[1].get("content", ""),
            })
            
            content = result.content if hasattr(result, 'content') else str(result)
            parsed = json.loads(content)
            scores = parsed.get("scores", [0.5, 0.5])
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            return {
                "key": "pairwise_coherence",
                "scores": scores,
                "comment": reasoning,
            }
        except Exception as e:
            return {
                "key": "pairwise_coherence",
                "scores": [0.5, 0.5],
                "comment": f"Error in evaluation: {str(e)}",
            }

    evaluator.name = "pairwise_coherence"
    return evaluator


@pairwise_evaluator("pairwise_completeness")
def create_completeness_evaluator(llm: ChatOpenAI):
    """Create a pairwise completeness evaluator."""
    chain = PAIRWISE_COMPLETENESS_PROMPT | llm

    def evaluator(inputs: dict, outputs: list[dict]) -> dict[str, Any]:
        """Compare two responses for completeness of answer."""
        try:
            result = chain.invoke({
                "question": inputs.get("question", ""),
                "answer_a": outputs[0].get("content", ""),
                "answer_b": outputs[1].get("content", ""),
            })
            
            content = result.content if hasattr(result, 'content') else str(result)
            parsed = json.loads(content)
            scores = parsed.get("scores", [0.5, 0.5])
            reasoning = parsed.get("reasoning", "No reasoning provided")
            
            return {
                "key": "pairwise_completeness",
                "scores": scores,
                "comment": reasoning,
            }
        except Exception as e:
            return {
                "key": "pairwise_completeness",
                "scores": [0.5, 0.5],
                "comment": f"Error in evaluation: {str(e)}",
            }

    evaluator.name = "pairwise_completeness"
    return evaluator
```

### Step 3: Modify run_evals.py

**File:** `/Users/chris/Desktop/historyBook/history_book/scripts/run_evals.py`

**Changes:**

```python
# Add to imports
from history_book.evals import (
    get_function_evaluators,
    get_pairwise_evaluators,  # NEW
    get_prompt_evaluators,
)

# Modify argparse
parser.add_argument(
    "--pairwise",
    action="store_true",
    help="Run pairwise comparison between two experiments (requires --experiments)",
)
parser.add_argument(
    "--experiments",
    nargs=2,
    metavar=("EXP_A", "EXP_B"),
    help="Two experiment IDs/names to compare (required with --pairwise)",
)

# Add validation after args parsing
if args.pairwise:
    if not args.experiments:
        parser.error("--pairwise requires --experiments with 2 experiment IDs")
    if args.mode:
        parser.error("--pairwise and --mode are mutually exclusive")
    if args.subset or args.full:
        parser.error("--pairwise does not use --subset/--full (compares existing experiments)")

# Add pairwise execution branch before existing evaluation logic
if args.pairwise:
    print(f"\nRunning pairwise comparison:")
    print(f"  Experiment A: {args.experiments[0]}")
    print(f"  Experiment B: {args.experiments[1]}")
    
    # Create pairwise evaluators
    pairwise_evals = get_pairwise_evaluators(llm=llm)
    eval_names = [e.name for e in pairwise_evals]
    print(f"  Evaluators: {eval_names}")
    
    # Run pairwise evaluation using evaluate() with experiments list
    _eval = await ls_client.evaluate(
        target=args.experiments,  # List of 2 experiment IDs
        evaluators=pairwise_evals,
        experiment_prefix="Pairwise Comparison",
        description=f"Comparing {args.experiments[0]} vs {args.experiments[1]}",
        metadata={
            "evaluation_type": "pairwise",
            "experiment_a": args.experiments[0],
            "experiment_b": args.experiments[1],
            "evaluator_llm_model": llm.model_name,
            "evaluator_llm_temperature": llm.temperature,
        },
        max_concurrency=5,
    )
    
    print("\n✅ Pairwise evaluation complete!")
    print(f"Compared: {args.experiments[0]} vs {args.experiments[1]}")
    print("View results in LangSmith")
    
else:
    # Existing standard evaluation logic (unchanged)
    ...
```

### Step 4: Update __init__.py

**File:** `/Users/chris/Desktop/historyBook/history_book/src/history_book/evals/__init__.py`

**Add:**
```python
from history_book.evals.pairwise_evaluators import (
    get_pairwise_evaluators,
    list_pairwise_evaluators,
)

__all__ = [
    # ... existing exports ...
    "get_pairwise_evaluators",
    "list_pairwise_evaluators",
]
```

---

## Usage Examples

### Get Experiment IDs from LangSmith

First, find the experiment IDs you want to compare:

```bash
# In LangSmith UI, navigate to your project
# Click on "Experiments" tab
# Copy the experiment IDs or names you want to compare

# Example IDs:
# - Experiment A (agent): "exp-abc123def"
# - Experiment B (legacy): "exp-xyz789ghi"
```

### Run Pairwise Comparison

```bash
# Basic pairwise comparison
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments exp-abc123def exp-xyz789ghi

# With descriptive names (if experiments have names)
poetry run python scripts/run_evals.py \
    --pairwise \
    --experiments "LangGraph Agent - Nov 27" "Legacy LCEL - Nov 20"
```

### View Results

1. Go to LangSmith dashboard
2. Navigate to your project
3. Find the "Pairwise Comparison" experiment
4. View which system won more comparisons across evaluators
5. Drill into individual examples to see reasoning

---

## Why This Design?

### DRY (Don't Repeat Yourself)

**Reused Components:**
- CLI framework from run_evals.py
- LLM setup pattern
- LangSmith client initialization
- Metadata structure
- Prompt adaptation from existing criteria

**Avoided Duplication:**
- No separate script with copied boilerplate
- No duplicate evaluator infrastructure
- Single source of truth for evaluation execution

### YAGNI (You Aren't Gonna Need It)

**What We're NOT Building:**
- PairwiseEvaluator base class (only 4 evaluators, classes add complexity)
- Complex registry integration (simple dict is sufficient)
- Automatic experiment discovery (users know which to compare)
- >2 experiment comparison (not requested)
- Statistical significance testing (premature)

**Why:**
- Each adds complexity without clear current benefit
- Can add later if needed (incremental approach)
- Simpler code is easier to understand and maintain

### KISS (Keep It Simple, Stupid)

**Simple Decisions:**
- Functions over classes for evaluators
- Flag-based CLI (--pairwise) instead of separate command
- Direct JSON parsing over complex abstraction
- Explicit error handling with fallback scores
- Single file per concern (prompts, evaluators)

**Complexity Avoided:**
- No inheritance hierarchies
- No factory patterns
- No dependency injection frameworks
- No metaprogramming

---

## Testing Strategy

### Manual Testing Checklist

1. **Setup:**
   ```bash
   source source_env.sh
   # Ensure OPENAI_API_KEY and LANGSMITH_API_KEY are set
   ```

2. **Run pairwise evaluation:**
   ```bash
   poetry run python scripts/run_evals.py \
       --pairwise \
       --experiments <exp_a_id> <exp_b_id>
   ```

3. **Validation:**
   - Check LangSmith UI for new "Pairwise Comparison" experiment
   - Verify 4 feedback keys: pairwise_helpfulness, pairwise_hallucination, pairwise_coherence, pairwise_completeness
   - Inspect individual example scores and reasoning
   - Confirm scores are [1,0], [0,1], or [0.5,0.5]

4. **Error Handling:**
   - Test with invalid experiment IDs (should fail gracefully)
   - Test with missing --experiments (should show error message)
   - Test with --pairwise and --mode together (should reject)

### Validation Criteria

- [ ] All 4 pairwise evaluators complete without errors
- [ ] Scores follow expected format: [1,0], [0,1], or [0.5,0.5]
- [ ] Reasoning in comments is coherent and explains choice
- [ ] Hallucination evaluator correctly references context A vs context B
- [ ] Results visible in LangSmith UI
- [ ] Can drill into individual examples and see comparative reasoning

---

## Future Enhancements (Out of Scope)

These are explicitly NOT part of this implementation (YAGNI):

1. **PairwiseEvaluator base class**
   - Wait until we have 10+ pairwise evaluators
   - Current 4 evaluators don't justify the abstraction overhead

2. **Integration with main evaluator registry**
   - Different signatures make unification complex
   - Separate registries are simpler and clearer

3. **Statistical significance testing**
   - Requires multiple runs and statistical frameworks
   - Users can do this analysis in LangSmith UI

4. **Automatic experiment discovery**
   - Users know which experiments they want to compare
   - Auto-discovery adds complexity for unclear benefit

5. **Custom prompt templates**
   - Wait for user feedback on existing prompts
   - Current prompts cover main use cases

6. **Multi-experiment comparison (>2)**
   - LangSmith supports 2-way comparison
   - Multi-way adds significant complexity

---

## References

**LangSmith Documentation:**
- [How to run pairwise evaluations](https://docs.langchain.com/langsmith/evaluate-pairwise)
- [evaluate_comparative API Reference](https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._runner.evaluate_comparative)
- [Pairwise Evaluations Blog Post](https://blog.langchain.com/pairwise-evaluations-with-langsmith/)
- [LangSmith Cookbook - Comparing Runs](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/comparing-runs/comparing-qa.ipynb)

**Internal Documentation:**
- `/src/history_book/evals/CLAUDE.md` - Existing evaluation framework
- `/scripts/run_evals.py` - Current evaluation script

---

## Success Criteria

This implementation is successful if:

1. **Users can run pairwise comparisons with a simple command**
   ```bash
   poetry run python scripts/run_evals.py --pairwise --experiments A B
   ```

2. **Results clearly show which system performed better**
   - Win/loss counts per evaluator
   - Individual example reasoning
   - Comparative scores in LangSmith UI

3. **Hallucination evaluator correctly handles different contexts**
   - Response A compared against Context A
   - Response B compared against Context B
   - Fair comparison of grounding quality

4. **Code follows project conventions**
   - Passes `poetry run ruff check`
   - Passes `poetry run ruff format`
   - Consistent with existing evaluator patterns

5. **Total implementation <250 lines of new code**
   - pairwise_prompts.py: ~80 lines
   - pairwise_evaluators.py: ~150 lines
   - run_evals.py modifications: ~40 lines
   - __init__.py modifications: ~5 lines

