# CLAUDE.md - RAG Evaluation Framework

Evaluation framework for measuring RAG chat quality. Integrates with LangSmith for experiment tracking.

## Quick Start

```bash
source source_env.sh

# Single-run evaluations
poetry run python scripts/run_evals.py

# Pairwise comparison (pass two experiment names)
poetry run python scripts/run_evals.py --pairwise <experiment_a> <experiment_b>
```

Results are uploaded to LangSmith ("History Book Eval Queries" dataset, 100 queries).

## Architecture

```
src/history_book/evals/
├── base.py                        # Base classes + parse helpers
├── __init__.py                    # Re-exports
└── evaluators/
    ├── __init__.py                # Builder functions
    ├── helpfulness.py
    ├── factual_accuracy.py
    ├── coherence.py
    ├── hallucination.py
    ├── idk.py
    ├── idk_appropriate.py
    ├── relevance.py
    ├── document_count.py          # FunctionEvaluator (no LLM)
    ├── pairwise_helpfulness.py
    ├── pairwise_hallucination.py
    ├── pairwise_coherence.py
    └── pairwise_completeness.py
```

## Base Classes (`base.py`)

### `LLMEvaluator`

For Y/N verdict evaluations. Owns `prompt` and `format_prompt()` inline — no separate prompt files.

```python
class HelpfulnessEvaluator(LLMEvaluator):
    name = "helpfulness"
    prompt = "..."  # inline

    def format_prompt(self, outputs, inputs) -> str:
        return self.prompt.format(
            question=inputs["question"],
            answer=outputs["answer"],
        )
```

### `FunctionEvaluator`

For deterministic metrics (no LLM call). Override `evaluate()` directly.

```python
class DocumentCountEvaluator(FunctionEvaluator):
    name = "document_count"

    async def evaluate(self, outputs, inputs) -> EvalResult:
        n = len(outputs.get("retrieved_paragraphs", []))
        return {"key": self.name, "score": min(n / 5, 1.0)}
```

### `PairwiseEvaluator`

For A/B comparisons via `evaluate_comparative`. Owns `prompt` and `format_prompt(run_a, run_b, example)`.
Returns `{key, scores: {run_id: float}}`. Verdict parsed with `parse_abtie`.

### `EvalResult` (TypedDict)

```python
class EvalResult(TypedDict, total=False):
    key: str
    score: float
    value: str | None   # raw verdict "Y" / "N"
    comment: str | None
```

### Parse helpers

- `parse_yn(text)` → `(score, verdict)` — extracts last alphabetic token from last non-empty line. Handles "Final judgment: Y", "**Y**", etc.
- `parse_abtie(text)` → `"A" | "B" | "TIE"` — same pattern for pairwise.

### `as_langsmith()`

Both `Evaluator` and `PairwiseEvaluator` have `as_langsmith()` which wraps the evaluator in the function shape LangSmith's `aevaluate` / `evaluate_comparative` expects.

## Builder Functions (`evaluators/__init__.py`)

```python
from history_book.evals.evaluators import (
    build_llm_evaluators,
    build_function_evaluators,
    build_pairwise_evaluators,
)

llm = init_chat_model("openai:gpt-4o-mini")
single_run = build_llm_evaluators(llm) + build_function_evaluators()
pairwise   = build_pairwise_evaluators(llm)
```

## Single-Run Evaluators

| Name | Class | Measures |
|------|-------|---------|
| `helpfulness` | `HelpfulnessEvaluator` | Did the answer actually help with the question? |
| `factual_accuracy` | `FactualAccuracyEvaluator` | Are facts correct vs. the retrieved context? |
| `coherence` | `CoherenceEvaluator` | Is the answer well-structured and readable? |
| `hallucination` | `HallucinationEvaluator` | Does the answer introduce unsupported claims? |
| `idk` | `IdkEvaluator` | Did the answer appropriately admit uncertainty? |
| `idk_appropriate` | `IdkAppropriateEvaluator` | Was the IDK response warranted given context? |
| `relevance` | `RelevanceEvaluator` | Is the answer on-topic for the question? |
| `document_count` | `DocumentCountEvaluator` | How many paragraphs were retrieved? (no LLM) |

## Pairwise Evaluators

| Name | Class |
|------|-------|
| `pairwise_helpfulness` | `PairwiseHelpfulnessEvaluator` |
| `pairwise_hallucination` | `PairwiseHallucinationEvaluator` |
| `pairwise_coherence` | `PairwiseCoherenceEvaluator` |
| `pairwise_completeness` | `PairwiseCompletenessEvaluator` |

## Adding a New Evaluator

1. Create `evaluators/my_evaluator.py`:

```python
from history_book.evals.base import LLMEvaluator

class MyEvaluator(LLMEvaluator):
    name = "my_metric"
    prompt = """You are evaluating a RAG response.

Question: {question}
Answer: {answer}

Does the answer X? Reply Y or N on the last line."""

    def format_prompt(self, outputs, inputs):
        return self.prompt.format(
            question=inputs["question"],
            answer=outputs["answer"],
        )
```

2. Add to `evaluators/__init__.py` — import and append to `LLM_EVALUATOR_CLASSES`.

## Related Files

- `scripts/run_evals.py` — entry point; builds evaluators, calls `aevaluate` / `evaluate_comparative`
- `services/chat_service.py` — `get_eval_metadata()` returns LLM config for experiment metadata
