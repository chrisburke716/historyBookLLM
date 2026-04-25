"""Evaluation framework for the History Book RAG system.

Each evaluator owns a single prompt + LLM call (or a direct computation).
`as_langsmith()` wraps an evaluator for use with `langsmith.evaluation`.
"""

from history_book.evals.base import (
    EvalResult,
    Evaluator,
    FunctionEvaluator,
    LLMEvaluator,
    PairwiseEvaluator,
    PairwiseResult,
)
from history_book.evals.evaluators import (
    build_function_evaluators,
    build_llm_evaluators,
    build_pairwise_evaluators,
)

__all__ = [
    "EvalResult",
    "Evaluator",
    "FunctionEvaluator",
    "LLMEvaluator",
    "PairwiseEvaluator",
    "PairwiseResult",
    "build_function_evaluators",
    "build_llm_evaluators",
    "build_pairwise_evaluators",
]
