"""
Evaluation framework for the History Book application.

This module provides a standardized way to define, register, and run evaluations
on the RAG system's performance.
"""

from history_book.evals.base import BaseEvaluator, FunctionEvaluator
from history_book.evals.evaluators import (
    CoherenceEvaluator,
    DocumentCountEvaluator,
    FactualAccuracyEvaluator,
    HallucinationEvaluator,
    HelpfulnessEvaluator,
    IdkAppropriateEvaluator,
    IdkEvaluator,
    RelevanceEvaluator,
)
from history_book.evals.pairwise_evaluators import (
    get_all_pairwise_evaluators,
    get_pairwise_evaluator,
)
from history_book.evals.registry import (
    get_all_evaluators,
    get_evaluator,
    get_function_evaluators,
    get_prompt_evaluators,
    list_evaluators,
    register_evaluator,
)

__all__ = [
    "BaseEvaluator",
    "FunctionEvaluator",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
    "get_all_evaluators",
    "get_prompt_evaluators",
    "get_function_evaluators",
    "CoherenceEvaluator",
    "DocumentCountEvaluator",
    "FactualAccuracyEvaluator",
    "HallucinationEvaluator",
    "HelpfulnessEvaluator",
    "IdkAppropriateEvaluator",
    "IdkEvaluator",
    "RelevanceEvaluator",
    "get_all_pairwise_evaluators",
    "get_pairwise_evaluator",
]
