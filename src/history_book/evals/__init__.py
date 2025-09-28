"""
Evaluation framework for the History Book application.

This module provides a standardized way to define, register, and run evaluations
on the RAG system's performance.
"""

from history_book.evals.base import BaseEvaluator
from history_book.evals.evaluators import (
    CoherenceEvaluator,
    FactualAccuracyEvaluator,
    HallucinationEvaluator,
    HelpfulnessEvaluator,
    IdkAppropriateEvaluator,
    IdkEvaluator,
    RelevanceEvaluator,
)
from history_book.evals.registry import (
    get_all_evaluators,
    get_evaluator,
    list_evaluators,
    register_evaluator,
)

__all__ = [
    "BaseEvaluator",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
    "get_all_evaluators",
    "CoherenceEvaluator",
    "FactualAccuracyEvaluator",
    "HallucinationEvaluator",
    "HelpfulnessEvaluator",
    "IdkAppropriateEvaluator",
    "IdkEvaluator",
    "RelevanceEvaluator",
]
