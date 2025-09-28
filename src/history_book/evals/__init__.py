"""
Evaluation framework for the History Book application.

This module provides a standardized way to define, register, and run evaluations
on the RAG system's performance.
"""

from .base import BaseEvaluator
from .evaluators import (
    CoherenceEvaluator,
    FactualAccuracyEvaluator,
    HallucinationEvaluator,
    HelpfulnessEvaluator,
)
from .registry import (
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
]
