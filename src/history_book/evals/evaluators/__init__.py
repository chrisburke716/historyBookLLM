"""Concrete evaluator implementations."""

from langchain_core.language_models import BaseChatModel

from history_book.evals.base import Evaluator, PairwiseEvaluator
from history_book.evals.evaluators.coherence import CoherenceEvaluator
from history_book.evals.evaluators.document_count import DocumentCountEvaluator
from history_book.evals.evaluators.factual_accuracy import FactualAccuracyEvaluator
from history_book.evals.evaluators.hallucination import HallucinationEvaluator
from history_book.evals.evaluators.helpfulness import HelpfulnessEvaluator
from history_book.evals.evaluators.idk import IdkEvaluator
from history_book.evals.evaluators.idk_appropriate import IdkAppropriateEvaluator
from history_book.evals.evaluators.pairwise_coherence import (
    PairwiseCoherenceEvaluator,
)
from history_book.evals.evaluators.pairwise_completeness import (
    PairwiseCompletenessEvaluator,
)
from history_book.evals.evaluators.pairwise_hallucination import (
    PairwiseHallucinationEvaluator,
)
from history_book.evals.evaluators.pairwise_helpfulness import (
    PairwiseHelpfulnessEvaluator,
)
from history_book.evals.evaluators.relevance import RelevanceEvaluator

LLM_EVALUATOR_CLASSES = [
    HelpfulnessEvaluator,
    FactualAccuracyEvaluator,
    CoherenceEvaluator,
    HallucinationEvaluator,
    IdkEvaluator,
    IdkAppropriateEvaluator,
    RelevanceEvaluator,
]

FUNCTION_EVALUATOR_CLASSES = [
    DocumentCountEvaluator,
]

PAIRWISE_EVALUATOR_CLASSES = [
    PairwiseHelpfulnessEvaluator,
    PairwiseHallucinationEvaluator,
    PairwiseCoherenceEvaluator,
    PairwiseCompletenessEvaluator,
]


def build_llm_evaluators(llm: BaseChatModel) -> list[Evaluator]:
    return [cls(llm) for cls in LLM_EVALUATOR_CLASSES]


def build_function_evaluators() -> list[Evaluator]:
    return [cls() for cls in FUNCTION_EVALUATOR_CLASSES]


def build_pairwise_evaluators(llm: BaseChatModel) -> list[PairwiseEvaluator]:
    return [cls(llm) for cls in PAIRWISE_EVALUATOR_CLASSES]


__all__ = [
    "CoherenceEvaluator",
    "DocumentCountEvaluator",
    "FactualAccuracyEvaluator",
    "HallucinationEvaluator",
    "HelpfulnessEvaluator",
    "IdkAppropriateEvaluator",
    "IdkEvaluator",
    "PairwiseCoherenceEvaluator",
    "PairwiseCompletenessEvaluator",
    "PairwiseHallucinationEvaluator",
    "PairwiseHelpfulnessEvaluator",
    "RelevanceEvaluator",
    "build_function_evaluators",
    "build_llm_evaluators",
    "build_pairwise_evaluators",
]
