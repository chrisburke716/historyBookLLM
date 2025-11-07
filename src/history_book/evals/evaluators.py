"""
Built-in evaluator implementations.
"""

from typing import Any

from langchain.evaluation import Criteria

from history_book.evals.base import (
    CriteriaEvaluator,
    FunctionEvaluator,
    LabeledCriteriaEvaluator,
)
from history_book.evals.criteria_prompts import (
    COHERENCE_PROMPT,
    FACTUAL_ACCURACY_PROMPT,
    IDK_PROMPT,
    RELEVANCE_PROMPT,
)
from history_book.evals.labeled_criteria_prompts import (
    HALLUCINATION_PROMPT,
    IDK_APPROPRIATE_PROMPT,
)
from history_book.evals.registry import register_evaluator


@register_evaluator
class HelpfulnessEvaluator(CriteriaEvaluator):
    """Evaluates how helpful the response is to the user."""

    @property
    def name(self) -> str:
        return "helpfulness"

    def get_criteria(self):
        return Criteria.HELPFULNESS


@register_evaluator
class FactualAccuracyEvaluator(CriteriaEvaluator):
    """Evaluates factual accuracy based on general knowledge."""

    @property
    def name(self) -> str:
        return "factual_accuracy"

    def get_criteria(self):
        return {
            "factual_accuracy": "Determine if this response contains factually accurate information based on general knowledge. The response should not contain incorrect dates, names, historical facts, or other verifiable information."
        }

    def get_prompt(self):
        return FACTUAL_ACCURACY_PROMPT


@register_evaluator
class CoherenceEvaluator(CriteriaEvaluator):
    """Evaluates logical coherence and structure of the response."""

    @property
    def name(self) -> str:
        return "coherence"

    def get_criteria(self):
        return {
            "coherence": "Determine if this response is logically coherent, well-structured, and flows naturally from one idea to the next without contradictions."
        }

    def get_prompt(self):
        return COHERENCE_PROMPT


@register_evaluator
class HallucinationEvaluator(LabeledCriteriaEvaluator):
    """Evaluates whether the response contains hallucinated information not supported by retrieved context."""

    @property
    def name(self) -> str:
        return "hallucination"

    def get_criteria(self):
        return {
            "hallucination": "Determine if this AI response contains hallucinated information - factual claims, specific details, dates, names, or events that are not supported by or present in the retrieved context. The response should only make claims that can be verified against the provided reference material."
        }

    def get_prompt(self):
        return HALLUCINATION_PROMPT


@register_evaluator
class IdkEvaluator(CriteriaEvaluator):
    """Evaluates whether the response appropriately expresses uncertainty or lack of knowledge."""

    @property
    def name(self) -> str:
        return "idk"

    def get_criteria(self):
        return {
            "idk": "Determine if this AI response appropriately expresses uncertainty or admits lack of knowledge when it doesn't know the answer or couldn't find the information."
        }

    def get_prompt(self):
        return IDK_PROMPT


@register_evaluator
class RelevanceEvaluator(CriteriaEvaluator):
    """Evaluates whether the retrieved context is relevant to the user's question."""

    @property
    def name(self) -> str:
        return "relevance"

    def get_criteria(self):
        return {
            "relevance": "Determine if the retrieved context contains information relevant to answering the user's question. The context should be on-topic and related to what was asked, even if it doesn't contain the complete answer."
        }

    def get_prompt(self):
        return RELEVANCE_PROMPT

    def prepare_data(self, run, example) -> dict[str, Any]:
        """Custom data preparation for context relevance evaluation."""
        return {
            "input": example.inputs.get("question"),
            "prediction": run.outputs.get(
                "retrieved_context"
            ),  # Context as thing being evaluated
        }


@register_evaluator
class IdkAppropriateEvaluator(LabeledCriteriaEvaluator):
    """Evaluates whether 'I don't know' responses are appropriate given the retrieved context."""

    @property
    def name(self) -> str:
        return "idk_appropriate"

    def get_criteria(self):
        return {
            "idk_appropriate": "Determine if the AI's level of confidence (knowing vs not knowing) is appropriate given the retrieved context. The AI should say 'I don't know' when context lacks information, and provide answers when context contains relevant information."
        }

    def get_prompt(self):
        return IDK_APPROPRIATE_PROMPT


@register_evaluator
class DocumentCountEvaluator(FunctionEvaluator):
    """Evaluates the number of documents retrieved for each query."""

    @property
    def name(self) -> str:
        return "document_count"

    def evaluate(self, run, example) -> dict[str, Any]:
        """
        Count the number of retrieved documents.

        Args:
            run: LangSmith run object containing outputs
            example: LangSmith example object containing inputs

        Returns:
            Dictionary with document count and metadata
        """
        retrieved_context = run.outputs.get("retrieved_context", [])

        # Handle both list of strings and list of objects
        if isinstance(retrieved_context, list):
            count = len(retrieved_context)
        else:
            count = 0

        return {
            "key": "document_count",
            "score": count,
            "value": count,
            "comment": f"Retrieved {count} documents for query",
        }
