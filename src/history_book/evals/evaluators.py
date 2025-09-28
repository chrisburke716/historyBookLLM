"""
Built-in evaluator implementations.
"""

from langchain.evaluation import Criteria

from .base import CriteriaEvaluator, LabeledCriteriaEvaluator
from .criteria_prompts import COHERENCE_PROMPT, FACTUAL_ACCURACY_PROMPT
from .labeled_criteria_prompts import HALLUCINATION_PROMPT
from .registry import register_evaluator


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
