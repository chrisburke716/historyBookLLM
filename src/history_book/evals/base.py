"""
Base classes and interfaces for the evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain.evaluation import EvaluatorType
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith.evaluation import LangChainStringEvaluator


class BaseEvaluator(ABC):
    """
    Base class for all evaluators in the system.

    Subclasses should implement the abstract methods to define:
    1. How to prepare data for LangChain evaluation
    2. The evaluation configuration (criteria, prompts, etc.)
    3. The evaluator type (CRITERIA or LABELED_CRITERIA)
    """

    def __init__(self, llm: ChatOpenAI | None = None):
        """
        Initialize the evaluator.

        Args:
            llm: Language model to use for evaluation. If None, will use default.
        """
        self.llm = llm or ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1.0)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this evaluator."""
        pass

    @property
    @abstractmethod
    def evaluator_type(self) -> EvaluatorType:
        """The type of LangChain evaluator to use."""
        pass

    @abstractmethod
    def get_criteria(self) -> Any:
        """
        Get the evaluation criteria.

        Returns:
            For CRITERIA type: A Criteria enum value
            For LABELED_CRITERIA type: A dict mapping criterion name to description
        """
        pass

    @abstractmethod
    def prepare_data(self, run, example) -> dict[str, Any]:
        """
        Prepare data for evaluation.

        Args:
            run: LangSmith run object containing outputs
            example: LangSmith example object containing inputs

        Returns:
            Dictionary with keys expected by the evaluator (e.g., prediction, input, reference)
        """
        pass

    def get_prompt(self) -> PromptTemplate | None:
        """
        Get custom prompt template for evaluation.

        Returns:
            PromptTemplate if custom prompt is needed, None for default
        """
        return None

    def get_config(self) -> dict[str, Any]:
        """
        Get additional configuration for the evaluator.

        Returns:
            Configuration dictionary to pass to LangChain evaluator
        """
        config = {
            "criteria": self.get_criteria(),
            "llm": self.llm,
        }

        prompt = self.get_prompt()
        if prompt:
            config["prompt"] = prompt

        return config

    def create_langchain_evaluator(self) -> LangChainStringEvaluator:
        """
        Create a LangChain evaluator instance.

        Returns:
            Configured LangChainStringEvaluator ready for use
        """
        return LangChainStringEvaluator(
            evaluator=self.evaluator_type,
            config=self.get_config(),
            prepare_data=self.prepare_data,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()


class CriteriaEvaluator(BaseEvaluator):
    """
    Base class for evaluators that use the CRITERIA type.

    These evaluators assess responses against predefined criteria without
    needing reference material.
    """

    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.CRITERIA

    def prepare_data(self, run, example) -> dict[str, Any]:
        """Default data preparation for criteria evaluators."""
        return {
            "prediction": run.outputs.get("content"),
            "input": example.inputs.get("question"),
        }


class LabeledCriteriaEvaluator(BaseEvaluator):
    """
    Base class for evaluators that use the LABELED_CRITERIA type.

    These evaluators compare responses against reference material or
    use custom criteria with additional context.
    """

    @property
    def evaluator_type(self) -> EvaluatorType:
        return EvaluatorType.LABELED_CRITERIA

    def prepare_data(self, run, example) -> dict[str, Any]:
        """Default data preparation for labeled criteria evaluators."""
        return {
            "prediction": run.outputs.get("content"),
            "input": example.inputs.get("question"),
            "reference": run.outputs.get("retrieved_context"),
        }


class FunctionEvaluator(BaseEvaluator):
    """
    Base class for evaluators that use direct functions instead of LLM prompts.

    These evaluators compute metrics directly from the data without requiring
    an LLM for evaluation.
    """

    def __init__(self, llm: ChatOpenAI | None = None):
        """Initialize function evaluator. LLM parameter is ignored."""
        # Function evaluators don't need LLMs
        pass

    @property
    def evaluator_type(self) -> str:
        """Function evaluators use a custom type."""
        return "function"

    @abstractmethod
    def evaluate(self, run, example) -> dict[str, Any]:
        """
        Evaluate directly without LLM.

        Args:
            run: LangSmith run object containing outputs
            example: LangSmith example object containing inputs

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def get_criteria(self) -> Any:
        """Function evaluators don't use criteria."""
        return None

    def prepare_data(self, run, example) -> dict[str, Any]:
        """Function evaluators don't use prepare_data."""
        return {}

    def get_prompt(self) -> PromptTemplate | None:
        """Function evaluators don't use prompts."""
        return None

    def create_langsmith_evaluator(self):
        """
        Create a custom function-based evaluator for LangSmith.

        Returns:
            Custom evaluator function
        """

        def custom_evaluator(run, example):
            """Custom evaluator function for LangSmith."""
            return self.evaluate(run, example)

        # Return the function with metadata
        custom_evaluator.name = self.name
        custom_evaluator.type = "function"
        return custom_evaluator
