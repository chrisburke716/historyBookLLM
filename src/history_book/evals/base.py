"""Base classes for the evaluation framework.

Each evaluator owns a single prompt and a single LLM call. Wrappers expose the
right shape for LangSmith's `evaluate` (single-run) or `evaluate_comparative`
(pairwise).
"""

import re
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel


class EvalResult(TypedDict, total=False):
    """Result returned by a single-run evaluator."""

    key: str
    score: float
    value: str | None  # raw verdict, e.g. "Y" / "N"
    comment: str | None


class PairwiseResult(TypedDict, total=False):
    """Result returned by a pairwise evaluator."""

    key: str
    scores: dict[str, float]  # maps run_id → score
    comment: str | None


def _last_alpha_token(line: str) -> str | None:
    """Return the last alphabetic token in a line, uppercased, or None if no tokens."""
    tokens = re.findall(r"[A-Za-z]+", line)
    return tokens[-1].upper() if tokens else None


def parse_yn(text: str) -> tuple[float, str | None]:
    """Parse a Y/N verdict from the last non-empty line of an LLM response.

    Looks at the last alphabetic token on the last non-empty line — tolerates
    prefixes like "Final judgment: Y" or markup like "**Y**".

    Returns (score, verdict): (1.0, "Y"), (0.0, "N"), or (0.5, None) if unparseable.
    """
    for line in reversed(text.strip().splitlines()):
        if not line.strip():
            continue
        verdict = _last_alpha_token(line)
        if verdict == "Y":
            return 1.0, "Y"
        if verdict == "N":
            return 0.0, "N"
        return 0.5, None
    return 0.5, None


def parse_abtie(text: str) -> str:
    """Parse an A/B/TIE verdict from the last non-empty line of an LLM response.

    Tolerates prefixes like "Winner: A" or "Final judgment: TIE".
    """
    for line in reversed(text.strip().splitlines()):
        if not line.strip():
            continue
        verdict = _last_alpha_token(line)
        if verdict in ("A", "B", "TIE"):
            return verdict
        return "TIE"
    return "TIE"


class Evaluator(ABC):
    """Base class for single-run evaluators."""

    name: str

    @abstractmethod
    async def evaluate(
        self, outputs: dict[str, Any], inputs: dict[str, Any]
    ) -> EvalResult: ...

    def as_langsmith(self):
        """Return a function with the signature LangSmith's `aevaluate` expects."""

        async def _wrapped(outputs: dict, inputs: dict) -> EvalResult:
            return await self.evaluate(outputs, inputs)

        _wrapped.__name__ = self.name
        return _wrapped


class LLMEvaluator(Evaluator):
    """Base class for evaluators that call an LLM with a Y/N verdict prompt."""

    prompt: str

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @abstractmethod
    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str: ...

    async def evaluate(
        self, outputs: dict[str, Any], inputs: dict[str, Any]
    ) -> EvalResult:
        response = await self.llm.ainvoke(self.format_prompt(outputs, inputs))
        text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        score, verdict = parse_yn(text)
        return {
            "key": self.name,
            "score": score,
            "value": verdict,
            "comment": text[:500],
        }


class FunctionEvaluator(Evaluator):
    """Base class for evaluators that compute a score directly, no LLM."""


class PairwiseEvaluator(ABC):
    """Base class for pairwise comparison evaluators.

    Used with `langsmith.evaluation.evaluate_comparative` — signature takes two
    runs and an example, returns {key, scores: {run_id: float}}.
    """

    name: str
    prompt: str

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @abstractmethod
    def format_prompt(self, run_a, run_b, example) -> str: ...

    def _scores(self, winner: str, run_a_id: str, run_b_id: str) -> dict[str, float]:
        if winner == "A":
            return {run_a_id: 1.0, run_b_id: 0.0}
        if winner == "B":
            return {run_a_id: 0.0, run_b_id: 1.0}
        return {run_a_id: 0.5, run_b_id: 0.5}

    def as_langsmith(self):
        """Return a function with the signature `evaluate_comparative` expects."""

        def _wrapped(runs: list, example) -> PairwiseResult:
            run_a, run_b = runs
            prompt_text = self.format_prompt(run_a, run_b, example)
            response = self.llm.invoke(prompt_text)
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            winner = parse_abtie(text)
            return {
                "key": self.name,
                "scores": self._scores(winner, run_a.id, run_b.id),
                "comment": f"Winner: {winner} | {text[:200]}",
            }

        _wrapped.evaluator_name = self.name
        _wrapped.__name__ = self.name
        return _wrapped
