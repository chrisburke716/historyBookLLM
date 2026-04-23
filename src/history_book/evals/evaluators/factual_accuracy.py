"""Factual accuracy evaluator (against general knowledge)."""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating the factual accuracy of an AI response based on general knowledge.

User Question: {question}
AI Response: {response}

Assess whether the response contains factually correct information. Look for:
- Incorrect dates, names, or historical facts
- Contradictory or impossible claims
- Misinformation or commonly confused facts

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.
- Y if the response is factually accurate
- N if the response contains factual inaccuracies

Reasoning:"""


class FactualAccuracyEvaluator(LLMEvaluator):
    name = "factual_accuracy"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        return self.prompt.format(
            question=inputs["question"],
            response=outputs["content"],
        )
