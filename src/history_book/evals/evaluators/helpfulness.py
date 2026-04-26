"""Helpfulness evaluator."""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating how helpful an AI response is to the user's question.

User Question: {question}
AI Response: {response}

Assess whether the response is helpful. Consider:
- Does it directly address the question?
- Does it provide useful, relevant information?
- Is it clear and easy to understand?

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.
- Y if the response is helpful
- N if the response is not helpful

Reasoning:"""


class HelpfulnessEvaluator(LLMEvaluator):
    name = "helpfulness"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        return self.prompt.format(
            question=inputs["question"],
            response=outputs["content"],
        )
