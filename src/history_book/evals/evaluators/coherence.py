"""Coherence evaluator."""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating the coherence and logical flow of an AI response.

User Question: {question}
AI Response: {response}

Assess whether the response is logically structured and flows naturally. Look for:
- Clear logical progression of ideas
- Consistent line of reasoning
- Well-organized structure
- Absence of contradictions within the response

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.
- Y if the response is coherent and well-structured
- N if the response lacks coherence or logical flow

Reasoning:"""


class CoherenceEvaluator(LLMEvaluator):
    name = "coherence"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        return self.prompt.format(
            question=inputs["question"],
            response=outputs["content"],
        )
