"""Pairwise completeness comparison."""

from history_book.evals.base import PairwiseEvaluator

PROMPT = """You are comparing two AI responses to determine which more completely answers the question.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Consider:
- Which addresses more aspects of the question?
- Which provides more thorough coverage of the topic?
- If the question has multiple parts, which addresses more parts?

Note: More complete doesn't necessarily mean longer. Focus on coverage of the question's scope.

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, write only A, B, or TIE.

Reasoning:"""


class PairwiseCompletenessEvaluator(PairwiseEvaluator):
    name = "pairwise_completeness"
    prompt = PROMPT

    def format_prompt(self, run_a, run_b, example) -> str:
        return self.prompt.format(
            question=example.inputs.get("question", ""),
            response_a=run_a.outputs.get("content", ""),
            response_b=run_b.outputs.get("content", ""),
        )
