"""Pairwise helpfulness comparison."""

from history_book.evals.base import PairwiseEvaluator

PROMPT = """You are comparing two AI responses to determine which is more helpful.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Consider:
- Which better answers the question?
- Which provides more relevant and useful information?
- Which is clearer and easier to understand?

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, write only A, B, or TIE.

Reasoning:"""


class PairwiseHelpfulnessEvaluator(PairwiseEvaluator):
    name = "pairwise_helpfulness"
    prompt = PROMPT

    def format_prompt(self, run_a, run_b, example) -> str:
        return self.prompt.format(
            question=example.inputs.get("question", ""),
            response_a=run_a.outputs.get("content", ""),
            response_b=run_b.outputs.get("content", ""),
        )
