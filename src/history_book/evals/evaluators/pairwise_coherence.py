"""Pairwise coherence comparison."""

from history_book.evals.base import PairwiseEvaluator

PROMPT = """You are comparing two AI responses to determine which is more coherent.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Consider:
- Which has clearer logical flow from one idea to the next?
- Which is better organized and structured?
- Which avoids contradictions within the response?

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, write only A, B, or TIE.

Reasoning:"""


class PairwiseCoherenceEvaluator(PairwiseEvaluator):
    name = "pairwise_coherence"
    prompt = PROMPT

    def format_prompt(self, run_a, run_b, example) -> str:
        return self.prompt.format(
            question=example.inputs.get("question", ""),
            response_a=run_a.outputs.get("content", ""),
            response_b=run_b.outputs.get("content", ""),
        )
