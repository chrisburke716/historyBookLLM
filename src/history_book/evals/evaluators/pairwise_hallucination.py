"""Pairwise hallucination comparison — each response judged against its own context."""

from history_book.evals.base import PairwiseEvaluator

PROMPT = """You are comparing two AI responses to determine which has fewer hallucinations.

User Question: {question}

Response A:
{response_a}

Context Available to Response A:
{context_a}

Response B:
{response_b}

Context Available to Response B:
{context_b}

For each response, check whether it makes claims not supported by ITS OWN context.
A response is NOT hallucinating just because it didn't retrieve the same documents.

First, provide your step-by-step reasoning.
Then, on the final line, state which response has FEWER hallucinations — write only A, B, or TIE.

Reasoning:"""


def _context_str(run) -> str:
    context = run.outputs.get("retrieved_context") or []
    return "\n\n".join(context) if context else "(No context retrieved)"


class PairwiseHallucinationEvaluator(PairwiseEvaluator):
    name = "pairwise_hallucination"
    prompt = PROMPT

    def format_prompt(self, run_a, run_b, example) -> str:
        return self.prompt.format(
            question=example.inputs.get("question", ""),
            response_a=run_a.outputs.get("content", ""),
            context_a=_context_str(run_a),
            response_b=run_b.outputs.get("content", ""),
            context_b=_context_str(run_b),
        )
