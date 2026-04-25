"""Relevance evaluator — is the retrieved context relevant to the question?

Unlike the other evaluators, this judges the retrieved context itself rather
than the AI response.
"""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating whether the retrieved context is relevant to the user's question.

User Question: {question}
Retrieved Context:
{context}

Determine if the context contains information relevant to answering the question:
- Does the context contain information that could help answer the question?
- Is the context on-topic and related to what was asked?
- If the question has multiple parts, does the context address any of them?

The context doesn't need the complete answer — just relevant information.

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.
- Y if the context is relevant
- N if the context is off-topic or unrelated

Reasoning:"""


class RelevanceEvaluator(LLMEvaluator):
    name = "relevance"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        context = outputs.get("retrieved_context") or []
        context_str = "\n\n".join(context) if context else "(No context retrieved)"
        return self.prompt.format(
            question=inputs["question"],
            context=context_str,
        )
