"""Hallucination evaluator (against retrieved context)."""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating an AI response for hallucinations against retrieved context.

Retrieved Context:
{context}

User Question: {question}
AI Response: {response}

Check whether the response contains factual claims not supported by the retrieved context. Look for:
- Made-up facts, dates, names, or events not mentioned in the context
- Information that contradicts the retrieved context
- Specific details or claims not present in the reference material

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.
- Y if the response contains hallucinated information
- N if the response is factually consistent with the context

Reasoning:"""


class HallucinationEvaluator(LLMEvaluator):
    name = "hallucination"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        context = outputs.get("retrieved_context") or []
        context_str = "\n\n".join(context) if context else "(No context retrieved)"
        return self.prompt.format(
            question=inputs["question"],
            response=outputs["content"],
            context=context_str,
        )
