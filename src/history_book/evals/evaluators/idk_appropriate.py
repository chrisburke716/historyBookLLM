"""IDK-appropriate evaluator — does confidence match available context?"""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating whether an AI response's confidence is appropriate given the retrieved context.

Retrieved Context:
{context}

User Question: {question}
AI Response: {response}

Determine if the AI's uncertainty or confidence is appropriate given the available context:

APPROPRIATE (respond Y):
- AI says "I don't know" AND the context lacks relevant information
- AI provides an answer AND the context contains supporting information

INAPPROPRIATE (respond N):
- AI says "I don't know" BUT the context contains relevant information
- AI provides a confident answer BUT the context lacks sufficient support

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.

Reasoning:"""


class IdkAppropriateEvaluator(LLMEvaluator):
    name = "idk_appropriate"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        context = outputs.get("retrieved_context") or []
        context_str = "\n\n".join(context) if context else "(No context retrieved)"
        return self.prompt.format(
            question=inputs["question"],
            response=outputs["content"],
            context=context_str,
        )
