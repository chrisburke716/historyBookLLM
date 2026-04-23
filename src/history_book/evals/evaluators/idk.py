"""IDK evaluator — does the response express uncertainty?"""

from typing import Any

from history_book.evals.base import LLMEvaluator

PROMPT = """You are evaluating whether an AI response expresses uncertainty or lack of knowledge.

User Question: {question}
AI Response: {response}

Determine if the response indicates the AI doesn't know the answer or couldn't find the info. Look for:
- Explicit statements like "I don't know", "I'm not sure", "I couldn't find"
- Hedging language indicating uncertainty
- Admissions of insufficient information or inability to answer

First, provide your step-by-step reasoning. Then, on the final line, write only Y or N.
- Y if the response expresses uncertainty or lack of knowledge
- N if the response provides a confident answer without expressing uncertainty

Reasoning:"""


class IdkEvaluator(LLMEvaluator):
    name = "idk"
    prompt = PROMPT

    def format_prompt(self, outputs: dict[str, Any], inputs: dict[str, Any]) -> str:
        return self.prompt.format(
            question=inputs["question"],
            response=outputs["content"],
        )
