"""Document count evaluator — non-LLM, counts retrieved paragraphs."""

from typing import Any

from history_book.evals.base import EvalResult, FunctionEvaluator


class DocumentCountEvaluator(FunctionEvaluator):
    name = "document_count"

    async def evaluate(
        self, outputs: dict[str, Any], inputs: dict[str, Any]
    ) -> EvalResult:
        context = outputs.get("retrieved_context") or []
        count = len(context) if isinstance(context, list) else 0
        return {
            "key": self.name,
            "score": float(count),
            "comment": f"Retrieved {count} documents",
        }
