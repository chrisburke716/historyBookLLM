"""
Pairwise evaluator implementations for comparing two experiment runs.

These evaluators compare outputs from two different experiments to determine
which is better according to specific criteria. They use an LLM judge to make
comparative assessments.
"""

from langchain_openai import ChatOpenAI

from history_book.evals.pairwise_prompts import (
    PAIRWISE_COHERENCE_PROMPT,
    PAIRWISE_COMPLETENESS_PROMPT,
    PAIRWISE_HALLUCINATION_PROMPT,
    PAIRWISE_HELPFULNESS_PROMPT,
)

# List of evaluator names for discovery
PAIRWISE_EVALUATOR_NAMES = [
    "pairwise_helpfulness",
    "pairwise_hallucination",
    "pairwise_coherence",
    "pairwise_completeness",
]


def _extract_winner(response_text: str) -> str:
    """
    Extract winner from LLM response.

    Args:
        response_text: LLM's reasoning + final judgment

    Returns:
        "A", "B", or "TIE"
    """
    # Check last few lines for the judgment
    lines = response_text.strip().split("\n")
    for line in reversed(lines[-3:]):  # Check last 3 lines
        cleaned = line.strip().upper()
        if cleaned in ("A", "B", "TIE"):
            return cleaned
    # Default to TIE if unclear
    return "TIE"


def _calculate_scores(winner: str, run_a_id: str, run_b_id: str) -> dict:
    """
    Convert winner to score dict.

    Args:
        winner: "A", "B", or "TIE"
        run_a_id: ID of run A
        run_b_id: ID of run B

    Returns:
        Dict mapping run IDs to scores (1.0 = winner, 0.0 = loser, 0.5 = tie)
    """
    if winner == "A":
        return {run_a_id: 1.0, run_b_id: 0.0}
    elif winner == "B":
        return {run_a_id: 0.0, run_b_id: 1.0}
    else:  # TIE
        return {run_a_id: 0.5, run_b_id: 0.5}


def get_all_pairwise_evaluators(llm: ChatOpenAI | None = None):
    """
    Factory function that creates all pairwise evaluators with LLM bound via closure.

    Args:
        llm: Language model to use for evaluation. If None, uses default.

    Returns:
        List of pairwise evaluator functions with LLM bound
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1.0)

    def pairwise_helpfulness(runs: list, example) -> dict:
        """
        Compare two responses for helpfulness.

        Args:
            runs: List of 2 Run objects from different experiments
            example: The evaluation example

        Returns:
            Evaluation result with scores for each run
        """
        run_a, run_b = runs

        # Extract data
        question = example.inputs.get("question", "")
        response_a = run_a.outputs.get("content", "")
        response_b = run_b.outputs.get("content", "")

        # Format prompt
        prompt_text = PAIRWISE_HELPFULNESS_PROMPT.format(
            question=question, response_a=response_a, response_b=response_b
        )

        # Get LLM judgment (uses llm from closure)
        response = llm.invoke(prompt_text)
        winner = _extract_winner(response.content)

        # Calculate scores
        scores = _calculate_scores(winner, run_a.id, run_b.id)

        return {
            "key": "pairwise_helpfulness",
            "scores": scores,
            "comment": f"Winner: {winner} | {response.content[:200]}...",
        }

    # Store name as attribute for LangSmith
    pairwise_helpfulness.evaluator_name = "pairwise_helpfulness"

    def pairwise_hallucination(runs: list, example) -> dict:
        """
        Compare two responses for hallucinations (fewer is better).

        Each response is judged against its own retrieved context.

        Args:
            runs: List of 2 Run objects from different experiments
            example: The evaluation example

        Returns:
            Evaluation result with scores for each run
        """
        run_a, run_b = runs

        # Extract data
        question = example.inputs.get("question", "")
        response_a = run_a.outputs.get("content", "")
        response_b = run_b.outputs.get("content", "")

        # Get contexts (each run has its own retrieved context)
        context_a_list = run_a.outputs.get("retrieved_context", [])
        context_b_list = run_b.outputs.get("retrieved_context", [])

        # Format contexts as strings
        context_a = (
            "\n\n".join(context_a_list) if context_a_list else "(No context retrieved)"
        )
        context_b = (
            "\n\n".join(context_b_list) if context_b_list else "(No context retrieved)"
        )

        # Format prompt
        prompt_text = PAIRWISE_HALLUCINATION_PROMPT.format(
            question=question,
            response_a=response_a,
            context_a=context_a,
            response_b=response_b,
            context_b=context_b,
        )

        # Get LLM judgment (uses llm from closure)
        response = llm.invoke(prompt_text)
        winner = _extract_winner(response.content)

        # Calculate scores
        scores = _calculate_scores(winner, run_a.id, run_b.id)

        return {
            "key": "pairwise_hallucination",
            "scores": scores,
            "comment": f"Fewer hallucinations: {winner} | {response.content[:200]}...",
        }

    pairwise_hallucination.evaluator_name = "pairwise_hallucination"

    def pairwise_coherence(runs: list, example) -> dict:
        """
        Compare two responses for coherence and logical structure.

        Args:
            runs: List of 2 Run objects from different experiments
            example: The evaluation example

        Returns:
            Evaluation result with scores for each run
        """
        run_a, run_b = runs

        # Extract data
        question = example.inputs.get("question", "")
        response_a = run_a.outputs.get("content", "")
        response_b = run_b.outputs.get("content", "")

        # Format prompt
        prompt_text = PAIRWISE_COHERENCE_PROMPT.format(
            question=question, response_a=response_a, response_b=response_b
        )

        # Get LLM judgment (uses llm from closure)
        response = llm.invoke(prompt_text)
        winner = _extract_winner(response.content)

        # Calculate scores
        scores = _calculate_scores(winner, run_a.id, run_b.id)

        return {
            "key": "pairwise_coherence",
            "scores": scores,
            "comment": f"More coherent: {winner} | {response.content[:200]}...",
        }

    pairwise_coherence.evaluator_name = "pairwise_coherence"

    def pairwise_completeness(runs: list, example) -> dict:
        """
        Compare two responses for completeness in answering the question.

        Args:
            runs: List of 2 Run objects from different experiments
            example: The evaluation example

        Returns:
            Evaluation result with scores for each run
        """
        run_a, run_b = runs

        # Extract data
        question = example.inputs.get("question", "")
        response_a = run_a.outputs.get("content", "")
        response_b = run_b.outputs.get("content", "")

        # Format prompt
        prompt_text = PAIRWISE_COMPLETENESS_PROMPT.format(
            question=question, response_a=response_a, response_b=response_b
        )

        # Get LLM judgment (uses llm from closure)
        response = llm.invoke(prompt_text)
        winner = _extract_winner(response.content)

        # Calculate scores
        scores = _calculate_scores(winner, run_a.id, run_b.id)

        return {
            "key": "pairwise_completeness",
            "scores": scores,
            "comment": f"More complete: {winner} | {response.content[:200]}...",
        }

    pairwise_completeness.evaluator_name = "pairwise_completeness"

    # Return all evaluators as a list
    return [
        pairwise_helpfulness,
        pairwise_hallucination,
        pairwise_coherence,
        pairwise_completeness,
    ]


# Convenience function to get a single evaluator by name
def get_pairwise_evaluator(name: str, llm: ChatOpenAI | None = None):
    """
    Get a specific pairwise evaluator by name.

    Args:
        name: Name of the evaluator (e.g., "pairwise_helpfulness")
        llm: Language model to use. If None, uses default.

    Returns:
        The evaluator function, or None if not found
    """
    evaluators = get_all_pairwise_evaluators(llm)
    for evaluator in evaluators:
        if evaluator.evaluator_name == name:
            return evaluator
    return None


# For backward compatibility, expose individual evaluator names
pairwise_helpfulness = None
pairwise_hallucination = None
pairwise_coherence = None
pairwise_completeness = None
