"""Run RAG evaluations via LangSmith."""

import argparse
import asyncio

from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate_comparative

from history_book.evals import (
    build_function_evaluators,
    build_llm_evaluators,
    build_pairwise_evaluators,
)
from history_book.services import ChatService

EVAL_LLM_MODEL = "gpt-5-mini-2025-08-07"
EVAL_LLM_TEMPERATURE = 1.0  # gpt-5 mini doesn't support other values


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluations on RAG system")
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run on 3-query subset for quick testing",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run on full 100-query dataset"
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Run pairwise comparison between two experiments (requires --experiments)",
    )
    parser.add_argument(
        "--experiments",
        nargs=2,
        metavar=("EXP1", "EXP2"),
        help="Two experiment IDs or names to compare (required with --pairwise)",
    )
    args = parser.parse_args()

    if args.subset and args.full:
        parser.error("Cannot specify both --subset and --full")
    if args.pairwise and not args.experiments:
        parser.error("--pairwise requires --experiments with 2 experiment IDs/names")
    if args.experiments and not args.pairwise:
        parser.error("--experiments can only be used with --pairwise")

    ls_client = Client()
    llm = ChatOpenAI(model=EVAL_LLM_MODEL, temperature=EVAL_LLM_TEMPERATURE)

    if args.pairwise:
        await _run_pairwise(ls_client, llm, args.experiments)
        return

    await _run_single(ls_client, llm, subset=args.subset or not args.full)


async def _run_single(ls_client: Client, llm: ChatOpenAI, subset: bool) -> None:
    dataset_mode = "subset (3 queries)" if subset else "full (100 queries)"
    description = f"RAG agent with book search prompt - {dataset_mode}"

    print(f"Running evaluation: {description}")

    chat_service = ChatService()

    async def target_wrapper(inputs: dict) -> dict:
        session = await chat_service.create_session()
        result = await chat_service.send_message(
            session_id=session.id, user_message=inputs["question"]
        )
        return {
            "content": result.message.content,
            "retrieved_context": [
                f"[Book {p.book_index}, Chapter {p.chapter_index}, Page {p.page}]\n{p.text}"
                for p in result.retrieved_paragraphs
            ],
        }

    dataset_name = "History Book Eval Queries"
    if subset:
        examples = ls_client.list_examples(
            dataset_name=dataset_name, metadata={"source": "user"}
        )
        data = list(examples)[:3]
        print(f"Using {len(data)} queries from subset")
    else:
        data = dataset_name
        print("Using full dataset")

    llm_evals = build_llm_evaluators(llm)
    fn_evals = build_function_evaluators()
    all_evals = [e.as_langsmith() for e in llm_evals + fn_evals]

    print(f"Running evaluations with: {[e.name for e in llm_evals + fn_evals]}")

    metadata = chat_service.get_eval_metadata() | {
        "evaluator_llm_model": llm.model_name,
        "evaluator_llm_temperature": llm.temperature,
        "eval_dataset_mode": "subset" if subset else "full",
    }
    print(f"Evaluation metadata: {metadata}")

    await ls_client.aevaluate(
        target_wrapper,
        data=data,
        evaluators=all_evals,
        description=description,
        metadata=metadata,
        max_concurrency=5,
    )

    print("\n✅ Evaluation complete!")
    print(f"Dataset: {dataset_mode}")
    print("View results in LangSmith")


async def _run_pairwise(
    ls_client: Client, llm: ChatOpenAI, experiments: list[str]
) -> None:
    exp1, exp2 = experiments
    print(f"Running pairwise evaluation: {exp1} vs {exp2}")

    pairwise_evals = build_pairwise_evaluators(llm)
    wrapped = [e.as_langsmith() for e in pairwise_evals]

    print(f"Pairwise evaluators: {[e.name for e in pairwise_evals]}")

    evaluate_comparative(
        (exp1, exp2),
        evaluators=wrapped,
        description=f"Pairwise comparison: {exp1} vs {exp2}",
        metadata={
            "comparison_type": "pairwise",
            "experiment_1": exp1,
            "experiment_2": exp2,
            "evaluator_count": len(pairwise_evals),
        },
        max_concurrency=5,
        client=ls_client,
    )

    print(f"\n✅ Pairwise evaluation complete: {exp1} vs {exp2}")
    print("View results in LangSmith")


if __name__ == "__main__":
    asyncio.run(main())
