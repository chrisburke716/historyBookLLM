# use langsmith 'evaluate' method

import argparse
import asyncio

from langchain_openai import ChatOpenAI
from langsmith import Client

from history_book.evals import get_function_evaluators, get_prompt_evaluators
from history_book.services import ChatService, GraphChatService


async def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run evaluations on RAG system")
    parser.add_argument(
        "--mode",
        choices=["agent", "legacy"],
        default="agent",
        help="Which system to evaluate: 'agent' (LangGraph) or 'legacy' (LCEL)",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run on 3-query subset for quick testing",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run on full 100-query dataset"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.subset and args.full:
        parser.error("Cannot specify both --subset and --full")

    # Determine dataset mode
    if args.subset:
        dataset_mode = "subset (3 queries)"
    elif args.full:
        dataset_mode = "full (100 queries)"
    else:
        # Default to subset for safety
        args.subset = True
        dataset_mode = "subset (3 queries, default)"

    # Set up description
    system_type = "LangGraph agent" if args.mode == "agent" else "Legacy RAG (LCEL)"
    eval_run_description = f"{system_type} with book search prompt - {dataset_mode}"

    print(f"Running evaluation: {eval_run_description}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {dataset_mode}")

    ls_client = Client()

    # Choose service based on mode
    if args.mode == "agent":
        chat_service = GraphChatService()
    else:
        chat_service = ChatService()

    async def target_wrapper(inputs):
        session = await chat_service.create_session()
        session_id = session.id
        user_message = inputs["question"]
        result = await chat_service.send_message(
            session_id=session_id, user_message=user_message
        )

        # Return both the message content and retrieved context for evaluation
        # Include metadata that the AI had access to during generation
        return {
            "content": result.message.content,
            "retrieved_context": [
                f"[Book {para.book_index}, Chapter {para.chapter_index}, Page {para.page}]\n{para.text}"
                for para in result.retrieved_paragraphs
            ],
        }

    dataset_name = "History Book Eval Queries"

    # Create LLM for evaluations
    llm = ChatOpenAI(
        model="gpt-5-mini-2025-08-07", temperature=1.0
    )  # gpt-5 models don't support temp != 1.0

    # Determine dataset to use
    if args.subset:
        # Get subset of dataset with metadata source = user (3 queries for quick testing)
        data_subset = ls_client.list_examples(
            dataset_name=dataset_name, metadata={"source": "user"}
        )
        data = list(data_subset)[:3]
        print(f"Using {len(data)} queries from subset")
    else:
        # Use full dataset
        data = dataset_name
        print("Using full dataset")

    # Create evaluators using the registry
    prompt_evaluators = get_prompt_evaluators(llm=llm)
    function_evaluators = get_function_evaluators()

    # Create LangSmith evaluators
    prompt_evals = [
        evaluator.create_langchain_evaluator() for evaluator in prompt_evaluators
    ]
    function_evals = [
        evaluator.create_langsmith_evaluator() for evaluator in function_evaluators
    ]

    # Combine all evaluators
    all_evals = prompt_evals + function_evals

    all_evaluator_names = [e.name for e in prompt_evaluators] + [
        e.name for e in function_evaluators
    ]
    print(f"Running evaluations with: {all_evaluator_names}")

    # Extract metadata for evaluation tracking
    metadata = chat_service.get_eval_metadata()

    # Add evaluator metadata
    metadata["evaluator_llm_model"] = llm.model_name
    metadata["evaluator_llm_temperature"] = llm.temperature
    metadata["eval_mode"] = args.mode
    metadata["eval_dataset_mode"] = "subset" if args.subset else "full"

    print(f"Evaluation metadata: {metadata}")

    _eval = await ls_client.aevaluate(
        target_wrapper,
        data=data,
        evaluators=all_evals,
        description=eval_run_description,
        metadata=metadata,
        max_concurrency=5,
    )

    print("\n✅ Evaluation complete!")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {dataset_mode}")
    print("View results in LangSmith")


if __name__ == "__main__":
    asyncio.run(main())
