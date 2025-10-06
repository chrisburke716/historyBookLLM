# use langsmith 'evaluate' method

import asyncio

from langchain_openai import ChatOpenAI
from langsmith import Client

from history_book.evals import get_prompt_evaluators, get_function_evaluators
from history_book.services import ChatService


async def main():

    eval_run_description = "RAG with min 5 retrieved docs, max 40, 0.4 similarity cutoff, gpt-4o-mini"

    ls_client = Client()

    chat_service = ChatService()

    async def target_wrapper(inputs):
        session = await chat_service.create_session()
        session_id = session.id
        user_message = inputs["question"]
        result = await chat_service.send_message(
            session_id=session_id, user_message=user_message
        )

        # Return both the message content and retrieved context for evaluation
        return {
            "content": result.message.content,
            "retrieved_context": [para.text for para in result.retrieved_paragraphs],
        }

    dataset_name = "History Book Eval Queries"

    # Create LLM for evaluations
    llm = ChatOpenAI(
        model="gpt-5-mini-2025-08-07", temperature=1.0
    )  # gpt-5 models don't support temp != 1.0

    # Get subset of dataset with metadata source = user
    data_subset = ls_client.list_examples(
        dataset_name=dataset_name, metadata={"source": "user"}
    )
    # even shorter for testing
    data_subset = list(data_subset)[:3]

    # Create evaluators using the registry
    prompt_evaluators = get_prompt_evaluators(llm=llm)
    function_evaluators = get_function_evaluators()

    # Create LangSmith evaluators
    prompt_evals = [evaluator.create_langchain_evaluator() for evaluator in prompt_evaluators]
    function_evals = [evaluator.create_langsmith_evaluator() for evaluator in function_evaluators]

    # Combine all evaluators
    all_evals = prompt_evals + function_evals

    all_evaluator_names = [e.name for e in prompt_evaluators] + [e.name for e in function_evaluators]
    print(f"Running evaluations with: {all_evaluator_names}")

    # Extract metadata for evaluation tracking
    metadata = chat_service.get_eval_metadata()

    # Add evaluator metadata
    metadata["evaluator_llm_model"] = llm.model_name
    metadata["evaluator_llm_temperature"] = llm.temperature

    print(f"Evaluation metadata: {metadata}")

    _eval = await ls_client.aevaluate(
        target_wrapper,
        data=dataset_name, # full dataset
        # data=data_subset, # smaller subset for testing
        evaluators=all_evals,
        description=eval_run_description,
        metadata=metadata,
        max_concurrency=10,
    )


if __name__ == "__main__":
    asyncio.run(main())
