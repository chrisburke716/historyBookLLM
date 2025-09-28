# use langsmith 'evaluate' method

import asyncio

from langchain_openai import ChatOpenAI
from langsmith import Client

from history_book.evals import get_all_evaluators
from history_book.services import ChatService


async def main():
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

    # Create all evaluators using the registry
    evaluators = get_all_evaluators(llm=llm)
    evals = [evaluator.create_langchain_evaluator() for evaluator in evaluators]

    print(f"Running evaluations with: {[eval.name for eval in evaluators]}")

    _eval = await ls_client.aevaluate(
        target_wrapper,
        data=data_subset,
        evaluators=evals,
        description="setup testing",
        max_concurrency=4,
    )


if __name__ == "__main__":
    asyncio.run(main())
