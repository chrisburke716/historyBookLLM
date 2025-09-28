# use langsmith 'evaluate' method

import asyncio

from langchain.evaluation import Criteria, EvaluatorType
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator

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

    def prepare_data_criteria(run, example):
        return {
            "prediction": run.outputs.get("content"),
            "input": example.inputs.get("question"),
        }

    def prepare_data_labeled_criteria(run, example):
        return {
            "prediction": run.outputs.get("content"),
            "input": example.inputs.get("question"),
            "reference": run.outputs.get("retrieved_context"),
        }

    dataset_name = "History Book Eval Queries"

    # Custom hallucination detection prompt
    _hallucination_prompt = PromptTemplate.from_template("""
You are evaluating an AI response for hallucinations against retrieved context.

Retrieved Context: {reference}
User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Check if the AI response contains any factual claims that are not supported by the retrieved context. Look for:
- Made-up facts, dates, names, or events not mentioned in the context
- Information that contradicts the retrieved context
- Specific details or claims not present in the reference material

Provide your reasoning step by step, then respond with:
- Y if the response contains hallucinated information
- N if the response is factually consistent with the context

Reasoning:""")

    # Evaluation configurations
    llm = ChatOpenAI(
        model="gpt-5-mini-2025-08-07", temperature=1.0
    )  # gpt-5 models don't support temp != 1.0

    eval_configs = [
        {
            "name": "helpfulness",
            "type": EvaluatorType.CRITERIA,
            "criteria": Criteria.HELPFULNESS,
            "prompt": None,
        },
        {
            "name": "hallucination",
            "type": EvaluatorType.LABELED_CRITERIA,
            "criteria": {
                "hallucination": "Determine if this AI response contains hallucinated information - factual claims, specific details, dates, names, or events that are not supported by or present in the retrieved context. The response should only make claims that can be verified against the provided reference material."
            },
            "prompt": None,
            # "prompt": hallucination_prompt
        },
    ]

    # get subset of dataset with metadata source = user
    data_subset = ls_client.list_examples(
        dataset_name=dataset_name, metadata={"source": "user"}
    )
    # even shorter for testing
    data_subset = list(data_subset)[:3]

    evals = []
    for config in eval_configs:
        evaluator_config = {
            "criteria": config["criteria"],
            "llm": llm,
        }
        if config["prompt"]:
            evaluator_config["prompt"] = config["prompt"]

        prepare_data = (
            prepare_data_labeled_criteria
            if config["type"] == EvaluatorType.LABELED_CRITERIA
            else prepare_data_criteria
        )

        evaluator = LangChainStringEvaluator(
            evaluator=config["type"], config=evaluator_config, prepare_data=prepare_data
        )
        evals.append(evaluator)

    _eval = await ls_client.aevaluate(
        target_wrapper,
        data=data_subset,
        evaluators=evals,
        description="setup testing",
        max_concurrency=4,
    )


if __name__ == "__main__":
    asyncio.run(main())
