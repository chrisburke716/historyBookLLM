

# use langsmith 'evaluate' method

import asyncio
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain.evaluation import Criteria, EvaluatorType

from history_book.services import ChatService

async def main():

    ls_client = Client()

    chat_service = ChatService()

    async def target_wrapper(inputs):
        session = await chat_service.create_session()
        session_id = session.id
        user_message = inputs["question"]
        return await chat_service.send_message(session_id=session_id, user_message=user_message)
    
    def prepare_data(run, example):
        return {
            "prediction": run.outputs.get("content"),
            "input": example.inputs.get("question"),
        }

    dataset_name = "History Book Eval Queries"

    eval_names = [
        # Criteria.CONCISENESS,
        # Criteria.CORRECTNESS, # should be labeled criteria
        # Criteria.RELEVANCE,
        Criteria.HELPFULNESS,
        # Criteria.HARMFULNESS,
    ]

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)


    evals = []
    for eval_name in eval_names:
        config={
            "criteria": eval_name,
            "llm": llm,
        }
        # evaluator_chain = load_evaluator(evaluator=Criteria, llm=eval_name, Criteria)
        evaluator = LangChainStringEvaluator(evaluator=EvaluatorType.CRITERIA, config=config, prepare_data=prepare_data)
        evals.append(evaluator)

    eval = await ls_client.aevaluate(
        target_wrapper,
        data = dataset_name,
        evaluators = evals,
        description = "setup testing"
    )

if __name__ == "__main__":
    asyncio.run(main())