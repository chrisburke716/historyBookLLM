"""
Prompt templates for pairwise comparison evaluations.

These prompts are used to compare two AI responses and determine which is better
according to specific criteria. Each prompt expects the LLM judge to provide
reasoning followed by a final judgment of A, B, or TIE.
"""

from langchain.prompts import PromptTemplate

PAIRWISE_HELPFULNESS_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which is more helpful.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Instructions: Compare these responses on helpfulness. Consider:
- Which response better answers the user's question?
- Which provides more relevant and useful information?
- Which is clearer and easier to understand?
- Which would be more helpful to the user?

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, state your judgment:
- "A" if Response A is more helpful
- "B" if Response B is more helpful
- "TIE" if both are equally helpful

On the final line, write only A, B, or TIE.

Reasoning:""")

PAIRWISE_HALLUCINATION_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which has fewer hallucinations.

User Question: {question}

Response A:
{response_a}

Context Available to Response A:
{context_a}

Response B:
{response_b}

Context Available to Response B:
{context_b}

Instructions: Compare these responses for hallucinations. For each response:
- Check if it makes claims not supported by ITS OWN context
- Look for invented facts, dates, names, or events
- Identify any contradictions with the provided context

NOTE: Each response should only be judged against its own retrieved context.
A response is NOT hallucinating just because it didn't retrieve the same documents.

First, provide your step-by-step reasoning.
Then, on the final line, state which response has FEWER hallucinations:
- "A" if Response A has fewer hallucinations (is better grounded)
- "B" if Response B has fewer hallucinations (is better grounded)
- "TIE" if both are equally well-grounded or equally hallucinated

On the final line, write only A, B, or TIE.

Reasoning:""")

PAIRWISE_COHERENCE_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which is more coherent.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Instructions: Compare these responses on coherence and structure. Consider:
- Which has clearer logical flow from one idea to the next?
- Which is better organized and structured?
- Which avoids contradictions within the response?
- Which presents ideas in a more understandable sequence?

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, state your judgment:
- "A" if Response A is more coherent
- "B" if Response B is more coherent
- "TIE" if both are equally coherent

On the final line, write only A, B, or TIE.

Reasoning:""")

PAIRWISE_COMPLETENESS_PROMPT = PromptTemplate.from_template("""
You are comparing two AI responses to determine which more completely answers the question.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Instructions: Compare these responses on completeness. Consider:
- Which addresses more aspects of the question?
- Which provides more thorough coverage of the topic?
- Which gives more comprehensive information?
- If the question has multiple parts, which addresses more parts?

Note: More complete doesn't necessarily mean longer. Focus on coverage of the question's scope.

First, provide your step-by-step reasoning comparing the responses.
Then, on the final line, state your judgment:
- "A" if Response A is more complete
- "B" if Response B is more complete
- "TIE" if both are equally complete

On the final line, write only A, B, or TIE.

Reasoning:""")
