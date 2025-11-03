"""
Prompt templates for LABELED_CRITERIA evaluations (with reference material).

These prompts expect inputs: {input}, {output}, {criteria}, {reference}
"""

from langchain.prompts import PromptTemplate

HALLUCINATION_PROMPT = PromptTemplate.from_template("""
You are evaluating an AI response for hallucinations against retrieved context.

Retrieved Context: {reference}
User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Check if the AI response contains any factual claims that are not supported by the retrieved context. Look for:
- Made-up facts, dates, names, or events not mentioned in the context
- Information that contradicts the retrieved context
- Specific details or claims not present in the reference material

First, provide your step-by-step reasoning. Then, on the final line, provide your answer:
- Y if the response contains hallucinated information
- N if the response is factually consistent with the context

On the final line, write only Y or N.

Reasoning:""")


IDK_APPROPRIATE_PROMPT = PromptTemplate.from_template("""
You are evaluating whether an "I don't know" response is appropriate given the retrieved context.

Retrieved Context: {reference}
User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Determine if the AI's uncertainty or knowledge claim is appropriate given the available context. Look for:

APPROPRIATE (respond Y):
- AI says "I don't know" AND the retrieved context lacks relevant information
- AI provides an answer AND the retrieved context contains supporting information

INAPPROPRIATE (respond N):
- AI says "I don't know" BUT the retrieved context contains relevant information to answer the question
- AI provides a confident answer BUT the retrieved context lacks sufficient information to support that answer

First, provide your step-by-step reasoning. Then, on the final line, provide your answer:
- Y if the response is appropriate given the available context
- N if there is a mismatch (inappropriate idk or inappropriate confidence)

On the final line, write only Y or N.

Reasoning:""")
