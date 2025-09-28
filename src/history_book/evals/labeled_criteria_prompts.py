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

Provide your reasoning step by step, then respond with:
- Y if the response contains hallucinated information
- N if the response is factually consistent with the context

Reasoning:""")
