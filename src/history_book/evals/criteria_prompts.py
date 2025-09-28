"""
Prompt templates for CRITERIA evaluations (no reference material needed).

These prompts expect inputs: {input}, {output}, {criteria}
"""

from langchain.prompts import PromptTemplate

FACTUAL_ACCURACY_PROMPT = PromptTemplate.from_template("""
You are evaluating the factual accuracy of an AI response based on general knowledge.

User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Assess whether the AI response contains factually correct information based on your knowledge. Look for:
- Incorrect dates, names, or historical facts
- Contradictory or impossible claims
- Misinformation or commonly confused facts

Provide your reasoning step by step, then respond with:
- Y if the response is factually accurate
- N if the response contains factual inaccuracies

Your evaluation:""")


COHERENCE_PROMPT = PromptTemplate.from_template("""
You are evaluating the coherence and logical flow of an AI response.

User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Assess whether the response is logically structured, coherent, and flows well from one idea to the next. Look for:
- Clear logical progression of ideas
- Consistent line of reasoning
- Well-organized structure
- Absence of contradictory statements within the response

Provide your reasoning step by step, then respond with:
- Y if the response is coherent and well-structured
- N if the response lacks coherence or logical flow

Your evaluation:""")
