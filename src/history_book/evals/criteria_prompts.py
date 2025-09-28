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


IDK_PROMPT = PromptTemplate.from_template("""
You are evaluating whether an AI response appropriately expresses uncertainty or lack of knowledge.

User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Determine if the AI response indicates that it doesn't know the answer or couldn't find the information. Look for:
- Explicit statements like "I don't know", "I'm not sure", "I couldn't find"
- Hedging language indicating uncertainty about the answer
- Admissions of insufficient information or inability to answer
- Phrases like "I don't have enough information", "This is unclear", "I cannot determine"

Provide your reasoning step by step, then respond with:
- Y if the response expresses uncertainty or lack of knowledge
- N if the response provides a confident answer without expressing uncertainty

Your evaluation:""")


RELEVANCE_PROMPT = PromptTemplate.from_template("""
You are evaluating whether an AI response is relevant to the user's question.

User Question: {input}
AI Response: {output}

Evaluation Criteria: {criteria}

Instructions: Determine if the AI response directly addresses the user's question. Look for:
- Does the response answer what was specifically asked?
- Is the response on-topic and focused on the question?
- Does it avoid going off on tangents or providing unrelated information?
- If the question has multiple parts, does it address the main components?

Provide your reasoning step by step, then respond with:
- Y if the response is relevant and directly addresses the question
- N if the response is off-topic, tangential, or doesn't answer what was asked

Your evaluation:""")
