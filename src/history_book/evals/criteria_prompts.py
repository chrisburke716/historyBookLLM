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

First, provide your step-by-step reasoning. Then, on the final line, provide your answer:
- Y if the response is factually accurate
- N if the response contains factual inaccuracies

On the final line, write only Y or N.

Reasoning:""")


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

First, provide your step-by-step reasoning. Then, on the final line, provide your answer:
- Y if the response is coherent and well-structured
- N if the response lacks coherence or logical flow

On the final line, write only Y or N.

Reasoning:""")


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

First, provide your step-by-step reasoning. Then, on the final line, provide your answer:
- Y if the response expresses uncertainty or lack of knowledge
- N if the response provides a confident answer without expressing uncertainty

On the final line, write only Y or N.

Reasoning:""")


RELEVANCE_PROMPT = PromptTemplate.from_template("""
You are evaluating whether the retrieved context is relevant to the user's question.

User Question: {input}
Retrieved Context: {output}

Evaluation Criteria: {criteria}

Instructions: Determine if the retrieved context contains information relevant to answering the user's question. Look for:
- Does the context contain information that could help answer the question?
- Is the context on-topic and related to what was asked?
- Does the context avoid being completely unrelated or off-topic?
- If the question has multiple parts, does the context address any of the main components?

Note: The context doesn't need to contain the complete answer, just relevant information that could contribute to answering the question.

First, provide your step-by-step reasoning. Then, on the final line, provide your answer:
- Y if the context is relevant and contains information related to the question
- N if the context is off-topic, unrelated, or doesn't contain relevant information

On the final line, write only Y or N.

Reasoning:""")
