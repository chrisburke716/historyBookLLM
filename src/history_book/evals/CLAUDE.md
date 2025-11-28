# CLAUDE.md - RAG Evaluation Framework

This directory contains a comprehensive evaluation framework for measuring the quality of the History Book RAG chat system. The framework integrates with LangSmith to track experiments and provides both LLM-based and function-based evaluators.

## Quick Start

### Running Evaluations

```bash
# Load environment variables (includes OPENAI_API_KEY and LANGSMITH_API_KEY)
source source_env.sh

# Run evaluations with CLI flags
poetry run python scripts/run_evals.py [OPTIONS]
```

**CLI Flags:**
- `--mode {agent,legacy}` - Choose which system to evaluate (default: agent)
  - `agent`: LangGraph-based RAG with iterative tool calling
  - `legacy`: LCEL-based RAG with direct retrieval
- `--subset` - Run on 3-query subset for quick testing
- `--full` - Run on full 100-query dataset (default behavior)

**Examples:**
```bash
# Quick test with agent system (3 queries)
poetry run python scripts/run_evals.py --mode agent --subset

# Full evaluation of legacy system
poetry run python scripts/run_evals.py --mode legacy --full

# Default: agent mode, full dataset
poetry run python scripts/run_evals.py
```

The evaluation runs against 100 queries in the "History Book Eval Queries" LangSmith dataset and measures 8 different quality dimensions.

### Viewing Results

Results are automatically uploaded to LangSmith where you can:
- Compare performance across evaluation runs
- View individual query results and scores
- Analyze metadata (LLM config, RAG parameters, etc.)
- Track improvements over time

## Architecture Overview

The evaluation framework follows a **registry pattern** for extensibility and uses **abstract base classes** to define clear interfaces.

### Core Components

```
src/history_book/evals/
├── base.py                          # Abstract base classes
├── evaluators.py                    # 8 concrete evaluator implementations
├── registry.py                      # Registration and discovery system
├── criteria_prompts.py              # Prompts for CRITERIA evaluators
└── labeled_criteria_prompts.py      # Prompts for LABELED_CRITERIA evaluators
```

### Base Classes (`base.py`)

**BaseEvaluator (Abstract)**
- Defines the interface all evaluators must implement
- Uses ChatOpenAI with "gpt-5-mini-2025-08-07" at temperature 1.0 by default
- Key methods:
  - `name`: Unique evaluator identifier
  - `evaluator_type`: Type of evaluation (CRITERIA, LABELED_CRITERIA, or function)
  - `create_langchain_evaluator()`: Creates LangSmith-compatible evaluator
  - `prepare_data(run, example)`: Formats data for evaluation
  - `get_prompt()`: Optional custom prompt template

**CriteriaEvaluator (extends BaseEvaluator)**
- Evaluates responses **without** reference material
- Uses: `prediction` (AI response) + `input` (user question)
- Example: Coherence, Helpfulness, Factual Accuracy

**LabeledCriteriaEvaluator (extends BaseEvaluator)**
- Evaluates responses **against** reference material
- Uses: `prediction` + `input` + `reference` (retrieved context)
- Example: Hallucination detection, IDK appropriateness

**FunctionEvaluator (extends BaseEvaluator)**
- Direct computation without LLM
- Implements `evaluate(run, example)` method
- Example: Document count

### Registry System (`registry.py`)

The registry provides **dynamic evaluator discovery and instantiation**:

```python
from history_book.evals import get_all_evaluators, register_evaluator

# Get all evaluators
evaluators = get_all_evaluators(llm=my_llm)

# Get specific evaluator
from history_book.evals import create_evaluator
evaluator = create_evaluator("hallucination", llm=my_llm)

# List available evaluators
from history_book.evals import list_evaluators
names = list_evaluators()  # ['helpfulness', 'hallucination', ...]
```

**Key Functions:**
- `@register_evaluator` - Decorator to auto-register new evaluators
- `get_evaluator(name)` - Retrieve evaluator class by name
- `create_evaluator(name, **kwargs)` - Instantiate evaluator
- `get_all_evaluators(**kwargs)` - Create all registered evaluators
- `get_prompt_evaluators(**kwargs)` - Get only LLM-based evaluators
- `get_function_evaluators(**kwargs)` - Get only function-based evaluators

## Built-in Evaluators

The framework includes **8 pre-built evaluators** measuring different quality dimensions:

### LLM-Based Evaluators (7 evaluators)

1. **HelpfulnessEvaluator** (CriteriaEvaluator)
   - Measures: How helpful the response is to the user
   - Uses: LangChain's built-in `Criteria.HELPFULNESS`

2. **FactualAccuracyEvaluator** (CriteriaEvaluator)
   - Measures: Factual correctness of dates, names, events
   - Checks: Against general historical knowledge
   - Prompt: `FACTUAL_ACCURACY_PROMPT`

3. **CoherenceEvaluator** (CriteriaEvaluator)
   - Measures: Logical flow and structure
   - Checks: Clear progression, consistent reasoning, organization
   - Prompt: `COHERENCE_PROMPT`

4. **HallucinationEvaluator** (LabeledCriteriaEvaluator)
   - Measures: Whether response contains information not in retrieved context
   - Checks: Made-up facts, contradictions, unsupported claims
   - Prompt: `HALLUCINATION_PROMPT`
   - Reference: Retrieved context from RAG system

5. **IdkEvaluator** (CriteriaEvaluator)
   - Measures: Whether AI expresses uncertainty
   - Checks: "I don't know", hedging language, admissions of insufficient info
   - Prompt: `IDK_PROMPT`

6. **IdkAppropriateEvaluator** (LabeledCriteriaEvaluator)
   - Measures: Whether uncertainty level matches available context
   - Appropriate: Says "IDK" when context lacks info OR provides answer when supported
   - Inappropriate: Says "IDK" despite relevant context OR answers without support
   - Prompt: `IDK_APPROPRIATE_PROMPT`
   - Reference: Retrieved context

7. **RelevanceEvaluator** (CriteriaEvaluator)
   - Measures: Whether retrieved context is relevant to the question
   - Special: Evaluates the **context itself**, not the AI response
   - Prompt: `RELEVANCE_PROMPT`

### Function-Based Evaluators (1 evaluator)

8. **DocumentCountEvaluator** (FunctionEvaluator)
   - Measures: Number of documents retrieved per query
   - Returns: Count as score with metadata
   - No LLM needed - direct computation

## Adding New Evaluators

### Step 1: Choose the Right Base Class

- **CriteriaEvaluator**: For evaluating response quality without reference material
- **LabeledCriteriaEvaluator**: For evaluating against retrieved context
- **FunctionEvaluator**: For direct computation without LLM

### Step 2: Create Evaluator Class

Example - Adding a "Conciseness" evaluator:

```python
# In evaluators.py

from history_book.evals.base import CriteriaEvaluator
from history_book.evals.registry import register_evaluator
from langchain.prompts import PromptTemplate

CONCISENESS_PROMPT = PromptTemplate.from_template(
    """You are assessing a submitted answer on the criterion of conciseness.

Criteria: {criteria}

Submission: {output}

Think through step by step:
1. Does the response contain unnecessary repetition or verbosity?
2. Could the same information be conveyed more concisely?
3. Is the response appropriately detailed for the question?

Based on your assessment, respond with Y or N on the last line.
Y: Response is appropriately concise
N: Response is unnecessarily verbose or repetitive
"""
)

@register_evaluator
class ConcisenessEvaluator(CriteriaEvaluator):
    @property
    def name(self) -> str:
        return "conciseness"

    def get_criteria(self) -> dict[str, str]:
        return {
            "conciseness": "Is the response concise without unnecessary verbosity?"
        }

    def get_prompt(self) -> PromptTemplate:
        return CONCISENESS_PROMPT
```

### Step 3: Register and Use

The `@register_evaluator` decorator automatically registers your evaluator. It will be included when calling:

```python
from history_book.evals import get_all_evaluators

evaluators = get_all_evaluators(llm=my_llm)
# Now includes ConcisenessEvaluator!
```

### Example: Function-Based Evaluator

For metrics that don't need LLM evaluation:

```python
from history_book.evals.base import FunctionEvaluator
from history_book.evals.registry import register_evaluator
from langsmith.schemas import Example, Run

@register_evaluator
class AverageTokenLengthEvaluator(FunctionEvaluator):
    @property
    def name(self) -> str:
        return "avg_token_length"

    def evaluate(self, run: Run, example: Example) -> dict:
        response = run.outputs.get("response", "")
        # Simple approximation: tokens ≈ words
        tokens = len(response.split())
        return {
            "key": self.name,
            "score": tokens,
            "comment": f"Response length: {tokens} tokens"
        }
```

## Evaluation Prompts

### Prompt Structure

All LLM-based evaluators use prompts that follow this pattern:

1. **Role definition**: "You are assessing a submitted answer..."
2. **Criteria statement**: Clear definition of what to evaluate
3. **Input variables**: `{input}`, `{output}`, `{criteria}`, and optionally `{reference}`
4. **Step-by-step reasoning**: Instructions to think through the evaluation
5. **Structured output**: "Respond with Y or N on the last line"

### Prompt Files

- `criteria_prompts.py` - Prompts for evaluators WITHOUT reference material:
  - `FACTUAL_ACCURACY_PROMPT`
  - `COHERENCE_PROMPT`
  - `IDK_PROMPT`
  - `RELEVANCE_PROMPT`

- `labeled_criteria_prompts.py` - Prompts that use retrieved context:
  - `HALLUCINATION_PROMPT`
  - `IDK_APPROPRIATE_PROMPT`

### Adding Custom Prompts

When creating a custom prompt, follow this template:

```python
from langchain.prompts import PromptTemplate

YOUR_PROMPT = PromptTemplate.from_template(
    """You are assessing a submitted answer on the criterion of [X].

Criteria: {criteria}

User Question: {input}
Submission: {output}
[Optional] Reference Material: {reference}

Think through step by step:
1. [Specific check 1]
2. [Specific check 2]
3. [Specific check 3]

Based on your assessment, respond with Y or N on the last line.
Y: [What Y means]
N: [What N means]
"""
)
```

## Dataset Management

### Current Dataset

**Name**: "History Book Eval Queries" (in LangSmith)
- **Size**: 100 queries
- **Location**: `/notebooks/eval_dataset_queries.csv`
- **Fields**: `query`, `source`, `complexity`
- **Sources**:
  - `user`: Real user questions
  - `synth`: Synthetic/generated questions

**Sample queries**:
- "What were the first civilizations?"
- "when did julius cesar rule?"
- "how does the author define barbarians?"
- "What were the main causes of World War I?"

### Building New Datasets

Use the Jupyter notebook to create or modify evaluation datasets:

```bash
# Open the dataset builder notebook
jupyter notebook notebooks/BuildEvalDataset.ipynb
```

The notebook provides tools for:
- Generating synthetic queries
- Importing user queries from chat logs
- Labeling query complexity
- Uploading to LangSmith

### Filtering Dataset

In `run_evals.py`, you can filter the dataset:

```python
# Filter by source
# examples = ls_client.list_examples(dataset_name=dataset_name, metadata={"source": "user"})

# Filter by complexity
# examples = ls_client.list_examples(dataset_name=dataset_name, metadata={"complexity": "simple"})
```

## Integration Points

The evaluation framework integrates with the main codebase through minimal modifications:

### ChatService Integration (`src/history_book/services/chat_service.py`)

**ChatResult dataclass**:
```python
@dataclass
class ChatResult:
    message: ChatMessage
    retrieved_paragraphs: list[Paragraph]
```

**Modified methods**:
- `send_message()` - Returns `ChatResult` instead of just `ChatMessage`
- `get_eval_metadata()` - Extracts configuration for experiment tracking

### API Integration (`src/history_book/api/routes/chat.py`)

**Optimizations**:
- Updated to handle `ChatResult`
- Uses retrieved paragraphs for citations (no extra DB calls)
- Passes context through for evaluation

### Evaluation Script (`scripts/run_evals.py`)

**Target wrapper function**:
```python
async def target_wrapper(inputs: dict) -> dict:
    result = await chat_service.send_message(...)
    return {
        "response": result.message.content,
        "retrieved_context": format_paragraphs(result.retrieved_paragraphs)
    }
```

The wrapper:
1. Creates a new chat session for each query
2. Sends the message through ChatService
3. Returns both the AI response and retrieved context
4. Enables LabeledCriteriaEvaluators to work properly

## Configuration & Metadata

### Tracked Metadata

Every evaluation run captures comprehensive metadata for reproducibility:

**LLM Configuration**:
- Provider (OpenAI/Anthropic)
- Model name
- Temperature
- Max tokens
- System message

**RAG Configuration**:
- Minimum context results
- Maximum context results
- Similarity cutoff threshold
- Retrieval strategy

**Evaluator Configuration**:
- Evaluator LLM model
- Evaluator temperature

### Accessing Metadata

```python
from history_book.services import ChatService

chat_service = ChatService()
metadata = chat_service.get_eval_metadata()

# metadata contains:
# {
#   "llm": {...},
#   "rag": {...},
#   "evaluator": {...}
# }
```

### Environment Variables

Required for running evaluations:

```bash
# OpenAI API key for chat responses AND evaluations
OPENAI_API_KEY=your-openai-api-key

# LangSmith API key for uploading results
LANGSMITH_API_KEY=your-langsmith-api-key

# Optional: LangSmith project
LANGCHAIN_PROJECT=history-book-evals

# Load all variables
source source_env.sh
```

## Development Workflow

### 1. Develop New Evaluator

```bash
# Edit evaluators.py to add your evaluator class
# Add prompt to criteria_prompts.py or labeled_criteria_prompts.py if needed
```

### 2. Test Locally

```python
# Quick test in Python REPL
from history_book.evals import create_evaluator
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=1.0)
evaluator = create_evaluator("your_evaluator_name", llm=llm)

# Test the evaluator
print(evaluator.name)
print(evaluator.get_criteria())
```

### 3. Run Full Evaluation

```bash
# Run against full dataset
source source_env.sh
poetry run python scripts/run_evals.py
```

### 4. Review Results

- Visit LangSmith dashboard
- Navigate to your project
- View the latest evaluation run
- Compare scores across evaluators
- Drill into individual examples

### 5. Iterate

Based on results:
- Adjust RAG parameters (similarity cutoff, max results, etc.)
- Modify LLM configuration (model, temperature, system message)
- Refine evaluator prompts
- Add new evaluators for additional quality dimensions

## Best Practices

### When Adding Evaluators

1. **Clear criteria**: Define exactly what constitutes "good" vs "bad"
2. **Step-by-step reasoning**: Break down the evaluation into specific checks
3. **Consistent format**: Follow the Y/N output pattern
4. **Appropriate base class**: Choose CriteriaEvaluator vs LabeledCriteriaEvaluator carefully
5. **Descriptive names**: Use clear, searchable evaluator names

### When Running Evaluations

1. **Use metadata**: Always include run description and configuration metadata
2. **Control concurrency**: Adjust `max_concurrency` based on rate limits
3. **Filter dataset**: Test on subset first before running full evaluation
4. **Track changes**: Document what changed between evaluation runs
5. **Version control**: Commit code changes before running evaluations

### When Analyzing Results

1. **Look for patterns**: Are certain query types failing?
2. **Cross-reference metrics**: High hallucination + high confidence = problem
3. **Check outliers**: Investigate queries with unexpected scores
4. **Compare runs**: Use LangSmith comparison view to track improvements
5. **Validate evaluators**: Spot-check LLM evaluator decisions for correctness

## Related Files

- Main eval script: `/scripts/run_evals.py`
- Dataset CSV: `/notebooks/eval_dataset_queries.csv`
- Dataset builder: `/notebooks/BuildEvalDataset.ipynb`
- Environment helper: `/source_env.sh`
- ChatService: `/src/history_book/services/chat_service.py`
- Chat API: `/src/history_book/api/routes/chat.py`

## Future Enhancements

Potential improvements to consider:

1. **More evaluators**: Tone, style, citation accuracy, response length
2. **Human evaluation**: Collect human labels for calibration
3. **A/B testing**: Compare different prompts or configurations
4. **Automatic optimization**: Use eval results to tune hyperparameters
5. **Regression detection**: Alert on metric drops between runs
6. **Custom datasets**: Create domain-specific evaluation sets
7. **Multi-model evaluation**: Compare OpenAI vs Anthropic vs others
