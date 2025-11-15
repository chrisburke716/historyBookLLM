"""LangGraph-based RAG service for agentic chat functionality."""

import logging

from langchain.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from history_book.data_models.graph_state import AgentState
from history_book.database.repositories import BookRepositoryManager
from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError
from history_book.llm.utils import format_context_for_llm
from history_book.services.agents.tools import search_book
from history_book.services.rag_service import RagService

logger = logging.getLogger(__name__)


class GraphRagService:
    """
    LangGraph-based RAG service for agentic chat functionality.

    Implements a simple RAG graph: retrieve → generate
    Designed to be extended with tool calling and multi-step reasoning.

    Uses composition with RagService to reuse existing logic.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        repository_manager: BookRepositoryManager,
        min_context_results: int = 5,
        max_context_results: int = 40,
        context_similarity_cutoff: float = 0.4,
    ):
        """
        Initialize the graph RAG service.

        Args:
            llm_config: Configuration for the LLM
            repository_manager: Repository manager for document retrieval
            min_context_results: Minimum number of documents to retrieve
            max_context_results: Maximum number of documents to retrieve
            context_similarity_cutoff: Similarity threshold for retrieval
        """
        self.config = llm_config
        self.min_results = min_context_results
        self.max_results = max_context_results
        self.similarity_cutoff = context_similarity_cutoff

        # Use composition: create RagService to reuse its methods
        self.rag_service = RagService(
            llm_config=llm_config, repository_manager=repository_manager
        )

        # Enable streaming on the LLM
        self.llm = self._create_streaming_model()

        # Initialize tools
        self.tools = [search_book]  # TODO: Make configurable via GraphConfig
        self.tools_node = ToolNode(self.tools)

        # Build and compile graph
        self.graph = self._create_graph()

    def _create_streaming_model(self):
        """Create LLM with streaming enabled for LangGraph."""
        # Use RagService's create_chat_model but enable streaming
        if self.config.provider == "openai":
            from langchain_openai import ChatOpenAI  # noqa PLC0415

            return ChatOpenAI(
                model=self.config.model_name,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                streaming=True,  # Enable streaming
                **self.config.provider_kwargs,
            )
        elif self.config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa PLC0415

            return ChatAnthropic(
                model=self.config.model_name,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                streaming=True,  # Enable streaming
                **self.config.provider_kwargs,
            )
        else:
            # Fallback to RagService's method
            return self.rag_service.create_chat_model()

    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if agent should call tools or end execution.

        Routing logic:
        - If last message has tool_calls AND iterations < max: route to "tools"
        - Otherwise: route to "end"

        Args:
            state: Current agent state

        Returns:
            "tools" if agent should execute tools, "end" if execution should finish
        """
        messages = state.get("messages", [])
        tool_iterations = state.get("tool_iterations", 0)

        # Check if max iterations reached
        if tool_iterations >= 3:  # TODO: Make configurable via GraphConfig
            logger.info(
                f"Max tool iterations ({3}) reached, ending execution"
            )
            return "end"

        # Check if last message has tool calls
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                logger.info(
                    f"Tool calls detected in last message, routing to tools node "
                    f"(iteration {tool_iterations + 1})"
                )
                return "tools"

        # No tool calls or empty messages, end execution
        logger.info("No tool calls detected, ending execution")
        return "end"

    def _create_graph(self):
        """
        Build and compile the RAG graph.

        Graph structure: START → retrieve → generate → END
        """
        # Initialize workflow with state schema
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        # Define edges (simple linear flow)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile with checkpointer for state persistence
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    async def _retrieve_node(self, state: AgentState) -> dict:
        """
        Retrieval node: Fetch relevant paragraphs from vector database.

        Delegates to RagService.retrieve_context() for reuse.

        Args:
            state: Current agent state

        Returns:
            Dict with updated retrieved_paragraphs field
        """
        question = state["question"]

        try:
            # Delegate to RagService
            paragraphs = await self.rag_service.retrieve_context(
                query=question,
                min_results=self.min_results,
                max_results=self.max_results,
                similarity_cutoff=self.similarity_cutoff,
                retrieval_strategy="similarity_search",
            )

            logger.info(f"Retrieved {len(paragraphs)} context paragraphs for query")
            return {"retrieved_paragraphs": paragraphs}

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Graceful degradation - continue without context
            return {"retrieved_paragraphs": []}

    async def _generate_node(self, state: AgentState) -> dict:
        """
        Generation node: Create response using LLM + context.

        Uses RagService for formatting but LangGraph for execution.

        Args:
            state: Current agent state

        Returns:
            Dict with updated generation and messages fields
        """
        question = state["question"]
        messages = state["messages"]
        paragraphs = state["retrieved_paragraphs"]

        try:
            # Build prompt based on whether we have context
            if paragraphs:
                # Use RagService to format context
                context = self.rag_service.format_context(paragraphs)
                formatted_context = format_context_for_llm(
                    context, self.config.max_context_length
                )

                # Create prompt template for RAG
                rag_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self.config.system_message),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{context}\n\nQuestion: {query}"),
                    ]
                )

                # Build message list for LLM
                message_history = list(messages) if messages else []

                # Invoke LLM with context
                chain = rag_prompt | self.llm
                response = await chain.ainvoke(
                    {
                        "chat_history": message_history,
                        "context": formatted_context,
                        "query": question,
                    }
                )
            else:
                # No context available - answer without retrieval
                simple_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self.config.system_message),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{query}"),
                    ]
                )

                message_history = list(messages) if messages else []

                # Invoke LLM without context
                chain = simple_prompt | self.llm
                response = await chain.ainvoke(
                    {"chat_history": message_history, "query": question}
                )

            return {
                "generation": response.content,
                "messages": [AIMessage(content=response.content)],
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise LLMError(f"Response generation failed: {e}") from e

    async def invoke(
        self,
        question: str,
        messages: list,
        session_id: str,
        config: dict | None = None,
    ) -> AgentState:
        """
        Execute graph synchronously.

        Args:
            question: User query
            messages: Chat history (ChatMessage entities)
            session_id: Session identifier (used as thread_id)
            config: Optional RunnableConfig for tracing

        Returns:
            Final state after graph execution
        """
        # Convert messages to LangChain format using RagService
        lc_messages = self.rag_service.convert_to_langchain_messages(messages)

        # Build initial state
        initial_state: AgentState = {
            "messages": lc_messages,
            "question": question,
            "retrieved_paragraphs": [],
            "generation": "",
            "session_id": session_id,
            "metadata": {},
        }

        # Execute graph with thread_id for checkpointing
        config = config or {}
        config["configurable"] = {"thread_id": session_id}

        # Add tags for LangSmith tracing
        config.setdefault("tags", []).extend(["agent", "langgraph", "simple_rag"])

        result = await self.graph.ainvoke(initial_state, config=config)
        return result

    async def stream(
        self,
        question: str,
        messages: list,
        session_id: str,
        config: dict | None = None,
    ):
        """
        Execute graph with streaming (token-by-token).

        Uses stream_mode="messages" to get LLM token chunks.

        Args:
            question: User query
            messages: Chat history
            session_id: Session identifier
            config: Optional RunnableConfig

        Yields:
            Token chunks and state updates
        """
        # Convert messages to LangChain format using RagService
        lc_messages = self.rag_service.convert_to_langchain_messages(messages)

        # Build initial state
        initial_state: AgentState = {
            "messages": lc_messages,
            "question": question,
            "retrieved_paragraphs": [],
            "generation": "",
            "session_id": session_id,
            "metadata": {},
        }

        config = config or {}
        config["configurable"] = {"thread_id": session_id}

        # Add tags for LangSmith tracing
        config.setdefault("tags", []).extend(
            ["agent", "langgraph", "simple_rag", "streaming"]
        )

        # Stream with messages mode to get token chunks
        async for chunk in self.graph.astream(
            initial_state,
            config=config,
            stream_mode="messages",  # Token-by-token streaming
        ):
            yield chunk
