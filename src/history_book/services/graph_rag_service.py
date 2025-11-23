"""LangGraph-based RAG service for agentic chat functionality."""

import json
import logging

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from history_book.data_models.entities import Paragraph
from history_book.data_models.graph_state import AgentState
from history_book.database.repositories import BookRepositoryManager
from history_book.llm.config import LLMConfig
from history_book.llm.exceptions import LLMError
from history_book.services.agents.tools import search_book
from history_book.services.rag_service import (
    ITERATIVE_BOOK_SEARCH_PROMPT,
    RagService,
)

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
        max_iterations = 3  # TODO: Make configurable via GraphConfig

        # Check if max iterations reached
        if tool_iterations >= max_iterations:
            logger.info(
                f"Max tool iterations ({max_iterations}) reached, ending execution. "
                f"Retrieved {len(state.get('retrieved_paragraphs', []))} total paragraphs."
            )
            return "end"

        # Check if last message has tool calls
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                logger.info(
                    f"Tool calls detected: routing to tools node "
                    f"(iteration {tool_iterations + 1}/{max_iterations})"
                )
                return "tools"

        # No tool calls - LLM provided final answer
        logger.info(
            f"No tool calls detected - LLM provided final answer. "
            f"Total iterations: {tool_iterations}, "
            f"Retrieved paragraphs: {len(state.get('retrieved_paragraphs', []))}"
        )
        return "end"

    def _create_graph(self):
        """
        Build and compile the RAG graph with tool support.

        Graph structure:
        START → generate → [tools OR END]
                             ↓
                           tools → generate (loop back)

        Note: Automatic retrieve_node removed - all retrieval now via tools
        """
        # Initialize workflow with state schema
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("tools", self.tools_node)
        # Removed: workflow.add_node("retrieve", self._retrieve_node)
        # All retrieval now happens via search_book tool

        # Define edges
        workflow.add_edge(START, "generate")

        # Conditional routing after generate: tools or end
        workflow.add_conditional_edges(
            "generate", self._should_continue, {"tools": "tools", "end": END}
        )

        # Tools loop back to generate for synthesis
        workflow.add_edge("tools", "generate")

        # Compile with checkpointer for state persistence
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    # REMOVED: Automatic retrieve_node - all retrieval now via search_book tool
    # async def _retrieve_node(self, state: AgentState) -> dict:
    #     """
    #     Retrieval node: Fetch relevant paragraphs from vector database.
    #
    #     Delegates to RagService.retrieve_context() for reuse.
    #
    #     Args:
    #         state: Current agent state
    #
    #     Returns:
    #         Dict with updated retrieved_paragraphs field
    #     """
    #     question = state["question"]
    #
    #     try:
    #         # Delegate to RagService
    #         paragraphs = await self.rag_service.retrieve_context(
    #             query=question,
    #             min_results=self.min_results,
    #             max_results=self.max_results,
    #             similarity_cutoff=self.similarity_cutoff,
    #             retrieval_strategy="similarity_search",
    #         )
    #
    #         logger.info(f"Retrieved {len(paragraphs)} context paragraphs for query")
    #         return {"retrieved_paragraphs": paragraphs}
    #
    #     except Exception as e:
    #         logger.error(f"Retrieval failed: {e}")
    #         # Graceful degradation - continue without context
    #         return {"retrieved_paragraphs": []}

    def _extract_paragraphs_from_tools(self, messages: list) -> list:
        """
        Extract paragraph data from tool result messages.

        Tool results are ToolMessage objects with JSON content from search_book.
        Parse them and reconstruct Paragraph objects.

        Args:
            messages: List of messages including ToolMessages

        Returns:
            List of Paragraph objects extracted from tool results
        """
        paragraphs = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    # Parse tool result JSON
                    result = json.loads(msg.content)

                    # Extract excerpts if this is from search_book
                    if "excerpts" in result:
                        for idx, excerpt in enumerate(result["excerpts"]):
                            # Reconstruct Paragraph object with all available fields
                            para = Paragraph(
                                id=excerpt.get("id"),
                                text=excerpt["text"],
                                chapter_index=excerpt["chapter"],
                                page=excerpt["page"],
                                book_index=excerpt.get("book", 1),
                                paragraph_index=idx,  # Use index in result list
                            )
                            paragraphs.append(para)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse tool result: {e}")
                    continue

        if paragraphs:
            logger.info(f"Extracted {len(paragraphs)} paragraphs from tool results")

        return paragraphs

    def _extract_tool_queries_from_messages(self, messages: list) -> list[str]:
        """
        Extract the search queries that were sent to tools in previous iterations.

        This helps prevent the LLM from repeating the same unsuccessful searches
        and encourages query refinement.

        Args:
            messages: List of messages including AI messages with tool calls

        Returns:
            List of query strings that were used in previous tool calls
        """
        queries = []

        for msg in messages:
            # Check if message has tool_calls attribute (AIMessage with tool calls)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Extract query from tool call arguments
                    if isinstance(tool_call, dict):
                        # Dictionary format
                        args = tool_call.get("args", {})
                        if "query" in args:
                            queries.append(args["query"])
                    elif hasattr(tool_call, "get"):
                        # Object with get method
                        args = tool_call.get("args", {})
                        if "query" in args:
                            queries.append(args["query"])

        if queries:
            logger.info(
                f"Found {len(queries)} previous tool queries in message history"
            )

        return queries

    async def _generate_node(self, state: AgentState) -> dict:
        """
        Generation node: Create response using LLM with iterative tool calling support.

        Single-path implementation where tools are always available to the LLM.
        The LLM decides on each iteration whether to:
        1. Call search_book to retrieve (more) information
        2. Synthesize a final answer from accumulated context

        This enables iterative refinement: LLM can search multiple times with
        refined queries if initial context is insufficient.

        Args:
            state: Current agent state

        Returns:
            Dict with updated generation, messages, retrieved_paragraphs, and tool_iterations
        """
        question = state["question"]
        messages = state.get("messages", [])
        tool_iterations = state.get("tool_iterations", 0)

        try:
            # Extract paragraphs from tool results in messages
            tool_paragraphs = self._extract_paragraphs_from_tools(messages)

            # Get all accumulated paragraphs from state (reducer will deduplicate when updated)
            existing_paragraphs = state.get("retrieved_paragraphs", [])
            all_paragraphs = existing_paragraphs + tool_paragraphs

            # Format context section for prompt
            if all_paragraphs:
                # We have retrieved context - show it to LLM
                context = self.rag_service.format_context_for_book_answer(
                    all_paragraphs
                )
                context_section = (
                    f"PREVIOUSLY RETRIEVED EXCERPTS:\n\n{context}\n\n"
                    "Review the above excerpts. If they are sufficient to answer the question, "
                    "provide your answer with inline citations. If you need more specific information, "
                    "call search_book again with a refined query."
                )
                logger.info(
                    f"Generation node invoked with {len(all_paragraphs)} accumulated paragraphs"
                )
            else:
                # No context yet - prompt LLM to use search_book tool
                context_section = (
                    "(No excerpts retrieved yet. Use the search_book tool to find "
                    "relevant information from the book.)"
                )
                logger.info(
                    "Generation node invoked with no context - expecting tool call"
                )

            # Extract previous tool queries to prevent repeating failed searches
            previous_queries = self._extract_tool_queries_from_messages(messages)
            if previous_queries:
                context_section += (
                    "\n\n**Previous search queries attempted:**\n"
                    + "\n".join(f'- "{q}"' for q in previous_queries)
                    + "\n\nDo not repeat these exact queries. If you need more information, try a different, more specific search query."
                )

            # On final iteration (3/3), force answer instead of allowing more tool calls
            if tool_iterations >= 2:  # 0-indexed: 0, 1, 2 = iterations 1, 2, 3
                # Final iteration - must provide answer
                llm_to_use = self.llm  # No tools bound
                context_section += (
                    "\n\n**FINAL ITERATION: You must provide an answer now. "
                    "If the excerpts above are sufficient, answer the question with inline citations. "
                    "If the excerpts are insufficient or irrelevant, clearly state: "
                    "'I could not find information about this topic in The Penguin History of the World.'**"
                )
                logger.info(
                    "Final iteration (3/3) - forcing answer, no tools available"
                )
            else:
                # Normal iteration - tools available
                llm_to_use = self.llm.bind_tools(self.tools)
                logger.info(
                    f"Iteration {tool_iterations + 1}/3 - tools available for refinement"
                )

            # Build prompt with iterative tool calling support
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", ITERATIVE_BOOK_SEARCH_PROMPT),
                    ("human", f"{context_section}\n\nUSER QUESTION: {{query}}"),
                ]
            )

            # Build message history (exclude system/human, just use chat history)
            message_history = list(messages) if messages else []

            # Invoke LLM (with or without tools depending on iteration)
            chain = prompt | llm_to_use
            response = await chain.ainvoke(
                {"query": question, "chat_history": message_history}
            )

            # Update iteration counter if tools were called
            new_tool_iterations = tool_iterations
            if hasattr(response, "tool_calls") and response.tool_calls:
                new_tool_iterations = tool_iterations + 1
                logger.info(
                    f"LLM called {len(response.tool_calls)} tool(s), "
                    f"iteration {new_tool_iterations}/3"
                )
            else:
                logger.info("LLM provided final answer without calling tools")

            return {
                "generation": response.content or "",
                "messages": [response],
                "retrieved_paragraphs": all_paragraphs,
                "tool_iterations": new_tool_iterations,
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
