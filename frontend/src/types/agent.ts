/**
 * TypeScript interfaces for the Agent API (LangGraph-based).
 *
 * These types extend the base types from index.ts with agent-specific fields.
 * Currently, the Agent API has the same request/response structure as Chat API,
 * with the addition of metadata in responses.
 *
 * Future extensions (Phase 7+):
 * - Tool call information
 * - Reasoning steps
 * - Graph execution details
 * - Planning information
 */

/**
 * Agent-specific metadata returned in message responses.
 * This provides insights into the graph execution.
 */
export interface AgentMetadata {
  /**
   * Number of paragraphs retrieved from vector database
   */
  num_retrieved_paragraphs?: number;

  /**
   * Type of graph execution (e.g., "simple_rag")
   */
  graph_execution?: string;

  /**
   * Future fields for advanced agent features:
   * - nodes_executed?: string[];  // List of graph nodes that ran
   * - execution_time_ms?: number;  // Total execution time
   * - tool_calls?: ToolCall[];  // Tools invoked during execution
   * - reasoning_steps?: ReasoningStep[];  // Planning/reflection steps
   */
}

/**
 * Future: Tool call information
 *
 * export interface ToolCall {
 *   name: string;
 *   args: Record<string, any>;
 *   result: any;
 * }
 */

/**
 * Future: Reasoning step information
 *
 * export interface ReasoningStep {
 *   type: 'plan' | 'reflection' | 'grading' | 'rewrite';
 *   content: string;
 *   metadata?: Record<string, any>;
 * }
 */

/**
 * Future: Graph visualization data
 *
 * export interface GraphVisualization {
 *   mermaid: string;
 *   nodes: string[];
 *   edges: [string, string][];
 * }
 */
