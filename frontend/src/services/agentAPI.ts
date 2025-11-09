/**
 * Agent API service for communicating with the LangGraph-based agent API.
 * Provides the same interface as ChatAPI for unified abstraction.
 */

import axios, { AxiosInstance } from 'axios';
import {
  SessionCreateRequest,
  MessageRequest,
  SessionResponse,
  SessionListResponse,
  MessageListResponse,
  ChatResponse,
} from '../types';

class AgentAPI {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 30000, // 30 second timeout for RAG responses
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Test if the API server is running
   */
  async healthCheck(): Promise<{ message: string }> {
    const response = await this.api.get('/');
    return response.data;
  }

  /**
   * Create a new agent session
   */
  async createSession(request: SessionCreateRequest): Promise<SessionResponse> {
    const response = await this.api.post('/api/agent/sessions', request);
    return response.data;
  }

  /**
   * Get list of recent agent sessions
   */
  async getSessions(limit: number = 10): Promise<SessionListResponse> {
    const response = await this.api.get('/api/agent/sessions', {
      params: { limit }
    });
    return response.data;
  }

  /**
   * Get all messages for an agent session
   */
  async getSessionMessages(sessionId: string): Promise<MessageListResponse> {
    const response = await this.api.get(`/api/agent/sessions/${sessionId}/messages`);
    return response.data;
  }

  /**
   * Send a message to the agent and get AI response
   */
  async sendMessage(sessionId: string, request: MessageRequest): Promise<ChatResponse> {
    // Agent API expects 'content' only (enable_retrieval and max_context_paragraphs not needed)
    // It always uses retrieval with configured defaults
    const agentRequest = {
      content: request.content,
      // Future: support agent-specific parameters like max_context_paragraphs, similarity_threshold
    };

    const response = await this.api.post(
      `/api/agent/sessions/${sessionId}/messages`,
      agentRequest
    );

    // Agent API returns AgentChatResponse which has same structure as ChatResponse
    return response.data;
  }
}

// Export singleton instance
export const agentAPI = new AgentAPI();
export default agentAPI;
