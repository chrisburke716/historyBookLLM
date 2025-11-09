/**
 * Centralized API service for communicating with the History Book Chat/Agent API.
 *
 * This file provides a unified abstraction over two backend implementations:
 * - ChatAPI: Legacy LCEL-based RAG (/api/chat/*)
 * - AgentAPI: LangGraph-based RAG (/api/agent/*)
 *
 * The active backend is controlled by the REACT_APP_USE_AGENT_API environment variable.
 * Default: Agent API (LangGraph)
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

class ChatAPI {
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
   * Create a new chat session
   */
  async createSession(request: SessionCreateRequest): Promise<SessionResponse> {
    const response = await this.api.post('/api/chat/sessions', request);
    return response.data;
  }

  /**
   * Get list of recent sessions
   */
  async getSessions(limit: number = 10): Promise<SessionListResponse> {
    const response = await this.api.get('/api/chat/sessions', {
      params: { limit }
    });
    return response.data;
  }

  /**
   * Get all messages for a session
   */
  async getSessionMessages(sessionId: string): Promise<MessageListResponse> {
    const response = await this.api.get(`/api/chat/sessions/${sessionId}/messages`);
    return response.data;
  }

  /**
   * Send a message and get AI response
   */
  async sendMessage(sessionId: string, request: MessageRequest): Promise<ChatResponse> {
    const response = await this.api.post(`/api/chat/sessions/${sessionId}/messages`, request);
    return response.data;
  }
}

// Legacy export for backward compatibility
export const chatAPI = new ChatAPI();

// Import AgentAPI
export { agentAPI } from './agentAPI';

// Unified API instance - switches based on environment variable
// Default to Agent API (LangGraph) for better performance
const USE_AGENT_API = process.env.REACT_APP_USE_AGENT_API !== 'false';

export const api = USE_AGENT_API ? require('./agentAPI').agentAPI : chatAPI;

// Default export is the unified API
export default api;