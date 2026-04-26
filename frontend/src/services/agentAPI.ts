/**
 * Chat API service for communicating with the LangGraph-based RAG agent.
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
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async healthCheck(): Promise<{ message: string }> {
    const response = await this.api.get('/');
    return response.data;
  }

  async createSession(request: SessionCreateRequest): Promise<SessionResponse> {
    const response = await this.api.post('/api/chat/sessions', request);
    return response.data;
  }

  async getSessions(limit: number = 10): Promise<SessionListResponse> {
    const response = await this.api.get('/api/chat/sessions', {
      params: { limit }
    });
    return response.data;
  }

  async getSessionMessages(sessionId: string): Promise<MessageListResponse> {
    const response = await this.api.get(`/api/chat/sessions/${sessionId}/messages`);
    return response.data;
  }

  async sendMessage(sessionId: string, request: MessageRequest): Promise<ChatResponse> {
    const response = await this.api.post(`/api/chat/sessions/${sessionId}/messages`, request);
    return response.data;
  }
}

// Export singleton instance
export const agentAPI = new AgentAPI();
export default agentAPI;
