/**
 * Centralized API service for communicating with the History Book Chat API.
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

// Export singleton instance
export const chatAPI = new ChatAPI();
export default chatAPI;