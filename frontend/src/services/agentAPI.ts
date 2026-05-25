/**
 * REST client for session + history endpoints.
 *
 * Live chat turns go through the AG-UI endpoint via CopilotKit's HttpAgent
 * (see components/CopilotProvider.tsx); this module only covers the
 * supporting REST surface.
 */

import axios, { AxiosInstance } from 'axios';
import {
  MessageListResponse,
  SessionCreateRequest,
  SessionListResponse,
  SessionResponse,
} from '../types';

class AgentAPI {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 60000,
      headers: { 'Content-Type': 'application/json' },
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
      params: { limit },
    });
    return response.data;
  }

  async getSessionMessages(sessionId: string): Promise<MessageListResponse> {
    const response = await this.api.get(
      `/api/chat/sessions/${sessionId}/messages`,
    );
    return response.data;
  }
}

export const agentAPI = new AgentAPI();
export default agentAPI;
