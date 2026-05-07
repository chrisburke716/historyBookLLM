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
  MessageResponse,
} from '../types';

export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'reset' }
  | { type: 'tool_start'; id: string; label: string }
  | { type: 'tool_end'; id: string; summary: string | null }
  | { type: 'done'; message: MessageResponse; session: SessionResponse }
  | { type: 'error'; message: string };

export interface StreamHandlers {
  onToken: (text: string) => void;
  onReset: () => void;
  onToolStart: (id: string, label: string) => void;
  onToolEnd: (id: string, summary: string | null) => void;
  onDone: (message: MessageResponse, session: SessionResponse) => void;
  onError: (message: string) => void;
}

class AgentAPI {
  private api: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
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

  async sendMessageStream(
    sessionId: string,
    request: MessageRequest,
    handlers: StreamHandlers,
  ): Promise<void> {
    let response: Response;
    try {
      response = await fetch(
        `${this.baseURL}/api/chat/sessions/${sessionId}/stream`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        },
      );
    } catch (e) {
      handlers.onError(`Network error: ${e}`);
      return;
    }

    if (!response.ok || !response.body) {
      handlers.onError(`HTTP ${response.status}`);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by blank lines (\n\n).
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? ''; // keep trailing partial frame

      for (const frame of frames) {
        const trimmed = frame.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const json = trimmed.slice(6);
        try {
          const event: StreamEvent = JSON.parse(json);
          if (event.type === 'token') handlers.onToken(event.text);
          else if (event.type === 'reset') handlers.onReset();
          else if (event.type === 'tool_start')
            handlers.onToolStart(event.id, event.label);
          else if (event.type === 'tool_end')
            handlers.onToolEnd(event.id, event.summary);
          else if (event.type === 'done')
            handlers.onDone(event.message, event.session);
          else if (event.type === 'error') handlers.onError(event.message);
        } catch (e) {
          console.error('Failed to parse SSE event:', json, e);
        }
      }
    }
  }
}

// Export singleton instance
export const agentAPI = new AgentAPI();
export default agentAPI;
