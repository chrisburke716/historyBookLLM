/**
 * Custom hook for managing chat session state.
 *
 * Owns session list + per-session historical messages (loaded from Weaviate).
 * Live message streaming and sending are handled by CopilotKit hooks inside
 * the chat surface — see `<CopilotProvider>` and `<MessageInput>`.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import {
  SessionResponse,
  MessageResponse,
  SessionCreateRequest,
} from '../types';

interface ChatState {
  currentSession: SessionResponse | null;
  sessions: SessionResponse[];
  historicalMessages: MessageResponse[];
  isLoading: boolean;
  error: string | null;
}

export const useChat = () => {
  const [state, setState] = useState<ChatState>({
    currentSession: null,
    sessions: [],
    historicalMessages: [],
    isLoading: false,
    error: null,
  });

  const setLoading = useCallback((loading: boolean) => {
    setState((prev) => ({ ...prev, isLoading: loading }));
  }, []);

  const setError = useCallback((error: string | null) => {
    setState((prev) => ({ ...prev, error }));
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      setError(null);
      const response = await api.getSessions();
      setState((prev) => ({ ...prev, sessions: response.sessions }));
    } catch (error) {
      setError(`Failed to load sessions: ${error}`);
    }
  }, [setError]);

  const createSession = useCallback(
    async (title?: string): Promise<SessionResponse | null> => {
      try {
        setLoading(true);
        setError(null);
        const request: SessionCreateRequest = title ? { title } : {};
        const session = await api.createSession(request);

        setState((prev) => ({
          ...prev,
          sessions: [session, ...prev.sessions],
          currentSession: session,
          historicalMessages: [],
        }));

        return session;
      } catch (error) {
        setError(`Failed to create session: ${error}`);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [setLoading, setError],
  );

  const switchToSession = useCallback(
    async (session: SessionResponse) => {
      try {
        setLoading(true);
        setError(null);
        const response = await api.getSessionMessages(session.id);
        setState((prev) => ({
          ...prev,
          currentSession: session,
          historicalMessages: response.messages,
        }));
      } catch (error) {
        setError(`Failed to load session messages: ${error}`);
      } finally {
        setLoading(false);
      }
    },
    [setLoading, setError],
  );

  const clearError = useCallback(() => setError(null), [setError]);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  return {
    ...state,
    loadSessions,
    createSession,
    switchToSession,
    clearError,
  };
};
