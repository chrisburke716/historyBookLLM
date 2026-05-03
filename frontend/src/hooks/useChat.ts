/**
 * Custom hook for managing chat state and API interactions.
 */

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import {
  SessionResponse,
  MessageResponse,
  ChatState,
  SessionCreateRequest,
} from '../types';

export const useChat = () => {
  const [state, setState] = useState<ChatState>({
    currentSession: null,
    sessions: [],
    messages: [],
    isLoading: false,
    error: null,
  });

  /**
   * Set loading state
   */
  const setLoading = useCallback((loading: boolean) => {
    setState(prev => ({ ...prev, isLoading: loading }));
  }, []);

  /**
   * Set error message
   */
  const setError = useCallback((error: string | null) => {
    setState(prev => ({ ...prev, error }));
  }, []);

  /**
   * Load list of sessions
   */
  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.getSessions();
      setState(prev => ({ ...prev, sessions: response.sessions }));
    } catch (error) {
      setError(`Failed to load sessions: ${error}`);
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError]);

  /**
   * Create a new session
   */
  const createSession = useCallback(async (title?: string): Promise<SessionResponse | null> => {
    try {
      setLoading(true);
      setError(null);
      const request: SessionCreateRequest = title ? { title } : {};
      const session = await api.createSession(request);

      // Add to sessions list
      setState(prev => ({
        ...prev,
        sessions: [session, ...prev.sessions],
        currentSession: session,
        messages: [], // Clear messages for new session
      }));

      return session;
    } catch (error) {
      setError(`Failed to create session: ${error}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError]);

  /**
   * Switch to a different session
   */
  const switchToSession = useCallback(async (session: SessionResponse) => {
    try {
      setLoading(true);
      setError(null);

      // Load messages for the session
      const response = await api.getSessionMessages(session.id);

      setState(prev => ({
        ...prev,
        currentSession: session,
        messages: response.messages,
      }));
    } catch (error) {
      setError(`Failed to load session messages: ${error}`);
    } finally {
      setLoading(false);
    }
  }, [setLoading, setError]);

  /**
   * Send a message in the current session (token-streaming).
   *
   * Adds an optimistic user message and an empty in-progress assistant
   * message, appends text on each token event, finalizes both on `done`.
   */
  const sendMessage = useCallback(async (content: string): Promise<boolean> => {
    if (!state.currentSession) {
      setError('No active session');
      return false;
    }

    const sessionId = state.currentSession.id;
    setLoading(true);
    setError(null);

    const userId = `temp-user-${Date.now()}`;
    const inProgressId = `streaming-${Date.now()}`;

    const userMessage: MessageResponse = {
      id: userId,
      content,
      role: 'user',
      timestamp: new Date().toISOString(),
      session_id: sessionId,
    };
    const inProgressMessage: MessageResponse = {
      id: inProgressId,
      content: '',
      role: 'assistant',
      timestamp: new Date().toISOString(),
      session_id: sessionId,
    };

    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage, inProgressMessage],
    }));

    return new Promise<boolean>((resolve) => {
      api
        .sendMessageStream(
          sessionId,
          { content },
          {
            onToken: (text) => {
              setState(prev => ({
                ...prev,
                messages: prev.messages.map(m =>
                  m.id === inProgressId
                    ? { ...m, content: m.content + text }
                    : m,
                ),
              }));
            },
            onReset: () => {
              // Tool boundary — drop preamble text, only the final answer
              // should remain in the in-progress message.
              setState(prev => ({
                ...prev,
                messages: prev.messages.map(m =>
                  m.id === inProgressId ? { ...m, content: '' } : m,
                ),
              }));
            },
            onDone: (message, session) => {
              setState(prev => {
                const updatedSessions = prev.sessions.map(s =>
                  s.id === session.id ? session : s,
                );
                return {
                  ...prev,
                  currentSession: session,
                  sessions: updatedSessions,
                  messages: prev.messages.map(m =>
                    m.id === inProgressId ? message : m,
                  ),
                  isLoading: false,
                };
              });
              resolve(true);
            },
            onError: (msg) => {
              setError(`Failed to send message: ${msg}`);
              setState(prev => ({
                ...prev,
                messages: prev.messages.filter(
                  m => m.id !== inProgressId && m.id !== userId,
                ),
                isLoading: false,
              }));
              resolve(false);
            },
          },
        )
        .catch((e) => {
          setError(`Failed to send message: ${e}`);
          setState(prev => ({
            ...prev,
            messages: prev.messages.filter(
              m => m.id !== inProgressId && m.id !== userId,
            ),
            isLoading: false,
          }));
          resolve(false);
        });
    });
  }, [state.currentSession, setLoading, setError]);

  /**
   * Clear current error
   */
  const clearError = useCallback(() => {
    setError(null);
  }, [setError]);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  return {
    ...state,
    // Actions
    loadSessions,
    createSession,
    switchToSession,
    sendMessage,
    clearError,
  };
};