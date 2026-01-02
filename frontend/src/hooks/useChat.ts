/**
 * Custom hook for managing chat state and API interactions.
 *
 * Uses the unified API abstraction which switches between Chat and Agent backends
 * based on the REACT_APP_USE_AGENT_API environment variable.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../services/api';
import {
  SessionResponse,
  MessageResponse,
  ChatState,
  SessionCreateRequest,
  MessageRequest,
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
   * Send a message in the current session
   */
  const sendMessage = useCallback(async (content: string, enableRetrieval: boolean = true): Promise<boolean> => {
    if (!state.currentSession) {
      setError('No active session');
      return false;
    }

    try {
      setLoading(true);
      setError(null);
      
      const request: MessageRequest = {
        content,
        enable_retrieval: enableRetrieval,
      };

      // Add user message optimistically
      const userMessage: MessageResponse = {
        id: `temp-${Date.now()}`,
        content,
        role: 'user',
        timestamp: new Date().toISOString(),
        session_id: state.currentSession.id,
      };

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Send to API and get response
      const response = await api.sendMessage(state.currentSession.id, request);

      // Replace temp message with real user message and add AI response
      // Also update currentSession and sessions list with new title
      setState(prev => {
        const messages = prev.messages.slice(0, -1); // Remove temp message

        // Update sessions list to reflect new title
        const updatedSessions = prev.sessions.map(s =>
          s.id === response.session.id ? response.session : s
        );

        return {
          ...prev,
          currentSession: response.session,  // Update with new title
          sessions: updatedSessions,         // Update sessions list
          messages: [...messages, ...prev.messages.slice(-1), response.message],
        };
      });

      return true;
    } catch (error) {
      setError(`Failed to send message: ${error}`);
      // Remove the optimistic user message on error
      setState(prev => ({
        ...prev,
        messages: prev.messages.slice(0, -1),
      }));
      return false;
    } finally {
      setLoading(false);
    }
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // Only run once on mount

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