/**
 * Custom hook for managing chat session state.
 *
 * Server state (session list, per-session message history) is owned by
 * TanStack Query — same pattern as the KG page. Local UI state
 * (currentSession, displayed error) stays as useState. Live message
 * streaming is handled by CopilotKit hooks inside the chat surface;
 * this hook only touches the static session/history endpoints.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';

import { api } from '../services/api';
import {
  MessageListResponse,
  MessageResponse,
  SessionCreateRequest,
  SessionListResponse,
  SessionResponse,
} from '../types';

const SESSIONS_KEY = ['sessions'] as const;
const sessionMessagesKey = (id: string) =>
  ['session-messages', id] as const;

export const useChat = () => {
  const queryClient = useQueryClient();
  const [currentSession, setCurrentSession] = useState<SessionResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  const sessionsQuery = useQuery<SessionListResponse, Error, SessionResponse[]>(
    {
      queryKey: SESSIONS_KEY,
      queryFn: () => api.getSessions(),
      select: (data) => data.sessions,
    },
  );

  // Per-session history. Query key changes with currentSession.id; disabled
  // when there is no current session so the placeholder key is never fetched.
  const messagesQuery = useQuery<MessageListResponse, Error, MessageResponse[]>(
    {
      queryKey: sessionMessagesKey(currentSession?.id ?? ''),
      queryFn: () => api.getSessionMessages(currentSession!.id),
      select: (data) => data.messages,
      enabled: !!currentSession,
    },
  );

  // Surface query errors via local error state so the snackbar can be
  // dismissed independently of the underlying query state.
  useEffect(() => {
    if (sessionsQuery.error) {
      setError(`Failed to load sessions: ${sessionsQuery.error.message}`);
    }
  }, [sessionsQuery.error]);
  useEffect(() => {
    if (messagesQuery.error) {
      setError(`Failed to load messages: ${messagesQuery.error.message}`);
    }
  }, [messagesQuery.error]);

  const createMutation = useMutation({
    mutationFn: (title?: string) => {
      const req: SessionCreateRequest = title ? { title } : {};
      return api.createSession(req);
    },
    onSuccess: (session) => {
      // Optimistically prepend to the cached session list so the dropdown
      // reflects the new session before the next list refetch.
      queryClient.setQueryData<SessionListResponse>(SESSIONS_KEY, (old) => ({
        sessions: [session, ...(old?.sessions ?? [])],
      }));
      // Seed an empty history so the messages query for this new session
      // hits the cache instead of round-tripping for [].
      queryClient.setQueryData<MessageListResponse>(
        sessionMessagesKey(session.id),
        { messages: [] },
      );
      setCurrentSession(session);
    },
    onError: (err: Error) =>
      setError(`Failed to create session: ${err.message}`),
  });

  const createSession = useCallback(
    async (title?: string): Promise<SessionResponse | null> => {
      try {
        return await createMutation.mutateAsync(title);
      } catch {
        return null;
      }
    },
    [createMutation],
  );

  const switchToSession = useCallback(
    async (session: SessionResponse) => {
      try {
        // Pre-fetch so historicalMessages is ready by the time
        // ChatThreadController mounts and hydrates CopilotKit.
        await queryClient.fetchQuery({
          queryKey: sessionMessagesKey(session.id),
          queryFn: () => api.getSessionMessages(session.id),
        });
        setCurrentSession(session);
      } catch (err) {
        setError(`Failed to load session: ${err}`);
      }
    },
    [queryClient],
  );

  // Called after a turn finishes. Invalidates the session list (titles may
  // have regenerated) and the active session's messages (Weaviate now has
  // the new user/assistant pair, so re-entering this session later should
  // re-fetch fresh history).
  const onRunEnd = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: SESSIONS_KEY });
    if (currentSession) {
      queryClient.invalidateQueries({
        queryKey: sessionMessagesKey(currentSession.id),
      });
    }
  }, [queryClient, currentSession]);

  const clearError = useCallback(() => setError(null), []);

  return {
    currentSession,
    sessions: sessionsQuery.data ?? [],
    historicalMessages: messagesQuery.data ?? [],
    sessionsLoaded: sessionsQuery.isFetched,
    isLoading: createMutation.isPending || messagesQuery.isFetching,
    error,
    onRunEnd,
    createSession,
    switchToSession,
    clearError,
  };
};
