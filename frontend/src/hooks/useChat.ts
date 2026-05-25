/**
 * Custom hook for managing chat session state.
 *
 * Server state (session list, per-session message history, create + switch
 * operations) is owned by TanStack Query. Errors surface via derived state:
 * `error` is computed during render from the queries/mutations, and a
 * `dismissed` ref tracks the most recently dismissed Error so the snackbar
 * can be closed without resetting the underlying query state.
 *
 * Local UI state (currentSession, dismissed-error reference) stays as
 * useState. Live message streaming is handled by CopilotKit hooks inside
 * the chat surface; this hook only touches the static endpoints.
 */

import { useCallback, useState } from 'react';
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

interface ErrorSource {
  error: Error;
  prefix: string;
}

export const useChat = () => {
  const queryClient = useQueryClient();
  const [currentSession, setCurrentSession] = useState<SessionResponse | null>(
    null,
  );
  // Stores the Error reference the user most recently dismissed. `error`
  // derives to null while liveError equals this; a new failure produces a
  // new Error reference and the snackbar re-appears.
  const [dismissed, setDismissed] = useState<Error | null>(null);

  const sessionsQuery = useQuery<SessionListResponse, Error, SessionResponse[]>(
    {
      queryKey: SESSIONS_KEY,
      queryFn: () => api.getSessions(),
      select: (data) => data.sessions,
    },
  );

  const messagesQuery = useQuery<MessageListResponse, Error, MessageResponse[]>(
    {
      queryKey: sessionMessagesKey(currentSession?.id ?? ''),
      queryFn: () => api.getSessionMessages(currentSession!.id),
      select: (data) => data.messages,
      enabled: !!currentSession,
    },
  );

  const createMutation = useMutation({
    mutationFn: (title?: string) => {
      const req: SessionCreateRequest = title ? { title } : {};
      return api.createSession(req);
    },
    onSuccess: (session) => {
      // Optimistically prepend to the cached session list.
      queryClient.setQueryData<SessionListResponse>(SESSIONS_KEY, (old) => ({
        sessions: [session, ...(old?.sessions ?? [])],
      }));
      // Seed empty history so the new session's messages query hits the
      // cache instead of round-tripping for [].
      queryClient.setQueryData<MessageListResponse>(
        sessionMessagesKey(session.id),
        { messages: [] },
      );
      setCurrentSession(session);
    },
  });

  const switchMutation = useMutation({
    mutationFn: async (session: SessionResponse) => {
      // Pre-fetch so historicalMessages is ready when
      // ChatThreadController mounts and hydrates CopilotKit.
      await queryClient.fetchQuery({
        queryKey: sessionMessagesKey(session.id),
        queryFn: () => api.getSessionMessages(session.id),
      });
      return session;
    },
    onSuccess: (session) => setCurrentSession(session),
  });

  // Derived error — first non-null source, unless it matches `dismissed`.
  const liveError: ErrorSource | null = sessionsQuery.error
    ? { error: sessionsQuery.error, prefix: 'Failed to load sessions' }
    : messagesQuery.error
      ? { error: messagesQuery.error, prefix: 'Failed to load messages' }
      : createMutation.error
        ? { error: createMutation.error, prefix: 'Failed to create session' }
        : switchMutation.error
          ? { error: switchMutation.error, prefix: 'Failed to switch session' }
          : null;
  const error =
    liveError && liveError.error !== dismissed
      ? `${liveError.prefix}: ${liveError.error.message}`
      : null;

  const createSession = useCallback(
    (title?: string) => createMutation.mutate(title),
    [createMutation],
  );

  const switchToSession = useCallback(
    (session: SessionResponse) => switchMutation.mutate(session),
    [switchMutation],
  );

  const clearError = useCallback(() => {
    if (liveError) setDismissed(liveError.error);
  }, [liveError]);

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

  return {
    currentSession,
    sessions: sessionsQuery.data ?? [],
    historicalMessages: messagesQuery.data ?? [],
    sessionsLoaded: sessionsQuery.isFetched,
    isLoading: createMutation.isPending || switchMutation.isPending,
    error,
    onRunEnd,
    createSession,
    switchToSession,
    clearError,
  };
};
