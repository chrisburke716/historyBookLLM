/**
 * ChatPage — main chat surface.
 *
 * Layers:
 *  - useChat hook: session list + per-session history (Weaviate-backed)
 *  - <CopilotProvider>: AG-UI HttpAgent + CopilotKit runtime, scoped to chat
 *  - <ChatThreadController>: hydrates the thread with history, fires run-end
 *  - <MessageList>: renders live messages from CopilotKit hooks
 *  - <MessageInput>: sends new turns via CopilotKit hooks
 */

import React, { useEffect, useRef } from 'react';
import {
  Alert,
  Box,
  CircularProgress,
  Container,
  Paper,
  Snackbar,
} from '@mui/material';

import { useChat } from '../hooks/useChat';
import CopilotProvider from '../components/CopilotProvider';
import ChatThreadController from '../components/ChatThreadController';
import MessageList from '../components/MessageList';
import MessageInput from '../components/MessageInput';
import SessionDropdown from '../components/SessionDropdown';

const ChatPage: React.FC = () => {
  const {
    currentSession,
    sessions,
    historicalMessages,
    sessionsLoaded,
    isLoading,
    error,
    onRunEnd,
    createSession,
    switchToSession,
    clearError,
  } = useChat();

  const hasInitialized = useRef(false);

  // Pick an initial session once the session list has loaded:
  //   - if there are existing sessions, open the most recent one
  //   - otherwise create a new one
  // Gated on `sessionsLoaded` so we don't fire before the GET /sessions
  // request has actually resolved (otherwise we'd race the fetch and
  // always create a fresh session on every page load).
  useEffect(() => {
    if (!sessionsLoaded || currentSession || hasInitialized.current) return;
    hasInitialized.current = true;
    if (sessions.length > 0) {
      switchToSession(sessions[0]);
    } else {
      createSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionsLoaded, currentSession, sessions]);

  return (
    <Container
      maxWidth="lg"
      sx={{ py: 2, height: '100vh', display: 'flex', flexDirection: 'column' }}
    >
      <Box sx={{ mb: 2 }}>
        <SessionDropdown
          sessions={sessions}
          currentSession={currentSession}
          onSessionChange={switchToSession}
          onNewSession={createSession}
          disabled={isLoading}
        />
      </Box>

      <Paper
        elevation={2}
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          borderRadius: 2,
        }}
      >
        {currentSession ? (
          <CopilotProvider threadId={currentSession.id}>
            <ChatThreadController
              historicalMessages={historicalMessages}
              onRunEnd={onRunEnd}
            />

            <Box sx={{ flex: 1, overflow: 'auto', bgcolor: 'grey.50' }}>
              <MessageList />
            </Box>

            <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
              <MessageInput
                placeholder="Ask a question about history..."
              />
            </Box>
          </CopilotProvider>
        ) : (
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            height="100%"
          >
            <CircularProgress />
          </Box>
        )}
      </Paper>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={clearError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={clearError}
          severity="error"
          variant="filled"
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default ChatPage;
