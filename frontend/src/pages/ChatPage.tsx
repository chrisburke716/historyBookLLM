/**
 * ChatPage - Main chat interface component.
 */

import React, { useEffect, useRef } from 'react';
import {
  Container,
  Box,
  Paper,
  CircularProgress,
  Alert,
  Snackbar,
} from '@mui/material';
import { useChat } from '../hooks/useChat';
import MessageList from '../components/MessageList';
import MessageInput from '../components/MessageInput';
import SessionDropdown from '../components/SessionDropdown';

const ChatPage: React.FC = () => {
  const {
    currentSession,
    sessions,
    messages,
    isLoading,
    error,
    createSession,
    switchToSession,
    sendMessage,
    clearError,
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const hasCreatedInitialSession = useRef(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Create initial session if none exists
  // Use ref to prevent duplicate creation in React StrictMode (which runs effects twice)
  useEffect(() => {
    if (!currentSession && sessions.length === 0 && !isLoading && !hasCreatedInitialSession.current) {
      hasCreatedInitialSession.current = true;
      createSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSession, sessions.length, isLoading]);

  const handleSendMessage = async (content: string, enableRetrieval: boolean) => {
    if (!currentSession) {
      // Create a new session if none exists
      const newSession = await createSession();
      if (!newSession) return;
    }
    
    await sendMessage(content, enableRetrieval);
  };

  const handleNewSession = async () => {
    await createSession();
  };

  const handleSessionChange = async (session: any) => {
    await switchToSession(session);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 2, height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header with Session Dropdown */}
      <Box sx={{ mb: 2 }}>
        <SessionDropdown
          sessions={sessions}
          currentSession={currentSession}
          onSessionChange={handleSessionChange}
          onNewSession={handleNewSession}
          disabled={isLoading}
        />
      </Box>

      {/* Main Chat Area */}
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
        {/* Messages Area */}
        <Box 
          sx={{ 
            flex: 1, 
            overflow: 'auto',
            bgcolor: 'grey.50',
          }}
        >
          {isLoading && messages.length === 0 ? (
            <Box 
              display="flex" 
              justifyContent="center" 
              alignItems="center" 
              height="100%"
            >
              <CircularProgress />
            </Box>
          ) : (
            <>
              <MessageList messages={messages} />
              <div ref={messagesEndRef} />
            </>
          )}
        </Box>

        {/* Input Area */}
        <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
          <MessageInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            placeholder={
              !currentSession 
                ? "Creating session..." 
                : "Ask a question about history..."
            }
          />
          
          {/* Loading indicator for message sending */}
          {isLoading && messages.length > 0 && (
            <Box 
              sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center',
                mt: 1,
                gap: 1,
              }}
            >
              <CircularProgress size={16} />
              <span style={{ fontSize: '0.875rem', color: '#666' }}>
                Generating response...
              </span>
            </Box>
          )}
        </Box>
      </Paper>

      {/* Error Snackbar */}
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