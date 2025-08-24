/**
 * MessageList component displays conversation history with citations.
 */

import React from 'react';
import {
  List,
  ListItem,
  Paper,
  Typography,
  Box,
  Chip,
  Avatar,
} from '@mui/material';
import {
  Person as PersonIcon,
  SmartToy as BotIcon,
} from '@mui/icons-material';
import { MessageResponse } from '../types';

interface MessageListProps {
  messages: MessageResponse[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const formatMessage = (content: string): JSX.Element => {
    // Simple markdown-like formatting for bold text
    const formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    return <span dangerouslySetInnerHTML={{ __html: formatted }} />;
  };

  if (messages.length === 0) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        height="100%" 
        color="text.secondary"
      >
        <Typography variant="h6">
          Start a conversation by typing a message below
        </Typography>
      </Box>
    );
  }

  return (
    <List sx={{ width: '100%', p: 1 }}>
      {messages.map((message, index) => {
        const isUser = message.role === 'user';
        const isLast = index === messages.length - 1;

        return (
          <ListItem 
            key={message.id} 
            alignItems="flex-start"
            sx={{ 
              mb: 2,
              justifyContent: isUser ? 'flex-end' : 'flex-start',
            }}
          >
            <Box
              sx={{
                display: 'flex',
                flexDirection: isUser ? 'row-reverse' : 'row',
                alignItems: 'flex-start',
                maxWidth: '80%',
                width: 'fit-content',
              }}
            >
              {/* Avatar */}
              <Avatar
                sx={{
                  bgcolor: isUser ? 'primary.main' : 'secondary.main',
                  mx: 1,
                }}
              >
                {isUser ? <PersonIcon /> : <BotIcon />}
              </Avatar>

              {/* Message Content */}
              <Paper
                elevation={1}
                sx={{
                  p: 2,
                  bgcolor: isUser ? 'primary.light' : 'grey.100',
                  color: isUser ? 'primary.contrastText' : 'text.primary',
                  borderRadius: 2,
                }}
              >
                <Typography 
                  variant="body1" 
                  sx={{ 
                    mb: 1,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {formatMessage(message.content)}
                </Typography>

                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
                      Sources:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {message.citations.map((citation, idx) => (
                        <Chip
                          key={idx}
                          label={citation}
                          size="small"
                          variant="outlined"
                          sx={{ 
                            fontSize: '0.75rem',
                            height: 'auto',
                            color: isUser ? 'primary.contrastText' : 'text.secondary',
                            borderColor: isUser ? 'primary.contrastText' : 'grey.400',
                          }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}

                {/* Timestamp */}
                <Typography 
                  variant="caption" 
                  color={isUser ? 'primary.contrastText' : 'text.secondary'}
                  sx={{ 
                    display: 'block', 
                    textAlign: 'right', 
                    mt: 1,
                    opacity: 0.7,
                  }}
                >
                  {formatTimestamp(message.timestamp)}
                </Typography>
              </Paper>
            </Box>
          </ListItem>
        );
      })}
    </List>
  );
};

export default MessageList;