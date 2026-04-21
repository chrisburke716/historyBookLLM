/**
 * MessageInput component for sending messages.
 */

import React, { useState, KeyboardEvent, memo } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Paper,
  Tooltip,
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = "Ask a question about history...",
}) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    const trimmed = message.trim();
    if (trimmed && !disabled) {
      onSendMessage(trimmed);
      setMessage('');
    }
  };

  const handleKeyPress = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        borderRadius: 2,
        bgcolor: 'background.paper',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1 }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          disabled={disabled}
          variant="outlined"
          size="small"
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
            },
          }}
        />

        <Tooltip title="Send message">
          <IconButton
            onClick={handleSend}
            disabled={!message.trim() || disabled}
            color="primary"
            size="large"
            sx={{
              bgcolor: 'primary.main',
              color: 'primary.contrastText',
              '&:hover': {
                bgcolor: 'primary.dark',
              },
              '&.Mui-disabled': {
                bgcolor: 'grey.300',
                color: 'grey.500',
              },
            }}
          >
            <SendIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {message.length > 0 && (
        <Box sx={{
          mt: 0.5,
          color: message.length > 500 ? 'warning.main' : 'text.secondary',
          fontSize: '0.75rem',
          textAlign: 'right',
        }}>
          {message.length} characters
        </Box>
      )}

      <Box sx={{ mt: 1, fontSize: '0.75rem', color: 'text.secondary' }}>
        💡 Press Enter to send • Shift+Enter for new line
      </Box>
    </Paper>
  );
};

export default memo(MessageInput);
