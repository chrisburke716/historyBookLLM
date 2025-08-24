/**
 * MessageInput component for sending messages.
 */

import React, { useState, KeyboardEvent } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Paper,
  FormControlLabel,
  Switch,
  Tooltip,
} from '@mui/material';
import {
  Send as SendIcon,
  Search as SearchIcon,
} from '@mui/icons-material';

interface MessageInputProps {
  onSendMessage: (message: string, enableRetrieval: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = "Ask a question about history...",
}) => {
  const [message, setMessage] = useState('');
  const [enableRetrieval, setEnableRetrieval] = useState(true);

  const handleSend = () => {
    const trimmed = message.trim();
    if (trimmed && !disabled) {
      onSendMessage(trimmed, enableRetrieval);
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

      {/* RAG Toggle */}
      <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <FormControlLabel
          control={
            <Switch
              checked={enableRetrieval}
              onChange={(e) => setEnableRetrieval(e.target.checked)}
              color="primary"
              size="small"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <SearchIcon fontSize="small" color={enableRetrieval ? 'primary' : 'disabled'} />
              <span style={{ fontSize: '0.875rem' }}>
                Search documents
              </span>
            </Box>
          }
          sx={{ 
            m: 0,
            '& .MuiFormControlLabel-label': {
              color: enableRetrieval ? 'text.primary' : 'text.secondary',
            },
          }}
        />
        
        {/* Character count hint */}
        {message.length > 0 && (
          <Box sx={{ 
            color: message.length > 500 ? 'warning.main' : 'text.secondary',
            fontSize: '0.75rem',
          }}>
            {message.length} characters
          </Box>
        )}
      </Box>
      
      {/* Hints */}
      <Box sx={{ mt: 1, fontSize: '0.75rem', color: 'text.secondary' }}>
        💡 Press Enter to send • Shift+Enter for new line • Toggle search to use RAG
      </Box>
    </Paper>
  );
};

export default MessageInput;