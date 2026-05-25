/**
 * MessageInput — drives the CopilotKit thread directly.
 *
 * Calls `sendMessage` from `useCopilotChatInternal()`; no parent callback.
 * The agent run streams back into the message thread via the AG-UI
 * connection set up by `<CopilotProvider>`.
 */

import React, { useState, KeyboardEvent } from 'react';
import { Box, IconButton, Paper, TextField, Tooltip } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { useCoAgent, useCopilotChatInternal } from '@copilotkit/react-core';

interface Props {
  placeholder?: string;
}

const RUN_AGENT_NAME = 'rag';

const MessageInput: React.FC<Props> = ({ placeholder = 'Ask a question about history...' }) => {
  const [message, setMessage] = useState('');
  const { sendMessage } = useCopilotChatInternal();
  const { running } = useCoAgent({ name: RUN_AGENT_NAME });

  const disabled = running;

  const handleSend = async () => {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;
    setMessage('');
    await sendMessage({
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmed,
    } as any);
  };

  const handleKeyPress = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 2, borderRadius: 2, bgcolor: 'background.paper' }}>
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
          sx={{ '& .MuiOutlinedInput-root': { borderRadius: 3 } }}
        />

        <Tooltip title="Send message">
          <span>
            <IconButton
              onClick={handleSend}
              disabled={!message.trim() || disabled}
              color="primary"
              size="large"
              sx={{
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
                '&:hover': { bgcolor: 'primary.dark' },
                '&.Mui-disabled': { bgcolor: 'grey.300', color: 'grey.500' },
              }}
            >
              <SendIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {message.length > 0 && (
        <Box
          sx={{
            mt: 0.5,
            color: message.length > 500 ? 'warning.main' : 'text.secondary',
            fontSize: '0.75rem',
            textAlign: 'right',
          }}
        >
          {message.length} characters
        </Box>
      )}

      <Box sx={{ mt: 1, fontSize: '0.75rem', color: 'text.secondary' }}>
        💡 Press Enter to send • Shift+Enter for new line
      </Box>
    </Paper>
  );
};

export default MessageInput;
