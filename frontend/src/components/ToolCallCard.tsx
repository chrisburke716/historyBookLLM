/**
 * ToolCallCard — small MUI pill for an in-progress / completed tool call.
 * Live-only; not persisted. Status flips to "done" when the matching
 * ToolMessage arrives.
 */

import React from 'react';
import { Box, Chip, CircularProgress, Tooltip } from '@mui/material';
import { Check as CheckIcon } from '@mui/icons-material';

interface Props {
  toolName: string;
  argsPreview?: string;
  done: boolean;
  resultPreview?: string;
}

const ToolCallCard: React.FC<Props> = ({ toolName, argsPreview, done, resultPreview }) => {
  const label = argsPreview ? `${toolName} (${argsPreview})` : toolName;
  return (
    <Tooltip title={done && resultPreview ? resultPreview : ''} arrow placement="top">
      <Chip
        size="small"
        variant="outlined"
        icon={done ? <CheckIcon /> : <CircularProgress size={12} />}
        label={
          <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
            {label}
          </Box>
        }
        sx={{
          bgcolor: done ? 'success.50' : 'grey.50',
          borderColor: done ? 'success.light' : 'grey.300',
          maxWidth: '100%',
          '& .MuiChip-label': {
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          },
        }}
      />
      {/* Tooltip needs a single child, but Chip has no built-in icon-styling
          for both states — wrap in Box if needed in the future. */}
    </Tooltip>
  );
};

export default ToolCallCard;
