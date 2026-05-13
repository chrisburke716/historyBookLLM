/**
 * ToolCallCard — compact MUI Chip representing a single tool call.
 *
 * No spinner / running state. Tool execution is brief and the transient
 * "running" flicker added visual noise; the chip always renders in a
 * resolved style. Result preview is shown on hover via tooltip when present.
 */

import React from 'react';
import { Box, Chip, Tooltip } from '@mui/material';
import { Check as CheckIcon } from '@mui/icons-material';

interface Props {
  toolName: string;
  argsPreview?: string;
  resultPreview?: string;
}

const ToolCallCard: React.FC<Props> = ({ toolName, argsPreview, resultPreview }) => {
  const label = argsPreview ? `${toolName} (${argsPreview})` : toolName;
  return (
    <Tooltip title={resultPreview ?? ''} arrow placement="top">
      <Chip
        size="small"
        variant="outlined"
        icon={<CheckIcon />}
        label={
          <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
            {label}
          </Box>
        }
        sx={{
          bgcolor: 'success.50',
          borderColor: 'success.light',
          maxWidth: '100%',
          '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' },
        }}
      />
    </Tooltip>
  );
};

export default ToolCallCard;
