/**
 * ToolStep — small Perplexity-style pill showing a single tool call's
 * status (running/done) and an optional result summary.
 *
 * These are live-only: they live alongside the in-progress assistant
 * message during streaming and disappear when the message finalizes.
 */

import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { CheckCircle as CheckIcon } from '@mui/icons-material';
import { ToolStep as ToolStepData } from '../types';

interface ToolStepProps {
  step: ToolStepData;
}

const ToolStep: React.FC<ToolStepProps> = ({ step }) => {
  const isDone = step.status === 'done';
  return (
    <Box
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 1,
        px: 1.25,
        py: 0.5,
        mb: 0.5,
        borderRadius: 999,
        bgcolor: isDone ? 'grey.100' : 'grey.200',
        color: 'text.secondary',
        fontSize: '0.8125rem',
        maxWidth: '100%',
      }}
    >
      {isDone ? (
        <CheckIcon sx={{ fontSize: 16, color: 'success.main' }} />
      ) : (
        <CircularProgress size={12} thickness={5} />
      )}
      <Typography variant="caption" sx={{ fontSize: 'inherit' }}>
        {step.label}
        {step.summary ? (
          <Box component="span" sx={{ opacity: 0.7, ml: 0.5 }}>
            — {step.summary}
          </Box>
        ) : null}
      </Typography>
    </Box>
  );
};

export default ToolStep;
