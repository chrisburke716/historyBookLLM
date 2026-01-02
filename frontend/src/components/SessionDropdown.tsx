/**
 * SessionDropdown component for selecting and managing chat sessions.
 */

import React, { useState, memo } from 'react';
import {
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  IconButton,
  Tooltip,
  Typography,
  Divider,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Add as AddIcon,
  Chat as ChatIcon,
  Schedule as RecentIcon,
} from '@mui/icons-material';
import { SelectChangeEvent } from '@mui/material/Select';
import { SessionResponse } from '../types';

interface SessionDropdownProps {
  sessions: SessionResponse[];
  currentSession: SessionResponse | null;
  onSessionChange: (session: SessionResponse) => void;
  onNewSession: () => void;
  disabled?: boolean;
}

const SessionDropdown: React.FC<SessionDropdownProps> = ({
  sessions,
  currentSession,
  onSessionChange,
  onNewSession,
  disabled = false,
}) => {
  const [isCreating, setIsCreating] = useState(false);

  const handleChange = (event: SelectChangeEvent) => {
    const sessionId = event.target.value;
    
    if (sessionId === 'new') {
      handleNewSession();
      return;
    }
    
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      onSessionChange(session);
    }
  };

  const handleNewSession = async () => {
    setIsCreating(true);
    try {
      await onNewSession();
    } finally {
      setIsCreating(false);
    }
  };

  const formatSessionTitle = (session: SessionResponse): string => {
    if (session.title) {
      return session.title.length > 30 
        ? session.title.substring(0, 30) + '...'
        : session.title;
    }
    
    // Create title from creation date
    const date = new Date(session.created_at);
    return `Chat ${date.toLocaleDateString()} ${date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })}`;
  };

  const formatRelativeTime = (timestamp: string): string => {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 300 }}>
      <FormControl fullWidth size="small">
        <InputLabel id="session-select-label">Chat Session</InputLabel>
        <Select
          labelId="session-select-label"
          value={currentSession?.id || ''}
          label="Chat Session"
          onChange={handleChange}
          disabled={disabled}
        >
          {/* New Session Option */}
          <MenuItem value="new" disabled={isCreating}>
            <ListItemIcon>
              <AddIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText 
              primary={isCreating ? "Creating new session..." : "New Chat Session"} 
            />
          </MenuItem>
          
          {sessions.length > 0 && <Divider />}
          
          {/* Recent Sessions */}
          {sessions.map((session) => (
            <MenuItem key={session.id} value={session.id}>
              <ListItemIcon>
                <ChatIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary={formatSessionTitle(session)}
                secondary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <RecentIcon sx={{ fontSize: 12 }} />
                    <Typography variant="caption">
                      {formatRelativeTime(session.updated_at)}
                    </Typography>
                  </Box>
                }
              />
            </MenuItem>
          ))}
          
          {sessions.length === 0 && (
            <MenuItem disabled>
              <ListItemText 
                primary="No previous sessions"
                secondary="Create a new session to get started"
              />
            </MenuItem>
          )}
        </Select>
      </FormControl>

      <Tooltip title="Create new session">
        <IconButton 
          onClick={handleNewSession}
          disabled={disabled || isCreating}
          color="primary"
          sx={{ ml: 1 }}
        >
          <AddIcon />
        </IconButton>
      </Tooltip>
    </Box>
  );
};

export default memo(SessionDropdown, (prevProps, nextProps) => {
  // Custom equality: only re-render if sessions IDs/titles changed
  return (
    prevProps.currentSession?.id === nextProps.currentSession?.id &&
    prevProps.sessions.length === nextProps.sessions.length &&
    prevProps.sessions.every((s, i) =>
      s.id === nextProps.sessions[i]?.id &&
      s.title === nextProps.sessions[i]?.title
    )
  );
});