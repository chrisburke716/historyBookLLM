import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  IconButton,
  CircularProgress,
  Tooltip,
  Button,
} from '@mui/material';
import {
  ArrowForward as OutgoingIcon,
  ArrowBack as IncomingIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store';
import { setFocus, clearFocus } from '../../store/graphSlice';
import { useEntityQuery } from '../../hooks/useKGQueries';
import { GraphResponse, ENTITY_TYPE_COLORS } from '../../types/kg';

interface EntityPanelProps {
  activeGraph: GraphResponse | undefined;
}

const EntityPanel: React.FC<EntityPanelProps> = ({ activeGraph }) => {
  const dispatch = useAppDispatch();
  const { focusEntityId, colorMap } = useAppSelector((s) => s.graph);

  const { data: entity, isLoading } = useEntityQuery(focusEntityId);

  const typeColor = (type: string) =>
    colorMap[type] ?? ENTITY_TYPE_COLORS[type] ?? '#999';

  if (!focusEntityId) {
    // Unfocused state: show graph stats
    return (
      <Box sx={{ p: 2, height: '100%', overflow: 'auto' }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Graph Stats
        </Typography>
        {activeGraph ? (
          <>
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <Box>
                <Typography variant="h5" fontWeight="bold">
                  {activeGraph.node_count}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Entities
                </Typography>
              </Box>
              <Box>
                <Typography variant="h5" fontWeight="bold">
                  {activeGraph.edge_count}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Relationships
                </Typography>
              </Box>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="caption" color="text.secondary">
              Click a node to explore entity details. Use N-hop mode to focus on a
              subgraph around a selected entity.
            </Typography>
          </>
        ) : (
          <Typography variant="caption" color="text.secondary">
            Select a graph to begin.
          </Typography>
        )}
      </Box>
    );
  }

  if (isLoading || !entity) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <Box sx={{ p: 2, pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="h6" noWrap title={entity.name}>
              {entity.name}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
              <Chip
                label={entity.entity_type}
                size="small"
                sx={{
                  bgcolor: typeColor(entity.entity_type),
                  color: '#fff',
                  fontWeight: 'bold',
                }}
              />
              <Typography variant="caption" color="text.secondary">
                ×{entity.occurrence_count}
              </Typography>
            </Box>
          </Box>
          <Tooltip title="Clear focus">
            <IconButton size="small" onClick={() => dispatch(clearFocus())} sx={{ ml: 1 }}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {entity.aliases.length > 0 && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            Also known as: {entity.aliases.join(', ')}
          </Typography>
        )}
      </Box>

      <Divider />

      {/* Scrollable body */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2, pt: 1.5 }}>
        {entity.descriptions.length > 0 && (
          <>
            <Box component="ul" sx={{ m: 0, pl: 2.5, mb: 2 }}>
              {entity.descriptions.map((desc, i) => (
                <Typography key={i} component="li" variant="body2" sx={{ lineHeight: 1.6, mb: 0.5 }}>
                  {desc}
                </Typography>
              ))}
            </Box>
            <Divider sx={{ mb: 1.5 }} />
          </>
        )}

        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Relationships ({entity.relationships.length})
        </Typography>

        {entity.relationships.length === 0 ? (
          <Typography variant="caption" color="text.secondary">
            No relationships found.
          </Typography>
        ) : (
          <List dense disablePadding>
            {entity.relationships.map((rel) => (
              <ListItem
                key={rel.relationship_id}
                disablePadding
                sx={{ mb: 0.5, alignItems: 'flex-start' }}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, width: '100%' }}>
                  <Tooltip title={rel.direction}>
                    <Box sx={{ mt: 0.25, color: rel.direction === 'outgoing' ? 'primary.main' : 'text.secondary' }}>
                      {rel.direction === 'outgoing' ? (
                        <OutgoingIcon sx={{ fontSize: 16 }} />
                      ) : (
                        <IncomingIcon sx={{ fontSize: 16 }} />
                      )}
                    </Box>
                  </Tooltip>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                      <Chip
                        label={rel.relation_type}
                        size="small"
                        variant="outlined"
                        sx={{ height: 18, fontSize: '0.65rem' }}
                      />
                      <Button
                        size="small"
                        variant="text"
                        onClick={() => dispatch(setFocus(rel.other_entity_id))}
                        sx={{ p: 0, minWidth: 0, textTransform: 'none', fontSize: '0.8rem', fontWeight: 'bold' }}
                      >
                        {rel.other_entity_name}
                      </Button>
                      <Typography variant="caption" color="text.disabled">
                        b{rel.book_index}:ch{rel.chapter_index}
                      </Typography>
                    </Box>
                    {rel.description && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ display: 'block', mt: 0.25 }}
                      >
                        {rel.description}
                      </Typography>
                    )}
                  </Box>
                </Box>
              </ListItem>
            ))}
          </List>
        )}
      </Box>
    </Box>
  );
};

export default EntityPanel;
