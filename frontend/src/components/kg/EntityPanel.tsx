import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Divider,
  List,
  ListItem,
  IconButton,
  CircularProgress,
  Tooltip,
  Button,
  Skeleton,
} from '@mui/material';
import {
  ArrowForward as OutgoingIcon,
  ArrowBack as IncomingIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../store';
import { setFocus, clearFocus } from '../../store/graphSlice';
import { useEntityQuery } from '../../hooks/useKGQueries';
import { GraphMetricsResponse, GraphResponse, ENTITY_TYPE_COLORS } from '../../types/kg';

interface EntityPanelProps {
  activeGraph: GraphResponse | undefined;
  graphMetrics?: GraphMetricsResponse;
  graphMetricsLoading?: boolean;
}

const EntityPanel: React.FC<EntityPanelProps> = ({ activeGraph, graphMetrics, graphMetricsLoading }) => {
  const dispatch = useAppDispatch();
  const { focusEntityId, colorMap } = useAppSelector((s) => s.graph);

  const { data: entity, isLoading } = useEntityQuery(focusEntityId);

  const typeColor = (type: string) =>
    colorMap[type] ?? ENTITY_TYPE_COLORS[type] ?? '#999';

  if (!focusEntityId) {
    // Unfocused state: show graph stats + network metrics
    const showMetrics = activeGraph != null;
    const isComputingMetrics = graphMetricsLoading || graphMetrics?.status === 'computing';

    const MetricRow: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.4 }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="caption" fontWeight="bold">
          {isComputingMetrics ? <Skeleton width={48} /> : value}
        </Typography>
      </Box>
    );

    return (
      <Box sx={{ p: 2, height: '100%', overflow: 'auto' }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Graph Metrics
        </Typography>
        {!showMetrics ? (
          <Typography variant="caption" color="text.secondary">
            Select a graph to begin.
          </Typography>
        ) : (
          <>
            {/* Counts — always available from graph data */}
            <Box sx={{ display: 'flex', gap: 3, mb: 1.5 }}>
              <Box>
                <Typography variant="h5" fontWeight="bold">{activeGraph.node_count.toLocaleString()}</Typography>
                <Typography variant="caption" color="text.secondary">Entities</Typography>
              </Box>
              <Box>
                <Typography variant="h5" fontWeight="bold">{activeGraph.edge_count.toLocaleString()}</Typography>
                <Typography variant="caption" color="text.secondary">Relationships</Typography>
              </Box>
            </Box>
            <Divider sx={{ mb: 1 }} />

            {/* Network metrics */}
            <MetricRow
              label="Density"
              value={graphMetrics ? graphMetrics.density.toFixed(4) : '—'}
            />
            <MetricRow
              label="Giant component"
              value={graphMetrics ? `${(graphMetrics.giant_component_ratio * 100).toFixed(1)}%` : '—'}
            />
            <MetricRow
              label="Components"
              value={graphMetrics?.num_connected_components ?? '—'}
            />
            <MetricRow
              label="Avg path length"
              value={
                graphMetrics
                  ? (graphMetrics.avg_shortest_path_length != null
                    ? graphMetrics.avg_shortest_path_length.toFixed(2)
                    : 'too large')
                  : '—'
              }
            />
            <MetricRow
              label="Diameter"
              value={
                graphMetrics
                  ? (graphMetrics.diameter != null ? graphMetrics.diameter : 'too large')
                  : '—'
              }
            />
            <MetricRow
              label="Clustering coeff."
              value={graphMetrics ? graphMetrics.global_clustering_coefficient.toFixed(3) : '—'}
            />
            <MetricRow
              label="Communities (Louvain)"
              value={graphMetrics?.num_communities ?? '—'}
            />
            <MetricRow
              label="Articulation points"
              value={graphMetrics?.articulation_point_count ?? '—'}
            />

            {isComputingMetrics && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                <CircularProgress size={12} />
                <Typography variant="caption" color="text.secondary">Computing…</Typography>
              </Box>
            )}

            <Divider sx={{ my: 1.5 }} />
            <Typography variant="caption" color="text.secondary">
              Click a node to explore entity details.
            </Typography>
          </>
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
