import React, { useMemo } from 'react';
import { Box, Divider } from '@mui/material';
import { useAppSelector } from '../store';
import {
  useGraphQuery,
  useSubgraphQuery,
} from '../hooks/useKGQueries';
import { GraphResponse } from '../types/kg';
import KGTopBar from '../components/kg/KGTopBar';
import ForceGraphPanel from '../components/kg/ForceGraphPanel';
import EntityPanel from '../components/kg/EntityPanel';

// react-force-graph-2d mutates link.source/target from string IDs to node objects
// during simulation. This extracts the string ID from either form.
function linkEndpointId(endpoint: unknown): string {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return (endpoint as { id: string }).id;
  }
  return endpoint as string;
}

function applyTrim(graph: GraphResponse, focusEntityId: string | null, threshold: number): GraphResponse {
  const keepIds = new Set(
    graph.nodes
      .filter((n) => n.occurrence_count >= threshold || n.id === focusEntityId)
      .map((n) => n.id)
  );
  return {
    ...graph,
    nodes: graph.nodes.filter((n) => keepIds.has(n.id)),
    links: graph.links.filter((l) => {
      const src = linkEndpointId(l.source);
      const tgt = linkEndpointId(l.target);
      return keepIds.has(src) && keepIds.has(tgt);
    }),
    node_count: keepIds.size,
  };
}

const KGPage: React.FC = () => {
  const { graphName, displayMode, hopCount, focusEntityId, occurrenceThreshold } = useAppSelector((s) => s.graph);

  const {
    data: fullGraph,
    isLoading: fullGraphLoading,
  } = useGraphQuery(graphName);

  const {
    data: subgraph,
    isLoading: subgraphLoading,
  } = useSubgraphQuery(
    focusEntityId,
    hopCount,
    graphName,
    displayMode === 'nhop'
  );

  const activeGraph = displayMode === 'nhop' && focusEntityId ? subgraph : fullGraph;
  const isLoading = displayMode === 'nhop' && focusEntityId ? subgraphLoading : fullGraphLoading;

  const displayGraph = useMemo(() => {
    if (!activeGraph || occurrenceThreshold <= 1) return activeGraph;
    return applyTrim(activeGraph, focusEntityId, occurrenceThreshold);
  }, [activeGraph, occurrenceThreshold, focusEntityId]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      <KGTopBar />
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <ForceGraphPanel graphData={displayGraph} isLoading={isLoading} />
        <Divider orientation="vertical" flexItem />
        <Box sx={{ width: 340, flexShrink: 0, overflow: 'hidden', borderLeft: 0 }}>
          <EntityPanel activeGraph={activeGraph} />
        </Box>
      </Box>
    </Box>
  );
};

export default KGPage;
