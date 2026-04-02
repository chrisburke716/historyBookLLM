import React from 'react';
import { Box, Divider } from '@mui/material';
import { useAppSelector } from '../store';
import {
  useGraphQuery,
  useSubgraphQuery,
} from '../hooks/useKGQueries';
import KGTopBar from '../components/kg/KGTopBar';
import ForceGraphPanel from '../components/kg/ForceGraphPanel';
import EntityPanel from '../components/kg/EntityPanel';

const KGPage: React.FC = () => {
  const { graphName, displayMode, hopCount, focusEntityId } = useAppSelector((s) => s.graph);

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

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      <KGTopBar />
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <ForceGraphPanel graphData={activeGraph} isLoading={isLoading} />
        <Divider orientation="vertical" flexItem />
        <Box sx={{ width: 340, flexShrink: 0, overflow: 'hidden', borderLeft: 0 }}>
          <EntityPanel activeGraph={activeGraph} />
        </Box>
      </Box>
    </Box>
  );
};

export default KGPage;
