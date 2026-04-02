import React, { useCallback, useEffect, useRef } from 'react';
import ForceGraph2D, { ForceGraphMethods } from 'react-force-graph-2d';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useAppDispatch, useAppSelector } from '../../store';
import { setFocus, clearFocus } from '../../store/graphSlice';
import { GraphResponse, GraphNode, ENTITY_TYPE_COLORS } from '../../types/kg';

// Utility: map occurrence_count to node display radius
function nodeSize(occurrenceCount: number): number {
  return Math.max(4, Math.sqrt(occurrenceCount) * 3);
}

interface ForceGraphPanelProps {
  graphData: GraphResponse | undefined;
  isLoading: boolean;
}

const ForceGraphPanel: React.FC<ForceGraphPanelProps> = ({ graphData, isLoading }) => {
  const dispatch = useAppDispatch();
  const { focusEntityId, colorMap } = useAppSelector((s) => s.graph);
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<ForceGraphMethods<GraphNode>>(null);

  // Position cache: preserves node positions across graph reloads (stubbed — ready to wire)
  const positionCache = useRef<Map<string, { x: number; y: number }>>(new Map());

  // Center on focused node when focusEntityId changes
  useEffect(() => {
    if (!focusEntityId || !graphRef.current || !graphData) return;
    const node = graphData.nodes.find((n) => n.id === focusEntityId);
    if (node?.x != null && node?.y != null) {
      graphRef.current.centerAt(node.x, node.y, 500);
      graphRef.current.zoom(2.5, 500);
    }
  }, [focusEntityId, graphData]);

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      dispatch(setFocus(node.id));
    },
    [dispatch]
  );

  const handleBackgroundClick = useCallback(() => {
    dispatch(clearFocus());
  }, [dispatch]);

  const nodeCanvasObject = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const { x, y } = node as { x: number; y: number };
      const radius = nodeSize(node.occurrence_count);
      const color = colorMap[node.entity_type] ?? ENTITY_TYPE_COLORS[node.entity_type] ?? '#999';

      // Draw circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // Highlight focused node
      if (node.id === focusEntityId) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2 / globalScale;
        ctx.stroke();
      }

      // Label (only render when zoomed in enough)
      if (globalScale >= 1.5) {
        const label = node.name;
        const fontSize = Math.min(12, radius * 1.2) / globalScale;
        ctx.font = `${fontSize}px Sans-Serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x, y);
      }
    },
    [colorMap, focusEntityId]
  );

  const nodeLabel = useCallback(
    (node: GraphNode) => {
      const aliasText = node.aliases.length > 0 ? `\nAliases: ${node.aliases.join(', ')}` : '';
      return `${node.name} (${node.entity_type})${aliasText}`;
    },
    []
  );

  const linkLabel = useCallback(
    (link: { relation_type?: string; description?: string }) =>
      [link.relation_type, link.description].filter(Boolean).join(' — '),
    []
  );

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 2 }}>
        <CircularProgress />
        <Typography color="text.secondary">Loading graph…</Typography>
      </Box>
    );
  }

  if (!graphData) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
        <Typography color="text.secondary">Select a graph to visualize.</Typography>
      </Box>
    );
  }

  return (
    <Box ref={containerRef} sx={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData as any}
        nodeId="id"
        linkSource="source"
        linkTarget="target"
        nodeVal={(node) => nodeSize((node as GraphNode).occurrence_count)}
        nodeCanvasObject={nodeCanvasObject as any}
        nodeCanvasObjectMode={() => 'replace'}
        nodeLabel={nodeLabel as any}
        linkLabel={linkLabel as any}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkColor={() => 'rgba(150,150,150,0.4)'}
        onNodeClick={handleNodeClick as any}
        onBackgroundClick={handleBackgroundClick}
        backgroundColor="#1a1a2e"
        width={containerRef.current?.clientWidth}
        height={containerRef.current?.clientHeight}
      />
    </Box>
  );
};

export default ForceGraphPanel;
