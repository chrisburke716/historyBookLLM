import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useAppDispatch, useAppSelector } from '../../store';
import { setFocus, clearFocus } from '../../store/graphSlice';
import { GraphResponse, GraphNode, ENTITY_TYPE_COLORS } from '../../types/kg';

// Utility: map occurrence_count to node display radius
function nodeSize(occurrenceCount: number): number {
  return Math.max(4, Math.log(occurrenceCount + 1) * 3);
}

// After the force simulation runs, link.source/target mutate from string IDs to node objects.
// This helper handles both forms.
function linkNodeId(endpoint: unknown): string {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return (endpoint as { id: string }).id;
  }
  return endpoint as string;
}

interface ForceGraphPanelProps {
  graphData: GraphResponse | undefined;
  isLoading: boolean;
}

const ForceGraphPanel: React.FC<ForceGraphPanelProps> = ({ graphData, isLoading }) => {
  const dispatch = useAppDispatch();
  const { focusEntityId, colorMap } = useAppSelector((s) => s.graph);
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);

  // Position cache: preserves node positions across graph reloads (stubbed — ready to wire)
  const positionCache = useRef<Map<string, { x: number; y: number }>>(new Map());

  // Set of node IDs directly connected to the focused entity (first-order neighbors)
  const neighborIds = useMemo((): Set<string> => {
    if (!focusEntityId || !graphData) return new Set();
    const neighbors = new Set<string>();
    for (const link of graphData.links) {
      if (link.source === focusEntityId) neighbors.add(link.target);
      if (link.target === focusEntityId) neighbors.add(link.source);
    }
    return neighbors;
  }, [focusEntityId, graphData]);

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

      const isFocused = node.id === focusEntityId;
      const isNeighbor = neighborIds.has(node.id);
      const isDimmed = focusEntityId !== null && !isFocused && !isNeighbor;

      ctx.globalAlpha = isDimmed ? 0.5 : 1;

      // Draw circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // Ring on focused node
      if (isFocused) {
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2 / globalScale;
        ctx.stroke();
      }

      // Label rendered below the node (only when zoomed in enough)
      if (globalScale >= 1.5) {
        const fontSize = 12 / globalScale;
        ctx.font = `${fontSize}px Sans-Serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = '#222';
        ctx.fillText(node.name, x, y + radius + 2 / globalScale);
      }

      ctx.globalAlpha = 1;
    },
    [colorMap, focusEntityId, neighborIds]
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
        linkColor={(link: any) => {
          if (!focusEntityId) return 'rgba(0,0,0,0.25)';
          const src = linkNodeId(link.source);
          const tgt = linkNodeId(link.target);
          const isFirstOrder = src === focusEntityId || tgt === focusEntityId;
          return isFirstOrder ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.15)';
        }}
        linkWidth={(link: any) => {
          if (!focusEntityId) return 1;
          const src = linkNodeId(link.source);
          const tgt = linkNodeId(link.target);
          return src === focusEntityId || tgt === focusEntityId ? 1.5 : 0.5;
        }}
        linkDirectionalArrowColor={(link: any) => {
          if (!focusEntityId) return 'rgba(0,0,0,0.5)';
          const src = linkNodeId(link.source);
          const tgt = linkNodeId(link.target);
          return src === focusEntityId || tgt === focusEntityId ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.25)';
        }}
        onNodeClick={handleNodeClick as any}
        onBackgroundClick={handleBackgroundClick}
        backgroundColor="#ffffff"
        width={containerRef.current?.clientWidth}
        height={containerRef.current?.clientHeight}
      />
    </Box>
  );
};

export default ForceGraphPanel;
