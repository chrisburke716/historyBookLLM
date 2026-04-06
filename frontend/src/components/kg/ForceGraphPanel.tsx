import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useAppDispatch, useAppSelector } from '../../store';
import { setFocus, clearFocus } from '../../store/graphSlice';
import { GraphResponse, GraphNode, ENTITY_TYPE_COLORS } from '../../types/kg';

// Deterministic community color palette (sorted community IDs → palette index)
const COMMUNITY_PALETTE = [
  '#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1',
  '#76b7b2', '#edc948', '#ff9da7', '#9c755f', '#bab0ac',
];

// Utility: map occurrence_count to node display radius (fallback)
function occurrenceNodeSize(occurrenceCount: number): number {
  return Math.max(4, Math.log(occurrenceCount + 1) * 3);
}

// Utility: metric value → node radius (4–20px range)
function nodeDisplaySize(
  occurrenceCount: number,
  metricValue: number | undefined,
  normMin: number,
  normMax: number
): number {
  if (metricValue === undefined) return occurrenceNodeSize(occurrenceCount);
  const t = normMax > normMin ? (metricValue - normMin) / (normMax - normMin) : 0.5;
  return 4 + t * 16;
}

// Utility: continuous metric value → hex color (cool blue-purple scale)
function metricToColor(value: number, normMin: number, normMax: number): string {
  const t = normMax > normMin ? (value - normMin) / (normMax - normMin) : 0.5;
  const r = Math.round(50 + (1 - t) * 170);
  const g = Math.round(50 + (1 - t) * 120);
  const b = Math.round(200 + t * 55);
  return `rgb(${r},${g},${b})`;
}

// Utility: community ID → hex color (deterministic)
function communityToColor(communityId: number, sortedCommunityIds: number[]): string {
  const idx = sortedCommunityIds.indexOf(communityId);
  return COMMUNITY_PALETTE[Math.max(0, idx) % COMMUNITY_PALETTE.length];
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
  // Size metric data (undefined = use occurrence_count fallback)
  sizeMetricValues?: Record<string, number>;
  sizeNormMin?: number;
  sizeNormMax?: number;
  // Color metric data — continuous
  colorMetricValues?: Record<string, number>;
  colorNormMin?: number;
  colorNormMax?: number;
  // Color metric data — community (categorical)
  communityValues?: Record<string, number>;
}

const ForceGraphPanel: React.FC<ForceGraphPanelProps> = ({
  graphData,
  isLoading,
  sizeMetricValues,
  sizeNormMin = 0,
  sizeNormMax = 1,
  colorMetricValues,
  colorNormMin = 0,
  colorNormMax = 1,
  communityValues,
}) => {
  const dispatch = useAppDispatch();
  const { focusEntityId, colorMap } = useAppSelector((s) => s.graph);

  // Pre-compute sorted community IDs for deterministic color assignment
  const sortedCommunityIds = useMemo((): number[] => {
    if (!communityValues) return [];
    const unique = Array.from(new Set(Object.values(communityValues)));
    return unique.sort((a, b) => a - b);
  }, [communityValues]);
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);

  // Set of node IDs directly connected to the focused entity (first-order neighbors)
  const neighborIds = useMemo((): Set<string> => {
    if (!focusEntityId || !graphData) return new Set();
    const neighbors = new Set<string>();
    for (const link of graphData.links) {
      const src = linkNodeId(link.source);
      const tgt = linkNodeId(link.target);
      if (src === focusEntityId) neighbors.add(tgt);
      if (tgt === focusEntityId) neighbors.add(src);
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

      // --- Node size ---
      const radius = nodeDisplaySize(
        node.occurrence_count,
        sizeMetricValues?.[node.id],
        sizeNormMin,
        sizeNormMax
      );

      // --- Node color (precedence: community > continuous > entity_type) ---
      let color: string;
      if (communityValues?.[node.id] !== undefined) {
        color = communityToColor(communityValues[node.id], sortedCommunityIds);
      } else if (colorMetricValues?.[node.id] !== undefined) {
        color = metricToColor(colorMetricValues[node.id], colorNormMin, colorNormMax);
      } else {
        color = colorMap[node.entity_type] ?? ENTITY_TYPE_COLORS[node.entity_type] ?? '#999';
      }

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
    [
      colorMap, focusEntityId, neighborIds,
      sizeMetricValues, sizeNormMin, sizeNormMax,
      colorMetricValues, colorNormMin, colorNormMax,
      communityValues, sortedCommunityIds,
    ]
  );

  const nodeLabel = useCallback(
    (node: GraphNode) => {
      const aliasText = node.aliases.length > 0 ? `\nAliases: ${node.aliases.join(', ')}` : '';
      const sizeVal = sizeMetricValues?.[node.id];
      const colorVal = colorMetricValues?.[node.id] ?? communityValues?.[node.id];
      const metricText = (sizeVal !== undefined || colorVal !== undefined)
        ? `\nSize: ${sizeVal?.toFixed(3) ?? 'occurrence_count'} | Color: ${colorVal?.toFixed(3) ?? 'entity_type'}`
        : '';
      return `${node.name} (${node.entity_type})${aliasText}${metricText}`;
    },
    [sizeMetricValues, colorMetricValues, communityValues]
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
        nodeVal={(node) => {
          const n = node as GraphNode;
          return nodeDisplaySize(n.occurrence_count, sizeMetricValues?.[n.id], sizeNormMin, sizeNormMax);
        }}
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
