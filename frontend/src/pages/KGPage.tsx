import React, { useMemo } from 'react';
import { Box, Divider } from '@mui/material';
import { useAppSelector } from '../store';
import {
  useGraphQuery,
  useSubgraphQuery,
  useGraphMetricsQuery,
  useNodeMetricQuery,
  useNodePairMetricQuery,
} from '../hooks/useKGQueries';
import { DISTANCE_METRICS, GraphResponse, METRIC_LABELS, NODE_PAIR_METRICS, NodeColorMetric, NodePairMetric, NodeSizeMetric } from '../types/kg';
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

function trimLeavesRecursive(
  nodes: GraphResponse['nodes'],
  links: GraphResponse['links'],
  protectedId: string | null
): { nodes: GraphResponse['nodes']; links: GraphResponse['links'] } {
  let activeNodes = new Set(nodes.map((n) => n.id));
  let activeLinks = links;

  while (true) {
    const degree = new Map<string, number>();
    Array.from(activeNodes).forEach((id) => degree.set(id, 0));
    for (const link of activeLinks) {
      const src = linkEndpointId(link.source);
      const tgt = linkEndpointId(link.target);
      if (activeNodes.has(src)) degree.set(src, (degree.get(src) ?? 0) + 1);
      if (activeNodes.has(tgt)) degree.set(tgt, (degree.get(tgt) ?? 0) + 1);
    }

    const leaves = new Set(
      Array.from(activeNodes).filter((id) => (degree.get(id) ?? 0) <= 1 && id !== protectedId)
    );
    if (leaves.size === 0) break;

    Array.from(leaves).forEach((id) => activeNodes.delete(id));
    activeLinks = activeLinks.filter((l) => {
      const src = linkEndpointId(l.source);
      const tgt = linkEndpointId(l.target);
      return !leaves.has(src) && !leaves.has(tgt);
    });
  }

  return {
    nodes: nodes.filter((n) => activeNodes.has(n.id)),
    links: activeLinks,
  };
}

// Compute min/max norm bounds from only the currently displayed nodes.
// Excludes sentinel -1 values (unreachable nodes in node-pair metrics).
function computeDisplayNorms(
  values: Record<string, number> | undefined,
  nodeIds: Set<string>
): [number, number] {
  if (!values || nodeIds.size === 0) return [0, 1];
  const displayed = Array.from(nodeIds)
    .map((id) => values[id])
    .filter((v): v is number => v !== undefined && v >= 0);
  if (displayed.length === 0) return [0, 1];
  return [Math.min(...displayed), Math.max(...displayed)];
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
  const { graphName, displayMode, hopCount, focusEntityId, occurrenceThreshold, trimLeaves } = useAppSelector((s) => s.graph);
  const { nodeSizeMetric, nodeColorMetric, nodeSizeParams, nodeColorParams } = useAppSelector((s) => s.metrics);

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

  // --------------- Metric queries ---------------

  // Graph-level metrics (shown in entity panel unfocused state)
  const graphMetricsQuery = useGraphMetricsQuery(graphName);

  // Node size metric
  const sizeMetricQuery = useNodeMetricQuery(
    graphName, nodeSizeMetric, nodeSizeParams,
    nodeSizeMetric !== NodeSizeMetric.OccurrenceCount
  );

  // Node color metric (node-level or node-pair)
  const isNodePairColorMetric = NODE_PAIR_METRICS.includes(nodeColorMetric as NodePairMetric);
  const colorMetricQuery = useNodeMetricQuery(
    graphName, nodeColorMetric as NodeColorMetric, nodeColorParams,
    !isNodePairColorMetric && nodeColorMetric !== NodeColorMetric.EntityType
  );
  const colorPairMetricQuery = useNodePairMetricQuery(
    graphName, focusEntityId, nodeColorMetric as NodePairMetric,
    isNodePairColorMetric && focusEntityId !== null
  );

  // Resolved metric values (full graph — norms are recomputed per displayed nodes below)
  const sizeMetricData = sizeMetricQuery.data;
  const sizeReady = sizeMetricData?.status === 'ready' && !('num_communities' in (sizeMetricData ?? {}));
  const sizeValues = sizeReady ? sizeMetricData!.values : undefined;

  // Determine active color metric data
  const activeColorData = isNodePairColorMetric
    ? colorPairMetricQuery.data
    : colorMetricQuery.data?.status === 'ready' ? colorMetricQuery.data : undefined;

  // Community metrics have a 'num_communities' field; continuous ones have 'norm_min'/'norm_max'
  const isCommunityMetric = activeColorData != null && 'num_communities' in activeColorData;
  const communityValues = isCommunityMetric
    ? (activeColorData as { values: Record<string, number> }).values
    : undefined;
  const colorMetricValues = !isCommunityMetric && activeColorData != null
    ? (activeColorData as { values: Record<string, number> }).values
    : undefined;

  // --------------- Display graph ---------------

  const displayGraph = useMemo(() => {
    if (!activeGraph) return activeGraph;
    const afterThreshold =
      occurrenceThreshold <= 1
        ? activeGraph
        : applyTrim(activeGraph, focusEntityId, occurrenceThreshold);
    if (!trimLeaves) return afterThreshold;
    const { nodes, links } = trimLeavesRecursive(
      afterThreshold.nodes,
      afterThreshold.links,
      focusEntityId
    );
    return { ...afterThreshold, nodes, links, node_count: nodes.length, edge_count: links.length };
  }, [activeGraph, occurrenceThreshold, trimLeaves, focusEntityId]);

  // Norm bounds recomputed from displayed nodes so trimmed nodes don't skew the color/size scale
  const displayNodeIds = useMemo(
    () => new Set((displayGraph?.nodes ?? []).map((n) => n.id)),
    [displayGraph]
  );
  const [sizeNormMin, sizeNormMax] = useMemo(
    () => computeDisplayNorms(sizeValues, displayNodeIds),
    [sizeValues, displayNodeIds]
  );
  const [colorNormMinRaw, colorNormMaxRaw] = useMemo(
    () => computeDisplayNorms(colorMetricValues, displayNodeIds),
    [colorMetricValues, displayNodeIds]
  );
  const invertColorScale = DISTANCE_METRICS.has(nodeColorMetric);
  const colorNormMin = invertColorScale ? colorNormMaxRaw : colorNormMinRaw;
  const colorNormMax = invertColorScale ? colorNormMinRaw : colorNormMaxRaw;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      <KGTopBar />
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <ForceGraphPanel
          graphData={displayGraph}
          isLoading={isLoading}
          sizeMetricValues={sizeValues}
          sizeNormMin={sizeNormMin}
          sizeNormMax={sizeNormMax}
          colorMetricValues={colorMetricValues}
          colorNormMin={colorNormMin}
          colorNormMax={colorNormMax}
          communityValues={communityValues}
          colorMetricLabel={METRIC_LABELS[nodeColorMetric as NodeColorMetric | NodePairMetric]}
        />
        <Divider orientation="vertical" flexItem />
        <Box sx={{ width: 340, flexShrink: 0, overflow: 'hidden', borderLeft: 0 }}>
          <EntityPanel
            activeGraph={activeGraph}
            graphMetrics={graphMetricsQuery.data}
            graphMetricsLoading={graphMetricsQuery.isLoading || graphMetricsQuery.isFetching}
          />
        </Box>
      </Box>
    </Box>
  );
};

export default KGPage;
