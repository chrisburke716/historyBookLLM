import { useQuery } from '@tanstack/react-query';
import { kgAPI } from '../services/kgAPI';

export function useGraphListQuery() {
  return useQuery({
    queryKey: ['kg', 'graphs'],
    queryFn: () => kgAPI.listGraphs(),
    staleTime: 5 * 60 * 1000, // 5 minutes — graph list changes rarely
  });
}

export function useGraphQuery(graphName: string) {
  return useQuery({
    queryKey: ['kg', 'graph', graphName],
    queryFn: () => kgAPI.getGraph(graphName),
    staleTime: 5 * 60 * 1000,
  });
}

export function useSubgraphQuery(
  entityId: string | null,
  hops: number,
  graphName: string,
  enabled: boolean
) {
  return useQuery({
    queryKey: ['kg', 'subgraph', graphName, entityId, hops],
    queryFn: () => kgAPI.getSubgraph(graphName, entityId!, hops),
    enabled: enabled && entityId !== null,
    staleTime: 5 * 60 * 1000,
  });
}

export function useEntityQuery(entityId: string | null) {
  return useQuery({
    queryKey: ['kg', 'entity', entityId],
    queryFn: () => kgAPI.getEntity(entityId!),
    enabled: entityId !== null,
    staleTime: 5 * 60 * 1000,
  });
}
