import { useQuery, useQueries } from '@tanstack/react-query';
import { kgAPI } from '../services/kgAPI';
import { booksAPI } from '../services/api';
import { NodeColorMetric, NodePairMetric, NodeSizeMetric } from '../types/kg';

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

// Returns lookup maps for book and chapter titles, built from existing /api/books endpoints.
// bookTitles: bookIndex → title
// chapterTitles: "bookIndex:chapterIndex" → title
export function useBooksWithChaptersQuery() {
  const booksQuery = useQuery({
    queryKey: ['books'],
    queryFn: () => booksAPI.getBooks(),
    staleTime: Infinity,
  });

  const books = booksQuery.data?.books ?? [];

  const chaptersQueries = useQueries({
    queries: books.map((book) => ({
      queryKey: ['chapters', book.book_index],
      queryFn: () => booksAPI.getChapters(book.book_index),
      staleTime: Infinity,
    })),
  });

  const bookTitles = new Map<number, string>();
  books.forEach((b) => bookTitles.set(b.book_index, b.title));

  const chapterTitles = new Map<string, string>();
  chaptersQueries.forEach((q) => {
    q.data?.chapters.forEach((ch) => {
      chapterTitles.set(`${ch.book_index}:${ch.chapter_index}`, ch.title);
    });
  });

  return { bookTitles, chapterTitles };
}

export function useEntityQuery(entityId: string | null) {
  return useQuery({
    queryKey: ['kg', 'entity', entityId],
    queryFn: () => kgAPI.getEntity(entityId!),
    enabled: entityId !== null,
    staleTime: 5 * 60 * 1000,
  });
}

export function useGraphMetricsQuery(graphName: string) {
  return useQuery({
    queryKey: ['kg', 'metrics', 'graph', graphName],
    queryFn: () => kgAPI.getGraphMetrics(graphName),
    // Poll every 2s while backend is still computing
    refetchInterval: (query) =>
      query.state.data?.status === 'computing' ? 2000 : false,
    staleTime: Infinity, // server caches indefinitely; no need to re-fetch once ready
  });
}

export function useNodeMetricQuery(
  graphName: string,
  metric: NodeSizeMetric | NodeColorMetric,
  params: Record<string, number>,
  enabled: boolean
) {
  return useQuery({
    queryKey: ['kg', 'metrics', 'node', graphName, metric, params],
    queryFn: () => kgAPI.getNodeMetric(graphName, metric, params),
    enabled,
    refetchInterval: (query) =>
      query.state.data?.status === 'computing' ? 2000 : false,
    staleTime: Infinity,
  });
}

export function useNodePairMetricQuery(
  graphName: string,
  focusId: string | null,
  metric: NodePairMetric,
  enabled: boolean
) {
  return useQuery({
    queryKey: ['kg', 'metrics', 'node-pair', graphName, focusId, metric],
    queryFn: () => kgAPI.getNodePairMetric(graphName, focusId!, metric),
    enabled: enabled && focusId !== null,
    staleTime: 0, // always fresh when focus changes
  });
}
