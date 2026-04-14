// Knowledge Graph API types — mirror backend kg_models.py exactly

export interface GraphNode {
  id: string;
  name: string;
  entity_type: string;
  occurrence_count: number;
  aliases: string[];
  // Added by react-force-graph-2d during simulation:
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  relation_type: string;
  description: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  links: GraphLink[];
  graph_name: string;
  node_count: number;
  edge_count: number;
}

export interface RelationshipSummary {
  relationship_id: string;
  relation_type: string;
  description: string;
  direction: 'outgoing' | 'incoming';
  other_entity_id: string;
  other_entity_name: string;
  book_index: number;
  chapter_index: number;
}

export interface EntityDetail {
  id: string;
  name: string;
  entity_type: string;
  aliases: string[];
  descriptions: string[];
  occurrence_count: number;
  relationships: RelationshipSummary[];
}

export interface SearchResult {
  id: string;
  name: string;
  entity_type: string;
  aliases: string[];
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
}

export interface SearchRequest {
  query: string;
  graph_name?: string;
  entity_types?: string[];
  limit?: number;
}

export interface KGGraphMeta {
  id: string;
  name: string;
  graph_type: string; // "chapter" | "book" | "volume"
  entity_count: number;
  relationship_count: number;
  book_chapters: string[]; // e.g. ["3:4"]
}

export interface GraphListResponse {
  graphs: KGGraphMeta[];
}

// Entity type values from backend EntityType enum
export const ENTITY_TYPES = ['person', 'polity', 'place', 'event', 'concept'] as const;
export type EntityType = typeof ENTITY_TYPES[number];

// Color palette for entity types
export const ENTITY_TYPE_COLORS: Record<string, string> = {
  person: '#4e79a7',
  polity: '#f28e2b',
  place: '#59a14f',
  event: '#e15759',
  concept: '#b07aa1',
};

// ---------------------------------------------------------------------------
// Metric enums (mirror backend NodeSizeMetric / NodeColorMetric / NodePairMetric)
// ---------------------------------------------------------------------------

export enum NodeSizeMetric {
  OccurrenceCount = 'occurrence_count',
  DegreeCentrality = 'degree_centrality',
  BetweennessCentrality = 'betweenness_centrality',
  PageRank = 'pagerank',
  ClosenessCentrality = 'closeness_centrality',
  KCoreNumber = 'kcore_number',
}

export enum NodeColorMetric {
  EntityType = 'entity_type',
  CommunityLouvain = 'community_louvain',
  CommunityGirvanNewman = 'community_girvan_newman',
  CommunityLabelPropagation = 'community_label_propagation',
  CommunitySpectral = 'community_spectral',
  LocalClusteringCoefficient = 'local_clustering_coefficient',
  KCoreNumber = 'kcore_number',
}

export enum NodePairMetric {
  CosineSimilarity = 'cosine_similarity',
  JaccardSimilarity = 'jaccard_similarity',
  AdamicAdar = 'adamic_adar',
  CommonNeighborCount = 'common_neighbor_count',
  ShortestPathLength = 'shortest_path_length',
  ResistanceDistance = 'resistance_distance',
}

export const NODE_PAIR_METRICS: NodePairMetric[] = Object.values(NodePairMetric);

// Human-readable labels for color metric selector and legend
export const METRIC_LABELS: Record<NodeColorMetric | NodePairMetric, string> = {
  [NodeColorMetric.EntityType]: 'Entity type',
  [NodeColorMetric.CommunityLouvain]: 'Community — Louvain',
  [NodeColorMetric.CommunityGirvanNewman]: 'Community — Girvan-Newman',
  [NodeColorMetric.CommunityLabelPropagation]: 'Community — Label prop.',
  [NodeColorMetric.CommunitySpectral]: 'Community — Spectral',
  [NodeColorMetric.LocalClusteringCoefficient]: 'Local clustering coeff.',
  [NodeColorMetric.KCoreNumber]: 'K-core number',
  [NodePairMetric.CosineSimilarity]: 'Cosine similarity',
  [NodePairMetric.JaccardSimilarity]: 'Jaccard similarity',
  [NodePairMetric.AdamicAdar]: 'Adamic-Adar',
  [NodePairMetric.CommonNeighborCount]: 'Common neighbors',
  [NodePairMetric.ShortestPathLength]: 'Shortest path length',
  [NodePairMetric.ResistanceDistance]: 'Resistance distance',
};

// Metrics where lower value = closer to focus — color scale is inverted so
// close nodes appear bright and distant nodes appear dark.
export const DISTANCE_METRICS = new Set<string>([
  NodePairMetric.ResistanceDistance,
  NodePairMetric.ShortestPathLength,
]);

// ---------------------------------------------------------------------------
// Metric response types (mirror backend Pydantic models)
// ---------------------------------------------------------------------------

export interface GraphMetricsResponse {
  graph_name: string;
  density: number;
  giant_component_ratio: number;
  num_connected_components: number;
  avg_shortest_path_length: number | null;
  diameter: number | null;
  global_clustering_coefficient: number;
  num_communities: number;
  articulation_point_count: number;
  status: 'ready' | 'computing';
}

export interface NodeMetricResponse {
  graph_name: string;
  metric: string;
  params: Record<string, number>;
  values: Record<string, number>; // entity UUID → metric value
  norm_min: number;
  norm_max: number;
  status: 'ready' | 'computing';
}

export interface CommunityMetricResponse {
  graph_name: string;
  metric: string;
  params: Record<string, number>;
  values: Record<string, number>; // entity UUID → community ID (int stored as number)
  num_communities: number;
  status: 'ready' | 'computing';
}

export interface NodePairMetricResponse {
  graph_name: string;
  focus_entity_id: string;
  metric: string;
  params: Record<string, number>;
  values: Record<string, number>; // entity UUID → value relative to focus
  norm_min: number;
  norm_max: number;
}
