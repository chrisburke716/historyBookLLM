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
