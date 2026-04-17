import axios, { AxiosInstance } from 'axios';
import {
  CommunityMetricResponse,
  EntityDetail,
  GraphListResponse,
  GraphMetricsResponse,
  GraphResponse,
  NodeMetricResponse,
  NodePairMetricResponse,
  SearchRequest,
  SearchResponse,
} from '../types/kg';

class KGAPI {
  private api: AxiosInstance;

  constructor(baseURL: string = process.env.REACT_APP_API_URL || 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 60000,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  async listGraphs(): Promise<GraphListResponse> {
    const response = await this.api.get<GraphListResponse>('/api/kg/graphs');
    return response.data;
  }

  async getGraph(graphName: string): Promise<GraphResponse> {
    const response = await this.api.get<GraphResponse>(`/api/kg/graphs/${graphName}`);
    return response.data;
  }

  async getSubgraph(
    graphName: string,
    entityId: string,
    hops: number
  ): Promise<GraphResponse> {
    const response = await this.api.get<GraphResponse>(
      `/api/kg/graphs/${graphName}/subgraph`,
      { params: { entity_id: entityId, hops } }
    );
    return response.data;
  }

  async getEntity(entityId: string): Promise<EntityDetail> {
    const response = await this.api.get<EntityDetail>(`/api/kg/entities/${entityId}`);
    return response.data;
  }

  async search(request: SearchRequest): Promise<SearchResponse> {
    const response = await this.api.post<SearchResponse>('/api/kg/search', request);
    return response.data;
  }

  async getGraphMetrics(graphName: string): Promise<GraphMetricsResponse> {
    const response = await this.api.get<GraphMetricsResponse>('/api/kg/metrics/graph', {
      params: { graph_name: graphName },
      validateStatus: (s) => s === 200 || s === 202,
    });
    return response.data;
  }

  async getNodeMetric(
    graphName: string,
    metric: string,
    params: Record<string, number>
  ): Promise<NodeMetricResponse | CommunityMetricResponse> {
    const response = await this.api.get<NodeMetricResponse | CommunityMetricResponse>(
      '/api/kg/metrics/node',
      {
        params: { graph_name: graphName, metric, ...params },
        validateStatus: (s) => s === 200 || s === 202,
      }
    );
    return response.data;
  }

  async getNodePairMetric(
    graphName: string,
    focusId: string,
    metric: string
  ): Promise<NodePairMetricResponse> {
    const response = await this.api.get<NodePairMetricResponse>(
      '/api/kg/metrics/node-pair',
      { params: { graph_name: graphName, focus_entity_id: focusId, metric } }
    );
    return response.data;
  }
}

export const kgAPI = new KGAPI();
