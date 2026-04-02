import axios, { AxiosInstance } from 'axios';
import {
  GraphListResponse,
  GraphResponse,
  EntityDetail,
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
}

export const kgAPI = new KGAPI();
