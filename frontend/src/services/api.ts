/**
 * Centralized API service for the History Book Chat API.
 */

import axios, { AxiosInstance } from 'axios';
import {
  BookListResponse,
  ChapterListResponse,
  ChapterContentResponse,
} from '../types';
import { agentAPI } from './agentAPI';

class BooksAPI {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async getBooks(): Promise<BookListResponse> {
    const response = await this.api.get('/api/books');
    return response.data;
  }

  async getChapters(bookIndex: number): Promise<ChapterListResponse> {
    const response = await this.api.get(`/api/books/${bookIndex}/chapters`);
    return response.data;
  }

  async getChapterContent(bookIndex: number, chapterIndex: number): Promise<ChapterContentResponse> {
    const response = await this.api.get(`/api/books/${bookIndex}/chapters/${chapterIndex}`);
    return response.data;
  }
}

export const booksAPI = new BooksAPI();
export { agentAPI };
export const api = agentAPI;
export default api;
