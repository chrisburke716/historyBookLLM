/**
 * TypeScript interfaces matching the API models.
 * These should stay in sync with the FastAPI Pydantic models.
 */

export interface SessionCreateRequest {
  title?: string;
}

export interface SessionResponse {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
}

export interface MessageResponse {
  id: string;
  content: string;
  role: string; // "user" or "assistant"
  timestamp: string;
  session_id: string;
  citations?: string[]; // e.g., ["[B3, Ch5, p.42]", ...]
  metadata?: Record<string, unknown>;
}

export interface SessionListResponse {
  sessions: SessionResponse[];
}

export interface MessageListResponse {
  messages: MessageResponse[];
}

// Book reading API types

export interface BookResponse {
  id: string;
  title: string;
  book_index: number;
  start_page: number;
  end_page: number;
}

export interface ChapterResponse {
  id: string;
  title: string;
  chapter_index: number;
  book_index: number;
  start_page: number;
  end_page: number;
}

export interface ParagraphResponse {
  text: string;
  page: number;
  paragraph_index: number;
}

export interface BookListResponse {
  books: BookResponse[];
}

export interface ChapterListResponse {
  chapters: ChapterResponse[];
}

export interface ChapterContentResponse {
  chapter: ChapterResponse;
  paragraphs: ParagraphResponse[];
}