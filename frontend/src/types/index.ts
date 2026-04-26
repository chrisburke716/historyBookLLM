/**
 * TypeScript interfaces matching the API models.
 * These should stay in sync with the FastAPI Pydantic models.
 */

export interface SessionCreateRequest {
  title?: string;
}

export interface MessageRequest {
  content: string;
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
  citations?: string[]; // e.g., ["Page 123", "Page 456"]
  metadata?: Record<string, unknown>;
}

export interface SessionListResponse {
  sessions: SessionResponse[];
}

export interface MessageListResponse {
  messages: MessageResponse[];
}

export interface ChatResponse {
  message: MessageResponse;
  session: SessionResponse;
}

// Additional types for UI state management
export interface ChatState {
  currentSession: SessionResponse | null;
  sessions: SessionResponse[];
  messages: MessageResponse[];
  isLoading: boolean;
  error: string | null;
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