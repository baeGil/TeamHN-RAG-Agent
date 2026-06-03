export interface Citation {
  label: number;
  chunk_id: number;
  document_id: number;
  doc_title: string;
  doc_source: string;
  page: number | null;
  section: string | null;
  text: string;
  score: number;
  cited: boolean;
}

export interface TraceEvent {
  type: string;
  data: any;
}

export interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  trace: TraceEvent[];
  status?: "processing" | "complete" | "failed";
  error_message?: string | null;
}

export interface DocumentItem {
  id: number;
  title: string;
  source: string;
  source_type: string;
  n_chunks: number;
  created_at: string;
  status?: "processing" | "ready" | "failed";
  error_message?: string | null;
}

export interface SessionItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface AppConfig {
  openai_configured: boolean;
  llm_model: string;
  llm_model_fast: string;
  embed_model: string;
  use_reranker: boolean;
  reranker_model: string;
  max_upload_size: number;
}
