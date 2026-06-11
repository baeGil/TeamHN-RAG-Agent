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

export interface ConflictPair {
  pair_index: number;
  conflict: boolean;
  raw_label: string;
  confidence: number;
  chunk_a_label: number;
  chunk_b_label: number;
  chunk_a_id: number;
  chunk_b_id: number;
  chunk_a_title: string;
  chunk_b_title: string;
  chunk_a_page: number | null;
  chunk_b_page: number | null;
  text_a_preview: string;
  text_b_preview: string;
}

export interface ConflictCheck {
  available: boolean;
  reason?: string;
  model?: string;
  checked_pairs: number;
  input_chars?: number;
  conflicts: ConflictPair[];
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
  conflicts?: ConflictPair[];
  conflict_check?: ConflictCheck | null;
  trace: TraceEvent[];
  status?: "processing" | "complete" | "failed" | "cancelled";
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
  enable_conflict_check: boolean;
  conflict_model: string;
  max_upload_size: number;
}
