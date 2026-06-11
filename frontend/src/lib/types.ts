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

export interface ConflictPair {
  doc_i_id: number | string | null;
  doc_j_id: number | string | null;
  doc_i_label?: number | null;
  doc_j_label?: number | null;
  doc_i_title?: string | null;
  doc_j_title?: string | null;
  doc_i_page?: number | null;
  doc_j_page?: number | null;
  conflict_probability: number;
  type_label: "no-conflict" | "factual" | "temporal" | string;
  type_probabilities?: Record<string, number>;
  doc_i_preview?: string;
  doc_j_preview?: string;
}

export interface ConflictReport {
  enabled: boolean;
  has_conflict: boolean;
  threshold?: number;
  num_documents?: number;
  num_pairs?: number;
  duration_ms?: number;
  conflict_pairs: ConflictPair[];
}
export interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
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
  max_upload_size: number;
}
