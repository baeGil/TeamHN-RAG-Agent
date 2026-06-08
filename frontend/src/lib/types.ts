export interface Citation {
  label: number;
  chunk_id: number;
  chunk_ids?: number[];      // RSE segment: all constituent chunk IDs
  document_id: number;
  doc_title: string;
  doc_source: string;
  page: number | string | null;
  pages?: number[];          // RSE segment: all page numbers in segment
  section: string | null;
  text: string;
  n_chunks?: number;         // RSE segment: how many chunks are merged
  score: number;
  cited: boolean;
  is_segment?: boolean;     // true if this is an RSE segment (merged chunks)
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
  reranker_type: string;
  reranker_model: string;
  jina_api_key: string;
  max_upload_size: number;
}

export interface AppSettings {
  connection: {
    openai_api_key: string;
    openai_base_url: string;
  };
  parsing: {
    parser: "pymupdf" | "mineru" | "reducto";
    vlm_parse: "off" | "auto" | "on";
    vlm_model: string;
    mineru_cmd: string;
    reducto_parse: "off" | "default" | "agentic";
    reducto_api_key: string;
    reducto_chunk_mode: string;
    reducto_chunk_size: number;
    reducto_filter_blocks: string;
    reducto_table_format: string;
    chunk_max_chars: number;
    chunk_overlap: number;
  };
  indexing: {
    embed_model: string;
    embed_dim: string;
    enable_doc_summary: boolean;
    doc_summary_chars: number;
    doc_summary_model: string;
    enable_section_summary: boolean;
    section_summary_chars: number;
  };
  retrieval: {
    bm25_top_k: number;
    dense_top_k: number;
    rrf_k: number;
    use_reranker: boolean;
    reranker_model: string;
    rerank_top_n: number;
    final_top_k: number;
    use_hyde: boolean;
    use_rse: boolean;
    rse_irrelevant_penalty: number;
    rse_max_segment_chunks: number;
    rse_overall_max_chunks: number;
    rse_window_extension: number;
    rse_chunk_length_adjustment: boolean;
    min_chunk_chars: number;
    complex_ctx_limit: number;
  };
  generation: {
    llm_model: string;
    llm_model_fast: string;
    enable_replan: boolean;
    max_replan_iters: number;
    enable_sufficiency: boolean;
    enable_answer_verify: boolean;
    enable_answer_verify_simple: boolean;
    enable_answer_verify_complex: boolean;
    max_answer_regenerations: number;
  };
  memory: {
    enable_summarization: boolean;
    summary_threshold: number;
    history_window: number;
    summary_model: string;
  };
}
