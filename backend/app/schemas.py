from typing import Optional

from pydantic import BaseModel


class UrlIn(BaseModel):
    url: str


# ─── Settings schemas ─────────────────────────────────────────────────────────

class SettingsIn(BaseModel):
    # Connection
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    # Parsing
    parser: Optional[str] = None          # "pymupdf" | "mineru" | "reducto"
    vlm_parse: Optional[str] = None       # "off" | "auto" | "on"
    vlm_model: Optional[str] = None
    mineru_cmd: Optional[str] = None
    reducto_parse: Optional[str] = None   # "off" | "default" | "agentic"
    reducto_api_key: Optional[str] = None
    reducto_chunk_mode: Optional[str] = None
    reducto_chunk_size: Optional[int] = None
    reducto_filter_blocks: Optional[str] = None
    reducto_table_format: Optional[str] = None
    chunk_max_chars: Optional[int] = None
    chunk_overlap: Optional[int] = None
    # Indexing
    embed_model: Optional[str] = None
    embed_dim: Optional[str] = None
    enable_doc_summary: Optional[bool] = None
    doc_summary_chars: Optional[int] = None
    doc_summary_model: Optional[str] = None
    enable_section_summary: Optional[bool] = None
    section_summary_chars: Optional[int] = None
    # Retrieval
    bm25_top_k: Optional[int] = None
    dense_top_k: Optional[int] = None
    rrf_k: Optional[int] = None
    use_reranker: Optional[bool] = None
    reranker_model: Optional[str] = None
    rerank_top_n: Optional[int] = None
    final_top_k: Optional[int] = None
    use_hyde: Optional[bool] = None
    use_rse: Optional[bool] = None
    rse_irrelevant_penalty: Optional[float] = None
    rse_max_segment_chunks: Optional[int] = None
    rse_overall_max_chunks: Optional[int] = None
    # Generation
    llm_model: Optional[str] = None
    llm_model_fast: Optional[str] = None
    enable_replan: Optional[bool] = None
    max_replan_iters: Optional[int] = None
    enable_sufficiency: Optional[bool] = None
    enable_answer_verify: Optional[bool] = None
    max_answer_regenerations: Optional[int] = None
    # Memory
    enable_summarization: Optional[bool] = None
    summary_threshold: Optional[int] = None
    history_window: Optional[int] = None
    summary_model: Optional[str] = None


class TextIn(BaseModel):
    text: str
    title: Optional[str] = "Văn bản"


class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str


class CancelChatIn(BaseModel):
    session_id: str


class SessionIn(BaseModel):
    title: Optional[str] = None
