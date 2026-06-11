import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val not in (None, "") else default
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL") or None

        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_model_fast = os.getenv("LLM_MODEL_FAST", "gpt-4o-mini")
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        _dim = os.getenv("EMBED_DIM", "")
        self.embed_dim = int(_dim) if _dim.strip() else None

        self.bm25_top_k = _get_int("BM25_TOP_K", 30)
        self.dense_top_k = _get_int("DENSE_TOP_K", 30)
        self.rrf_k = _get_int("RRF_K", 60)
        self.rerank_top_n = _get_int("RERANK_TOP_N", 20)
        self.final_top_k = _get_int("FINAL_TOP_K", 5)
        self.use_reranker = _get_bool("USE_RERANKER", True)
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.reranker_device = os.getenv("RERANKER_DEVICE", "auto")
        self.reranker_batch_size = _get_int("RERANKER_BATCH_SIZE", 16)
        self.use_hyde = _get_bool("USE_HYDE", False)

        self.max_replan_iters = _get_int("MAX_REPLAN_ITERS", 3)
        self.max_answer_regenerations = _get_int("MAX_ANSWER_REGENERATIONS", 1)
        self.enable_replan = _get_bool("ENABLE_REPLAN", True)
        self.enable_sufficiency = _get_bool("ENABLE_SUFFICIENCY", True)
        self.enable_answer_verify = _get_bool("ENABLE_ANSWER_VERIFY", True)
        self.enable_drag = _get_bool("ENABLE_DRAG", True)

        self.enable_summarization = _get_bool("ENABLE_SUMMARIZATION", True)
        self.summary_threshold = _get_int("SUMMARY_THRESHOLD", 12)
        self.history_window = _get_int("HISTORY_WINDOW", 6)
        self.summary_model = os.getenv("SUMMARY_MODEL", "") or self.llm_model_fast
        self.turbovec_bit_width = _get_int("TURBOVEC_BIT_WIDTH", 4)
        log_file = os.getenv("LOG_FILE", "logs/app.log")
        self.log_file = (BASE_DIR / log_file) if not os.path.isabs(log_file) else Path(log_file)

        # Ingestion / parsing
        # VLM_PARSE: "off" (local only), "auto" (VLM fallback for empty/scanned pages),
        # "on" (always VLM). Default "auto".
        self.vlm_parse = (os.getenv("VLM_PARSE", "auto") or "auto").strip().lower()
        self.vlm_model = os.getenv("VLM_MODEL", "gpt-4o-mini")
        self.chunk_max_chars = _get_int("CHUNK_MAX_CHARS", 1000)
        self.chunk_overlap = _get_int("CHUNK_OVERLAP", 200)

        # Reducto parser
        # REDUCTO_PARSE: "off" (PyMuPDF+VLM, free, local),
        #   "default" (1 credit/page, good quality),
        #   "agentic" (2 credits/page, best quality, VLM text+table review).
        self.reducto_api_key = os.getenv("REDUCTO_API_KEY", "")
        self.reducto_parse = (os.getenv("REDUCTO_PARSE", "off") or "off").strip().lower()
        self.reducto_chunk_mode = os.getenv("REDUCTO_CHUNK_MODE", "page_sections")
        self.reducto_chunk_size = _get_int("REDUCTO_CHUNK_SIZE", 1200)
        self.reducto_filter_blocks = [
            b.strip()
            for b in os.getenv("REDUCTO_FILTER_BLOCKS", "Header,Footer,Page Number").split(",")
            if b.strip()
        ]
        self.reducto_table_format = os.getenv("REDUCTO_TABLE_FORMAT", "dynamic")

        self.max_upload_size = _get_int("MAX_UPLOAD_SIZE", 5 * 1024 * 1024)

        storage = os.getenv("STORAGE_DIR", "storage")
        self.storage_dir = (BASE_DIR / storage) if not os.path.isabs(storage) else Path(storage)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "rag.db"

    @property
    def vector_path(self) -> Path:
        return self.storage_dir / "vector.tvim"

    @property
    def bm25_path(self) -> Path:
        return self.storage_dir / "bm25.pkl"

    @property
    def meta_path(self) -> Path:
        return self.storage_dir / "index_meta.json"

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> "Settings":
    return Settings()
