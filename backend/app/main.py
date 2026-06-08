import asyncio
import json
import logging
import os
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import FileResponse, JSONResponse
from starlette.responses import StreamingResponse

from contextlib import asynccontextmanager

from .agent.graph import Agent
from .config import get_settings
from .indexing.store import KnowledgeBase
from .schemas import CancelChatIn, ChatIn, SessionIn, SettingsIn, TextIn, UrlIn

settings = get_settings()


def _configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"

    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "baseFilename", None) == str(settings.log_file)
    ]

    if not any(
        getattr(handler, "baseFilename", None) == str(settings.log_file)
        for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)


_configure_logging()

kb = KnowledgeBase(settings)


@asynccontextmanager
async def lifespan(app):
    cleaned = kb.repo.cleanup_stale_processing(max_age_minutes=10)
    if cleaned:
        logging.getLogger("rag.flow").info("RAG_FLOW startup_cleanup cleaned=%s stale records", cleaned)
    yield


app = FastAPI(title="Vietnamese RAG Agent", version="1.0.0", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Quá nhiều yêu cầu. Vui lòng thử lại sau."},
    )

kb = KnowledgeBase(settings)


def _maybe_summarize(session_id: str, agent: Agent, knowledge_base: KnowledgeBase) -> None:
    total = knowledge_base.repo.message_count(session_id)
    window = settings.history_window
    threshold = settings.summary_threshold
    if total <= threshold:
        return
    summary_row = knowledge_base.repo.get_summary(session_id)
    summarized_up_to = summary_row["summarized_up_to"] if summary_row else 0
    new_messages_start = summarized_up_to + 1
    window_messages = knowledge_base.repo.recent_history(session_id, limit=window)
    all_messages_raw = knowledge_base.repo.get_messages(session_id)
    recent_ids = {m["id"] for m in all_messages_raw[-len(window_messages):]} if window_messages else set()
    min_recent_id = min(recent_ids) if recent_ids else float("inf")
    messages_to_summarize = [
        {"role": m["role"], "content": m["content"]}
        for m in all_messages_raw
        if m["id"] >= new_messages_start and m["id"] < min_recent_id
    ]
    if not messages_to_summarize:
        return
    old_summary = summary_row["summary"] if summary_row else ""
    try:
        new_summary = agent.summarize_conversation(session_id, old_summary, messages_to_summarize)
        old_ids = [m["id"] for m in all_messages_raw if m["id"] < min_recent_id]
        last_summarized_id = max(old_ids) if old_ids else all_messages_raw[-1]["id"]
        knowledge_base.repo.save_summary(session_id, new_summary, last_summarized_id)
    except Exception:
        pass


@app.get("/api/health")
def health():
    return {"status": "ok", "openai_configured": settings.has_openai}


@app.get("/api/config")
def config():
    return {
        "openai_configured": settings.has_openai,
        "llm_model": settings.llm_model,
        "llm_model_fast": settings.llm_model_fast,
        "embed_model": settings.embed_model,
        "use_reranker": settings.use_reranker,
        "reranker_type": settings.reranker_type,
        "reranker_model": settings.reranker_model,
        "jina_api_key": _mask(settings.jina_api_key) if settings.jina_api_key else "",
        "max_upload_size": settings.max_upload_size,
    }


@app.get("/api/stats")
def stats():
    return kb.stats()


# ─── Settings read/write ──────────────────────────────────────────────────────

def _mask(value: str) -> str:
    """Show first 6 chars then asterisks."""
    if not value:
        return ""
    return value[:6] + "***" if len(value) > 6 else "***"


def _bool_str(v: bool) -> str:
    return "true" if v else "false"


@app.get("/api/settings")
def get_settings_endpoint():
    s = settings
    # Derive virtual "parser" field from the three parser flags
    if s.reducto_parse != "off":
        parser = "reducto"
    elif s.mineru_parse != "off":
        parser = "mineru"
    else:
        parser = "pymupdf"
    return {
        "connection": {
            "openai_api_key": _mask(s.openai_api_key),
            "openai_base_url": s.openai_base_url or "",
        },
        "parsing": {
            "parser": parser,
            "vlm_parse": s.vlm_parse,
            "vlm_model": s.vlm_model,
            "mineru_cmd": s.mineru_cmd,
            "reducto_parse": s.reducto_parse,
            "reducto_api_key": _mask(s.reducto_api_key),
            "reducto_chunk_mode": s.reducto_chunk_mode,
            "reducto_chunk_size": s.reducto_chunk_size,
            "reducto_filter_blocks": ",".join(s.reducto_filter_blocks),
            "reducto_table_format": s.reducto_table_format,
            "chunk_max_chars": s.chunk_max_chars,
            "chunk_overlap": s.chunk_overlap,
        },
        "indexing": {
            "embed_model": s.embed_model,
            "embed_dim": str(s.embed_dim) if s.embed_dim else "",
            "enable_doc_summary": s.enable_doc_summary,
            "doc_summary_chars": s.doc_summary_chars,
            "doc_summary_model": s.doc_summary_model if s.doc_summary_model != s.llm_model_fast else "",
            "enable_section_summary": s.enable_section_summary,
            "section_summary_chars": s.section_summary_chars,
        },
        "retrieval": {
            "bm25_top_k": s.bm25_top_k,
            "dense_top_k": s.dense_top_k,
            "rrf_k": s.rrf_k,
            "use_reranker": s.use_reranker,
            "reranker_type": s.reranker_type,
            "reranker_model": s.reranker_model,
            "jina_api_key": _mask(s.jina_api_key) if s.jina_api_key else "",
            "rerank_top_n": s.rerank_top_n,
            "final_top_k": s.final_top_k,
            "use_hyde": s.use_hyde,
            "use_rse": s.use_rse,
            "rse_irrelevant_penalty": s.rse_irrelevant_penalty,
            "rse_max_segment_chunks": s.rse_max_segment_chunks,
            "rse_overall_max_chunks": s.rse_overall_max_chunks,
            "rse_window_extension": s.rse_window_extension,
            "rse_chunk_length_adjustment": s.rse_chunk_length_adjustment,
            "complex_ctx_limit": s.complex_ctx_limit,
            "min_chunk_chars": s.min_chunk_chars,
        },
        "generation": {
            "llm_model": s.llm_model,
            "llm_model_fast": s.llm_model_fast,
            "enable_replan": s.enable_replan,
            "max_replan_iters": s.max_replan_iters,
            "enable_sufficiency": s.enable_sufficiency,
            "enable_answer_verify": s.enable_answer_verify,
            "enable_answer_verify_simple": s.enable_answer_verify_simple,
            "enable_answer_verify_complex": s.enable_answer_verify_complex,
            "max_answer_regenerations": s.max_answer_regenerations,
        },
        "memory": {
            "enable_summarization": s.enable_summarization,
            "summary_threshold": s.summary_threshold,
            "history_window": s.history_window,
            "summary_model": s.summary_model if s.summary_model != s.llm_model_fast else "",
        },
    }


# Settings that require re-indexing if changed
_REINDEX_KEYS = {"EMBED_MODEL", "EMBED_DIM", "CHUNK_MAX_CHARS", "CHUNK_OVERLAP"}


@app.put("/api/settings")
def update_settings_endpoint(body: SettingsIn):
    from dotenv import set_key as dotenv_set_key
    from pathlib import Path as _Path
    from .config import BASE_DIR

    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        env_path.touch()

    changed: dict[str, str] = {}
    needs_reindex = False

    def _write(env_key: str, value: str) -> None:
        nonlocal needs_reindex
        dotenv_set_key(str(env_path), env_key, value, quote_mode="never")
        changed[env_key] = value
        if env_key in _REINDEX_KEYS:
            needs_reindex = True

    # ── Connection ────────────────────────────────────────────────────────────
    if body.openai_api_key is not None and not body.openai_api_key.endswith("***"):
        _write("OPENAI_API_KEY", body.openai_api_key)
    if body.openai_base_url is not None:
        _write("OPENAI_BASE_URL", body.openai_base_url)

    # ── Parsing ───────────────────────────────────────────────────────────────
    if body.parser is not None:
        if body.parser == "mineru":
            _write("MINERU_PARSE", "on")
            _write("REDUCTO_PARSE", "off")
        elif body.parser == "reducto":
            _write("MINERU_PARSE", "off")
            _write("REDUCTO_PARSE", body.reducto_parse or "default")
        else:  # pymupdf
            _write("MINERU_PARSE", "off")
            _write("REDUCTO_PARSE", "off")
    if body.vlm_parse is not None:
        _write("VLM_PARSE", body.vlm_parse)
    if body.vlm_model is not None:
        _write("VLM_MODEL", body.vlm_model)
    if body.mineru_cmd is not None:
        _write("MINERU_CMD", body.mineru_cmd)
    if body.reducto_api_key is not None and not body.reducto_api_key.endswith("***"):
        _write("REDUCTO_API_KEY", body.reducto_api_key)
    if body.reducto_parse is not None:
        _write("REDUCTO_PARSE", body.reducto_parse)
    if body.reducto_chunk_mode is not None:
        _write("REDUCTO_CHUNK_MODE", body.reducto_chunk_mode)
    if body.reducto_chunk_size is not None:
        _write("REDUCTO_CHUNK_SIZE", str(body.reducto_chunk_size))
    if body.reducto_filter_blocks is not None:
        _write("REDUCTO_FILTER_BLOCKS", body.reducto_filter_blocks)
    if body.reducto_table_format is not None:
        _write("REDUCTO_TABLE_FORMAT", body.reducto_table_format)
    if body.chunk_max_chars is not None:
        _write("CHUNK_MAX_CHARS", str(body.chunk_max_chars))
    if body.chunk_overlap is not None:
        _write("CHUNK_OVERLAP", str(body.chunk_overlap))

    # ── Indexing ──────────────────────────────────────────────────────────────
    if body.embed_model is not None:
        _write("EMBED_MODEL", body.embed_model)
    if body.embed_dim is not None:
        _write("EMBED_DIM", body.embed_dim)
    if body.enable_doc_summary is not None:
        _write("ENABLE_DOC_SUMMARY", _bool_str(body.enable_doc_summary))
    if body.doc_summary_chars is not None:
        _write("DOC_SUMMARY_CHARS", str(body.doc_summary_chars))
    if body.doc_summary_model is not None:
        _write("DOC_SUMMARY_MODEL", body.doc_summary_model)
    if body.enable_section_summary is not None:
        _write("ENABLE_SECTION_SUMMARY", _bool_str(body.enable_section_summary))
    if body.section_summary_chars is not None:
        _write("SECTION_SUMMARY_CHARS", str(body.section_summary_chars))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    if body.bm25_top_k is not None:
        _write("BM25_TOP_K", str(body.bm25_top_k))
    if body.dense_top_k is not None:
        _write("DENSE_TOP_K", str(body.dense_top_k))
    if body.rrf_k is not None:
        _write("RRF_K", str(body.rrf_k))
    if body.use_reranker is not None:
        _write("USE_RERANKER", _bool_str(body.use_reranker))
    if body.reranker_type is not None:
        _write("RERANKER_TYPE", body.reranker_type)
    if body.reranker_model is not None:
        _write("RERANKER_MODEL", body.reranker_model)
    if body.jina_api_key is not None and not _is_masked(body.jina_api_key):
        _write("JINA_API_KEY", body.jina_api_key)
    if body.rerank_top_n is not None:
        _write("RERANK_TOP_N", str(body.rerank_top_n))
    if body.final_top_k is not None:
        _write("FINAL_TOP_K", str(body.final_top_k))
    if body.use_hyde is not None:
        _write("USE_HYDE", _bool_str(body.use_hyde))
    if body.use_rse is not None:
        _write("USE_RSE", _bool_str(body.use_rse))
    if body.rse_irrelevant_penalty is not None:
        _write("RSE_IRRELEVANT_PENALTY", str(body.rse_irrelevant_penalty))
    if body.rse_max_segment_chunks is not None:
        _write("RSE_MAX_SEGMENT_CHUNKS", str(body.rse_max_segment_chunks))
    if body.rse_overall_max_chunks is not None:
        _write("RSE_OVERALL_MAX_CHUNKS", str(body.rse_overall_max_chunks))
    if body.rse_window_extension is not None:
        _write("RSE_WINDOW_EXTENSION", str(body.rse_window_extension))
    if body.rse_chunk_length_adjustment is not None:
        _write("RSE_CHUNK_LENGTH_ADJUSTMENT", _bool_str(body.rse_chunk_length_adjustment))
    if body.complex_ctx_limit is not None:
        _write("COMPLEX_CTX_LIMIT", str(body.complex_ctx_limit))
    if body.min_chunk_chars is not None:
        _write("MIN_CHUNK_CHARS", str(body.min_chunk_chars))

    # ── Generation ────────────────────────────────────────────────────────────
    if body.llm_model is not None:
        _write("LLM_MODEL", body.llm_model)
    if body.llm_model_fast is not None:
        _write("LLM_MODEL_FAST", body.llm_model_fast)
    if body.enable_replan is not None:
        _write("ENABLE_REPLAN", _bool_str(body.enable_replan))
    if body.max_replan_iters is not None:
        _write("MAX_REPLAN_ITERS", str(body.max_replan_iters))
    if body.enable_sufficiency is not None:
        _write("ENABLE_SUFFICIENCY", _bool_str(body.enable_sufficiency))
    if body.enable_answer_verify is not None:
        _write("ENABLE_ANSWER_VERIFY", _bool_str(body.enable_answer_verify))
    if body.enable_answer_verify_simple is not None:
        _write("ENABLE_ANSWER_VERIFY_SIMPLE", _bool_str(body.enable_answer_verify_simple))
    if body.enable_answer_verify_complex is not None:
        _write("ENABLE_ANSWER_VERIFY_COMPLEX", _bool_str(body.enable_answer_verify_complex))
    if body.max_answer_regenerations is not None:
        _write("MAX_ANSWER_REGENERATIONS", str(body.max_answer_regenerations))

    # ── Memory ────────────────────────────────────────────────────────────────
    if body.enable_summarization is not None:
        _write("ENABLE_SUMMARIZATION", _bool_str(body.enable_summarization))
    if body.summary_threshold is not None:
        _write("SUMMARY_THRESHOLD", str(body.summary_threshold))
    if body.history_window is not None:
        _write("HISTORY_WINDOW", str(body.history_window))
    if body.summary_model is not None:
        _write("SUMMARY_MODEL", body.summary_model)

    # ── Hot-reload Settings singleton ─────────────────────────────────────────
    if changed:
        import importlib
        from dotenv import load_dotenv
        load_dotenv(str(env_path), override=True)
        get_settings.cache_clear()
        new_settings = get_settings()
        # Update module-level references
        import sys
        this_module = sys.modules[__name__]
        this_module.settings = new_settings
        kb.settings = new_settings

    return {"updated": list(changed.keys()), "needs_reindex": needs_reindex}


# ---------------- documents ----------------
@app.get("/api/documents")
def list_documents():
    return kb.repo.list_documents()


@app.get("/api/documents/{doc_id}/pdf")
def view_document_pdf(doc_id: int):
    doc = kb.repo.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Không tìm thấy tài liệu.")
    if doc["source_type"] != "pdf":
        raise HTTPException(404, "Tài liệu này không phải PDF.")

    path = kb.pdf_path(doc_id)
    if not path.exists():
        raise HTTPException(
            404,
            "PDF gốc chưa được lưu. Hãy tải lại file PDF để có thể xem trực tiếp.",
        )

    filename = doc["source"] or f"{doc['title']}.pdf"
    encoded_filename = quote(filename, safe="")
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="document.pdf"; filename*=UTF-8\'\'{encoded_filename}'
            ),
        },
    )


@app.post("/api/documents/upload")
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > settings.max_upload_size:
        raise HTTPException(
            413,
            f"File quá lớn ({len(data) // 1024 // 1024}MB). Tối đa {settings.max_upload_size // 1024 // 1024}MB.",
        )
    if not data[:5].startswith(b"%PDF"):
        raise HTTPException(400, "File không phải PDF hợp lệ.")
    name = file.filename or "document.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Chỉ hỗ trợ tệp PDF qua kênh upload.")

    doc_id = kb.repo.add_document(name, name, "pdf", status="processing")

    def _ingest_bg():
        try:
            # Check if document still exists before processing
            doc = kb.repo.get_document(doc_id)
            if not doc:
                return None  # User cancelled (deleted)
            result = kb.ingest_pdf(data, name, doc_id=doc_id)
            # Check again before updating status
            doc = kb.repo.get_document(doc_id)
            if doc:
                kb.repo.update_document_status(doc_id, "ready")
            return result
        except Exception as e:
            try:
                doc = kb.repo.get_document(doc_id)
                if doc:
                    kb.repo.update_document_status(doc_id, "failed", error_message=str(e))
            except Exception:
                pass  # Document already deleted, ignore
            logging.getLogger("rag.flow").exception("RAG_FLOW ingest_error doc_id=%s", doc_id)
            return None

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _ingest_bg)

    return {
        "document_id": doc_id,
        "title": name,
        "source": name,
        "source_type": "pdf",
        "status": "processing",
        "n_chunks": 0,
    }


@app.post("/api/documents/url")
@limiter.limit("10/minute")
def ingest_url(request: Request, body: UrlIn):
    try:
        return kb.ingest_url(body.url)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/documents/text")
def ingest_text(body: TextIn):
    try:
        return kb.ingest_text(body.text, body.title or "Văn bản")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    kb.delete_document(doc_id)
    return {"deleted": doc_id}


# ---------------- sessions ----------------
@app.get("/api/sessions")
def list_sessions():
    return kb.repo.list_sessions()


@app.post("/api/sessions")
def create_session(body: SessionIn):
    sid = kb.repo.create_session(body.title)
    return {"id": sid}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    summary_row = kb.repo.get_summary(sid) if settings.enable_summarization else None
    return {
        "id": sid,
        "messages": kb.repo.get_messages(sid),
        "summary": summary_row["summary"] if summary_row else None,
    }


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    kb.repo.delete_session(sid)
    return {"deleted": sid}


# ---------------- chat (SSE via StreamingResponse + background processing) ----------------
_active_streams: dict[str, asyncio.Queue] = {}
_cancelled_sessions: set[str] = set()


def _sse_event(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _run_agent_sync(
    session_id: str,
    msg_id: int,
    question: str,
    history: list[dict],
    summary: str | None,
    queue: asyncio.Queue,
) -> None:
    agent = Agent(kb)
    final_payload: dict = {"answer": "", "citations": [], "trace": []}
    trace_snapshot: list[dict] = []
    cancelled = False

    try:
        gen = agent.run(question, history, summary=summary)

        while True:
            # Cooperative cancellation check
            if session_id in _cancelled_sessions:
                cancelled = True
                break

            try:
                ev = next(gen)
            except StopIteration:
                break

            try:
                queue.put_nowait(ev)
            except Exception:
                pass

            # Persist trace events incrementally so UI survives interrupt/reload
            if ev["type"] not in ("token", "final"):
                trace_snapshot.append(ev)
                # Strip "thinking" events for DB storage (they're UI-only)
                clean_trace = [e for e in trace_snapshot if e.get("type") != "thinking"]
                kb.repo.update_message(msg_id, trace=clean_trace)

            if ev["type"] == "final":
                final_payload.update(ev["data"])
                answer_text = ev["data"].get("answer", "")
                if answer_text:
                    kb.repo.update_message(
                        msg_id,
                        content=answer_text,
                        citations=ev["data"].get("citations", []),
                        trace=[e for e in ev["data"].get("trace", []) if e.get("type") != "thinking"],
                        status="complete",
                    )
                    kb.repo.touch_session(session_id)
                    if settings.enable_summarization:
                        _maybe_summarize(session_id, agent, kb)

        if cancelled:
            existing = kb.repo.get_message(msg_id)
            if existing and existing.get("status") == "processing":
                kb.repo.update_message(msg_id, status="cancelled", error_message="Đã hủy bởi người dùng.")
            kb.repo.touch_session(session_id)
        elif not final_payload.get("answer"):
            existing = kb.repo.get_message(msg_id)
            if existing and existing.get("status") == "processing":
                kb.repo.update_message(msg_id, status="failed", error_message="Agent không tạo được câu trả lời.")

    except Exception as e:
        logging.getLogger("rag.flow").exception("RAG_FLOW agent_error session=%s msg=%s", session_id, msg_id)
        try:
            queue.put_nowait({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass
        existing = kb.repo.get_message(msg_id)
        if existing and existing.get("status") == "processing":
            kb.repo.update_message(msg_id, content=f"Lỗi: {e}", status="failed", error_message=str(e))
    finally:
        try:
            queue.put_nowait(None)
        except Exception:
            pass
        _active_streams.pop(session_id, None)
        _cancelled_sessions.discard(session_id)


@app.post("/api/chat")
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatIn):
    if not settings.has_openai:
        raise HTTPException(400, "OPENAI_API_KEY chưa được cấu hình trong backend/.env")

    sid = body.session_id
    if not sid:
        sid = kb.repo.create_session()
    else:
        kb.repo.ensure_session(sid)

    processing = kb.repo.get_processing_messages(sid)
    if processing:
        raise HTTPException(409, "Vui lòng đợi câu trả lời trước khi gửi tin nhắn mới.")

    history = kb.repo.recent_history(sid, limit=settings.history_window)

    summary = None
    if settings.enable_summarization:
        summary_row = kb.repo.get_summary(sid)
        if summary_row:
            summary = summary_row["summary"]

    kb.repo.add_message(sid, "user", body.message)
    if len(history) == 0:
        kb.repo.touch_session(sid, title=body.message[:60])

    msg_id = kb.repo.add_message(sid, "assistant", "", status="processing")

    queue: asyncio.Queue = asyncio.Queue()
    _active_streams[sid] = queue

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None,
        _run_agent_sync,
        sid,
        msg_id,
        body.message,
        history,
        summary,
        queue,
    )

    async def stream():
        yield _sse_event("session", {"session_id": sid})
        if summary:
            yield _sse_event("summary", {"has_summary": True})

        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=25)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if ev is None:
                    break

                yield _sse_event(ev["type"], ev["data"])
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Pragma": "no-cache",
        },
    )


@app.post("/api/chat/cancel")
@limiter.limit("30/minute")
async def cancel_chat(request: Request, body: CancelChatIn):
    sid = body.session_id
    if not sid:
        raise HTTPException(400, "Thiếu session_id.")

    processing = kb.repo.get_processing_messages(sid)
    if not processing:
        return {"cancelled": False, "message": "Không có câu trả lời nào đang xử lý."}

    _cancelled_sessions.add(sid)

    # Mark all processing messages as cancelled immediately
    for msg in processing:
        kb.repo.update_message(msg["id"], status="cancelled", error_message="Đã hủy bởi người dùng.")

    return {"cancelled": True, "message": "Đã hủy."}
