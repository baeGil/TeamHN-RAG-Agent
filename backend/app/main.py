import asyncio
import json
import logging
import os
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.responses import StreamingResponse

from .agent.graph import Agent
from .config import get_settings
from .indexing.store import KnowledgeBase
from .schemas import ChatIn, SessionIn, TextIn, UrlIn

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
app = FastAPI(title="Vietnamese RAG Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "reranker_model": settings.reranker_model,
    }


@app.get("/api/stats")
def stats():
    return kb.stats()


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
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "document.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Chỉ hỗ trợ tệp PDF qua kênh upload.")
    try:
        return kb.ingest_pdf(data, name)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/documents/url")
def ingest_url(body: UrlIn):
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


# ---------------- chat (SSE via StreamingResponse) ----------------
def _sse_event(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatIn):
    if not settings.has_openai:
        raise HTTPException(400, "OPENAI_API_KEY chưa được cấu hình trong backend/.env")

    sid = body.session_id
    if not sid:
        sid = kb.repo.create_session()
    else:
        kb.repo.ensure_session(sid)

    history = kb.repo.recent_history(sid, limit=settings.history_window)

    summary = None
    if settings.enable_summarization:
        summary_row = kb.repo.get_summary(sid)
        if summary_row:
            summary = summary_row["summary"]

    kb.repo.add_message(sid, "user", body.message)
    if len(history) == 0:
        kb.repo.touch_session(sid, title=body.message[:60])

    agent = Agent(kb)
    final_payload = {"answer": "", "citations": [], "trace": []}

    async def stream():
        yield _sse_event("session", {"session_id": sid})
        if summary:
            yield _sse_event("summary", {"has_summary": True})

        gen = agent.run(body.message, history, summary=summary)
        gen_exhausted = False

        def _next_event():
            try:
                return next(gen)
            except StopIteration:
                return None

        loop = asyncio.get_running_loop()

        try:
            pending_next = loop.run_in_executor(None, _next_event)

            while not gen_exhausted:
                try:
                    ev = await asyncio.wait_for(
                        asyncio.shield(pending_next),
                        timeout=25,
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if ev is None:
                    gen_exhausted = True
                    break

                pending_next = loop.run_in_executor(None, _next_event)

                if ev["type"] == "final":
                    final_payload.update(ev["data"])
                    # Persist assistant message immediately on "final" so it
                    # survives a client disconnect mid-stream.
                    answer_text = ev["data"].get("answer", "")
                    if answer_text:
                        kb.repo.add_message(
                            sid,
                            "assistant",
                            answer_text,
                            citations=ev["data"].get("citations", []),
                            trace=[e for e in ev["data"].get("trace", []) if e.get("type") != "thinking"],
                        )
                        kb.repo.touch_session(sid)

                        if settings.enable_summarization:
                            _maybe_summarize(sid, agent, kb)
                yield _sse_event(ev["type"], ev["data"])
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
            final_payload["answer"] = f"Lỗi: {e}"

        # Fallback: persist error messages or answers that bypassed "final".
        if final_payload.get("answer") and not any(
            m["role"] == "assistant" and m["content"] == final_payload["answer"]
            for m in kb.repo.get_messages(sid)
        ):
            kb.repo.add_message(
                sid,
                "assistant",
                final_payload["answer"],
                citations=final_payload.get("citations", []),
                trace=[e for e in final_payload.get("trace", []) if e.get("type") != "thinking"],
            )
            kb.repo.touch_session(sid)

            if settings.enable_summarization:
                _maybe_summarize(sid, agent, kb)

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
