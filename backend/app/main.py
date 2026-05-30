import asyncio
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from .agent.graph import Agent
from .config import get_settings
from .indexing.store import KnowledgeBase
from .schemas import ChatIn, SessionIn, TextIn, UrlIn

settings = get_settings()
app = FastAPI(title="Vietnamese RAG Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

kb = KnowledgeBase(settings)


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
    return {"id": sid, "messages": kb.repo.get_messages(sid)}


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

    history = kb.repo.recent_history(sid, limit=6)
    kb.repo.add_message(sid, "user", body.message)
    if len(history) == 0:
        kb.repo.touch_session(sid, title=body.message[:60])

    agent = Agent(kb)
    final_payload = {"answer": "", "citations": [], "trace": []}

    async def stream():
        yield _sse_event("session", {"session_id": sid})

        gen = agent.run(body.message, history)
        gen_exhausted = False

        def _next_event():
            try:
                return next(gen)
            except StopIteration:
                return None

        loop = asyncio.get_running_loop()

        try:
            while not gen_exhausted:
                try:
                    ev = await asyncio.wait_for(
                        loop.run_in_executor(None, _next_event),
                        timeout=25,
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if ev is None:
                    gen_exhausted = True
                    break

                if ev["type"] == "final":
                    final_payload.update(ev["data"])
                yield _sse_event(ev["type"], ev["data"])
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
            final_payload["answer"] = f"Lỗi: {e}"

        kb.repo.add_message(
            sid,
            "assistant",
            final_payload.get("answer", ""),
            citations=final_payload.get("citations", []),
            trace=[e for e in final_payload.get("trace", []) if e.get("type") != "thinking"],
        )
        kb.repo.touch_session(sid)

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