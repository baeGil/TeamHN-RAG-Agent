# RAG Agent Tiếng Việt — Production-grade Hybrid RAG

Hệ thống **RAG Agent** hỏi đáp dựa trên tài liệu tiếng Việt (kiểu NotebookLM): nạp PDF/URL/văn bản,
trả lời câu hỏi **kèm trích dẫn nguồn**, **không bịa đặt ngoài tài liệu**, hiển thị **quá trình suy luận
của agent theo thời gian thực**.

- **Hybrid search**: BM25 (keyword, tách từ tiếng Việt bằng `underthesea`) + Dense (embedding OpenAI)
  + **Reciprocal Rank Fusion (RRF)** + **cross-encoder reranker** (`bge-reranker-v2-m3`).
- **Vector index**: [`turbovec`](https://github.com/RyanCodrai/turbovec) — index dựa trên thuật toán
  **TurboQuant** (ICLR 2026), nén 16×, SIMD → latency truy hồi cực thấp.
- **Agent điều khiển được** (kế thừa ý tưởng từ
  [NirDiamant/Controllable-RAG-Agent](https://github.com/NirDiamant/Controllable-RAG-Agent)):
  **Adaptive Router** → single-hop đi đường nhanh, multi-hop chạy đồ thị suy luận tất định
  (plan → decompose → retrieve → distill → **verify groundedness** → synthesize).
- **UI**: Vite + React + TypeScript — chat, upload tài liệu, **render LaTeX**, **citation hover + click**,
  **lưu phiên (reload không mất dữ liệu)**.

---

## 1. Kiến trúc

```
                ┌──────────────────────── Frontend (Vite + React + TS) ───────────────────────┐
                │  Chat · Upload PDF/URL · Agent Trace realtime · LaTeX · Citation hover/click │
                └───────────────▲───────────────────────────────────────────────┬─────────────┘
                                │ REST + SSE (stream)                            │
                ┌───────────────┴────────────────── Backend (FastAPI) ───────────▼─────────────┐
                │  Ingestion      Indexing            Retrieval            Agent (control graph)│
                │  PDF/URL/text → chunk+meta → BM25  → RRF fusion  → rerank → route → plan →     │
                │  underthesea          + turbovec                          retrieve → distill → │
                │                       (TurboQuant)                        verify → synthesize  │
                │  SQLite: documents · chunks · sessions · messages (persistence)               │
                └──────────────────────────────────────────────────────────────────────────────┘
```

Luồng agent (đồ thị tất định — "bộ não" điều khiển được):

```
route ─┬─ no_retrieval ─────────────────────────────► answer (chitchat)
       ├─ simple  ─► retrieve ─────────────────► synthesize ─► answer + citations
       └─ complex ► plan ► (retrieve ► distill ► verify)* ► synthesize ─► answer + citations
```

Mỗi node phát một sự kiện qua **SSE** để UI theo dõi realtime; câu trả lời cuối được **stream token**.

---

## 2. Cài đặt

### Yêu cầu
- Python 3.10+ (đã test 3.11), Node 18+ (đã test 23), macOS/Linux (arm64 hoặc x86-64).
- Một OpenAI API key.

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # tuỳ chọn
pip install -r requirements.txt

cp .env.example .env
# Mở .env và điền:
#   OPENAI_API_KEY=sk-...
#   LLM_MODEL=...          (vd: gpt-4o)
#   LLM_MODEL_FAST=...     (vd: gpt-4o-mini)
#   EMBED_MODEL=...        (vd: text-embedding-3-small)
```

### Frontend
```bash
cd frontend
npm install
```

---

## 3. Chạy

Cách nhanh nhất (chạy cả hai):
```bash
./run.sh
```

Hoặc chạy thủ công ở 2 terminal:
```bash
# Terminal 1 — backend
cd backend && uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Mở **http://localhost:5173**. Tải lên PDF (vd `test/mô_hình_hóa_ĐATN.pdf`) hoặc dán URL, rồi đặt câu hỏi.

> Lần đầu dùng reranker, model `bge-reranker-v2-m3` (~2GB) sẽ được tải tự động. Có thể tắt bằng
> `USE_RERANKER=false` trong `.env` (vẫn còn BM25 + Dense + RRF).

---

## 4. Đánh giá (Recall@5, MRR@5)

Bộ test có sẵn trong `test/`: `testQA.md` (10 câu single-hop) và `testQA_hard.md` (10 câu suy luận).

```bash
cd backend
python -m eval.run_eval --test-dir ../test
```

Script sẽ:
1. Tự nạp PDF trong `test/` (vào kho riêng `storage_eval`).
2. Với mỗi câu hỏi, sinh ứng viên từ 4 cấu hình: **BM25 / Dense / Hybrid (RRF) / Hybrid + Rerank**.
3. Gộp ứng viên và dùng **LLM-as-judge** để xác định tập đoạn liên quan (pooling kiểu TREC).
4. Tính **Recall@5** và **MRR@5** (kèm Hit@5) cho từng cấu hình, theo tổng thể và theo độ khó.
5. Ghi báo cáo: `data/eval/eval_report.md` + `data/eval/eval_results.json`, kèm **phân tích lỗi**.

---

## 5. Cấu trúc dự án

```
backend/
  app/
    config.py                 # đọc .env
    main.py                   # FastAPI: /documents, /sessions, /chat (SSE), /stats
    db/                       # SQLite (documents, chunks, sessions, messages)
    ingestion/                # loaders (PDF/URL/text) · chunker · vn_text (underthesea)
    indexing/                 # embeddings (OpenAI) · vector_index (turbovec) · bm25 · store
    retrieval/                # hybrid (RRF) · reranker (bge-reranker-v2-m3)
    agent/                    # graph (control graph) · nodes · prompts · llm
  eval/                       # dataset · metrics · run_eval
frontend/
  src/components/             # ChatPanel · Sidebar · AgentTrace · Markdown (LaTeX + citations)
  src/lib/                    # api (REST + SSE) · types
test/                         # PDF + 2 bộ câu hỏi test
data/eval/                    # báo cáo đánh giá (sinh ra khi chạy eval)
slides/outline.md             # đề cương slide demo
```

---

## 6. Tối ưu chi phí token & latency
- **Adaptive Router**: câu đơn giản chỉ tốn 1 lần gọi LLM trả lời; chỉ câu phức tạp mới chạy agent đầy đủ.
- **Reranker chạy local** (cross-encoder) → tăng precision mà **không tốn token**.
- **turbovec**: nén 16×, SIMD → search dưới mili-giây ngay cả khi corpus lớn.
- **Cache embedding** trên đĩa; **batch** khi nạp tài liệu; dùng model rẻ (`LLM_MODEL_FAST`) cho
  router/plan/distill/verify, model mạnh chỉ cho câu trả lời cuối.
- **Distillation** ngữ cảnh ở nhánh multi-hop để giảm token đưa vào LLM.

## 7. Chống bịa đặt (anti-hallucination)
- Prompt ràng buộc trả lời **chỉ** dựa trên ngữ cảnh; nếu không có → trả đúng
  `"Không tìm thấy thông tin trong tài liệu."`.
- Nhánh multi-hop có node **verify groundedness** (kiểu Self-RAG) kiểm tra ghi chú chắt lọc có
  bám nguồn không trước khi tổng hợp.
- Mọi khẳng định kèm trích dẫn `[số]` ánh xạ tới đoạn nguồn (xem được khi hover/click).

---

## 8. Ánh xạ Deliverables (theo `task.md`)

| # | Deliverable | Trạng thái |
|---|---|---|
| 1 | App demo có UI, agent trả lời thật | ✅ React + FastAPI, SSE realtime |
| 2 | Hybrid search BM25 + Dense + Fusion | ✅ BM25 + turbovec + RRF (+ rerank) |
| 3 | Evaluation report (10+ câu, Recall@5, MRR@5) | ✅ `eval/run_eval.py` → `data/eval/` |
| 4 | Citations (nguồn, trang, đoạn) | ✅ hover preview + click drawer |
| 5 | README cài đặt & chạy lại | ✅ tài liệu này |
| 6 | Slides trình bày | ✅ `slides/outline.md` |
| 7 | Hỗ trợ tiếng Việt tốt | ⭐ underthesea + embedding đa ngôn ngữ |
| 8 | Advanced method | ⭐ cross-encoder reranker (HyDE tuỳ chọn) |
| 9 | Failure analysis | ⭐ trong báo cáo eval |
