## Mục tiêu

Cả nhóm thực tập sinh cùng **xây dựng và demo** một ứng dụng RAG Agent hoàn chỉnh có khả năng:

1. **Nhận tài liệu đầu vào** (PDF, file text, hoặc URL website)
2. **Trả lời câu hỏi** dựa trên nội dung tài liệu đã đưa vào
3. **Sử dụng hybrid search** kết hợp keyword search (BM25) + vector search (dense retrieval)
4. **Trích dẫn nguồn** — mỗi câu trả lời phải chỉ rõ đoạn nào trong tài liệu nào

> **Tham khảo:** [Google NotebookLM](https://notebooklm.google.com/) — user upload tài liệu, hệ thống trả lời dựa trên nội dung đã upload, không hallucinate ngoài scope.

## Yêu cầu chi tiết

### 1. Ingestion Pipeline

- Hỗ trợ **ít nhất 2 loại nguồn**: PDF + URL (hoặc plain text)
- Parse tài liệu → tách chunk có metadata (source, page/section, chunk_id)
- Tạo **2 loại index**:
  - BM25 index (keyword-based)
  - Vector index (embedding-based)

### 2. Retrieval Pipeline (Hybrid Search)

- **BM25 retrieval**: lấy top-k chunks theo keyword matching
- **Dense retrieval**: lấy top-k chunks theo vector similarity
- **Fusion**: kết hợp kết quả từ cả 2 (gợi ý: Reciprocal Rank Fusion — RRF)

### 3. Generation

- Gửi top chunks vào LLM để sinh câu trả lời
- **Bắt buộc**: câu trả lời phải dựa trên retrieved chunks, **không được tự suy diễn**
- **Bắt buộc**: kèm citation (trích nguồn tài liệu, trang, đoạn)
- Nếu không tìm thấy thông tin → trả lời rõ "không có trong tài liệu"

### 4. App & UI

- Xây dựng **ứng dụng demo** với giao diện chat (Streamlit, Gradio, hoặc web app)
- User có thể upload tài liệu / nhập URL và hỏi đáp trực tiếp trên app
- Agent phải **trả lời được thực sự** — không chỉ là pipeline chạy trên script

### 5. Evaluation

- Chuẩn bị **tối thiểu 10 câu hỏi test** trên bộ tài liệu đã ingest
- Đo và báo cáo ít nhất: **Recall@5** và **MRR@5**
- Phân tích kết quả: case nào pipeline trả lời tốt, case nào chưa tốt, tại sao

## Deliverables

| # | Deliverable | Bắt buộc |
|---|------------|----------|
| 1 | **App demo chạy được** — ứng dụng có UI, Agent trả lời được thực sự | ✅ |
| 2 | **Hybrid search** — BM25 + Dense + Fusion hoạt động | ✅ |
| 3 | **Evaluation report** — 10+ câu hỏi, Recall@5, MRR@5 | ✅ |
| 4 | **Citations** — mỗi câu trả lời kèm nguồn | ✅ |
| 5 | **README** — hướng dẫn cài đặt & chạy lại | ✅ |
| 6 | **Slides trình bày** — 5-10 slides cho buổi demo | ✅ |
| 7 | Hỗ trợ tiếng Việt tốt | ⭐ Bonus |
| 8 | Advanced method (SPLADE, ColBERT, HyDE...) | ⭐ Bonus |
| 9 | Failure analysis (phân tích case pipeline trả lời sai) | ⭐ Bonus |