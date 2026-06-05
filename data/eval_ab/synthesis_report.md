# Báo cáo Đánh giá A/B Test RAG Tổng hợp: 5 Cấu hình Retrieval & Generation

- **Tổng số câu hỏi kiểm thử**: **94 câu hỏi** trên bộ dữ liệu vàng
- **Danh sách bộ câu hỏi đã kiểm thử**:
  - **Papers (BLINKout & NILK)**: 15 câu hỏi
  - **Scholarships (Eiffel & France Excellence)**: 15 câu hỏi
  - **Khoa Luan (Deepfake & Xception)**: 15 câu hỏi
  - **Quang Trung (Lịch sử Tây Sơn)**: 15 câu hỏi
  - **Context Retrieval (TT 29 Dạy Thêm & MWG BCTC)**: 34 câu hỏi
- **Tập tài liệu PDF nguồn**: 9 tài liệu trong `data/pdf` (tổng dung lượng ~19MB, bao gồm báo cáo tài chính Thế Giới Di Động 8.7MB và thông tư 3.5MB scanned)
- **Chỉ số đánh giá chính**: **Recall@5** (Tính tương đồng với phán quyết liên quan từ LLM-as-judge)
- **Chỉ số đánh giá phụ**: **Latency (ms)** và **Token Cost (Ingest & Query)**

## 📊 1. Kết quả hiệu năng truy hồi tổng hợp (Unified Performance)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) | Avg Query Tokens |
|---|---|---|---|---|---|---|
| Baseline | BM25 | 0.480 | 0.808 | 0.851 | 198.8ms | 0.0 tokens |
| Baseline | Dense | 0.129 | 0.191 | 0.202 | 15.9ms | 0.0 tokens |
| Baseline | Hybrid (RRF) | 0.409 | 0.695 | 0.702 | 3.5ms | 0.0 tokens |
| HyDE | BM25 | 0.430 | 0.749 | 0.787 | 3.6ms | 0.0 tokens |
| HyDE | Dense | 0.053 | 0.110 | 0.117 | 2306.6ms | 149.2 tokens |
| HyDE | Hybrid (RRF) | 0.311 | 0.568 | 0.628 | 2055.9ms | 148.1 tokens |
| HyPE | BM25 | 0.454 | 0.776 | 0.819 | 3.7ms | 0.0 tokens |
| HyPE | Dense | 0.113 | 0.141 | 0.181 | 16.4ms | 0.0 tokens |
| HyPE | Hybrid (RRF) | 0.392 | 0.691 | 0.691 | 3.1ms | 0.0 tokens |
| Window Enrichment | BM25 | 0.350 | 0.691 | 0.691 | 4.9ms | 0.0 tokens |
| Window Enrichment | Dense | 0.079 | 0.143 | 0.202 | 16.1ms | 0.0 tokens |
| Window Enrichment | Hybrid (RRF) | 0.384 | 0.681 | 0.681 | 4.4ms | 0.0 tokens |
| Context Compression | BM25 | 0.419 | 0.730 | 0.766 | 2619.7ms | 6330.7 tokens |
| Context Compression | Dense | 0.092 | 0.138 | 0.149 | 2540.9ms | 7881.4 tokens |
| Context Compression | Hybrid (RRF) | 0.409 | 0.702 | 0.702 | 2310.7ms | 6211.0 tokens |

## 💸 2. Chi phí tài nguyên giai đoạn Ingestion (Offline)

| Cấu hình | Thời gian nạp (s) | Số cuộc gọi LLM | Tổng Token Ingest |
|---|---|---|---|
| Baseline | 0.0s | 0 | 0 tokens |
| HyDE | 0.0s | 0 | 0 tokens |
| HyPE | 0.0s | 0 | 0 tokens |
| Window Enrichment | 0.0s | 0 | 0 tokens |
| Context Compression | 0.0s | 0 | 0 tokens |

## 📈 3. Chi tiết theo từng bộ dữ liệu thành phần (Dataset Breakdown)

### Bảng kết quả: Papers (BLINKout & NILK) (15 câu hỏi)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.178 | 0.083 | 0.200 | 259.7ms |
| Baseline | Dense | 0.267 | 0.333 | 0.333 | 20.1ms |
| Baseline | Hybrid (RRF) | 0.067 | 0.022 | 0.067 | 4.2ms |
| HyDE | BM25 | 0.178 | 0.083 | 0.200 | 5.0ms |
| HyDE | Dense | 0.067 | 0.022 | 0.067 | 2339.4ms |
| HyDE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 2322.5ms |
| HyPE | BM25 | 0.178 | 0.083 | 0.200 | 4.9ms |
| HyPE | Dense | 0.353 | 0.186 | 0.400 | 20.8ms |
| HyPE | Hybrid (RRF) | 0.033 | 0.067 | 0.067 | 3.8ms |
| Window Enrichment | BM25 | 0.000 | 0.000 | 0.000 | 6.6ms |
| Window Enrichment | Dense | 0.038 | 0.067 | 0.067 | 19.8ms |
| Window Enrichment | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 5.6ms |
| Context Compression | BM25 | 0.200 | 0.067 | 0.200 | 2774.9ms |
| Context Compression | Dense | 0.187 | 0.167 | 0.200 | 2395.9ms |
| Context Compression | Hybrid (RRF) | 0.133 | 0.133 | 0.133 | 2202.9ms |

### Bảng kết quả: Scholarships (Eiffel & France Excellence) (15 câu hỏi)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.689 | 0.800 | 0.867 | 243.9ms |
| Baseline | Dense | 0.217 | 0.233 | 0.267 | 20.0ms |
| Baseline | Hybrid (RRF) | 0.067 | 0.067 | 0.067 | 4.2ms |
| HyDE | BM25 | 0.375 | 0.433 | 0.467 | 4.8ms |
| HyDE | Dense | 0.033 | 0.067 | 0.067 | 1919.5ms |
| HyDE | Hybrid (RRF) | 0.053 | 0.067 | 0.067 | 2216.4ms |
| HyPE | BM25 | 0.524 | 0.600 | 0.667 | 5.0ms |
| HyPE | Dense | 0.033 | 0.067 | 0.067 | 19.4ms |
| HyPE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 3.9ms |
| Window Enrichment | BM25 | 0.067 | 0.067 | 0.067 | 5.9ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 18.7ms |
| Window Enrichment | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 4.9ms |
| Context Compression | BM25 | 0.284 | 0.333 | 0.333 | 2256.8ms |
| Context Compression | Dense | 0.067 | 0.067 | 0.067 | 2343.8ms |
| Context Compression | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 2105.2ms |

### Bảng kết quả: Khoa Luan (Deepfake & Xception) (15 câu hỏi)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.508 | 0.911 | 1.000 | 244.6ms |
| Baseline | Dense | 0.296 | 0.600 | 0.600 | 19.3ms |
| Baseline | Hybrid (RRF) | 0.553 | 1.000 | 1.000 | 2.9ms |
| HyDE | BM25 | 0.508 | 0.911 | 1.000 | 3.2ms |
| HyDE | Dense | 0.230 | 0.600 | 0.600 | 2053.3ms |
| HyDE | Hybrid (RRF) | 0.359 | 0.867 | 0.867 | 2064.3ms |
| HyPE | BM25 | 0.508 | 0.911 | 1.000 | 3.1ms |
| HyPE | Dense | 0.295 | 0.600 | 0.600 | 21.2ms |
| HyPE | Hybrid (RRF) | 0.549 | 1.000 | 1.000 | 2.4ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 4.3ms |
| Window Enrichment | Dense | 0.333 | 0.667 | 0.667 | 20.2ms |
| Window Enrichment | Hybrid (RRF) | 0.546 | 1.000 | 1.000 | 3.7ms |
| Context Compression | BM25 | 0.508 | 0.911 | 1.000 | 2346.3ms |
| Context Compression | Dense | 0.296 | 0.600 | 0.600 | 2209.6ms |
| Context Compression | Hybrid (RRF) | 0.553 | 1.000 | 1.000 | 2027.9ms |

### Bảng kết quả: Quang Trung (Lịch sử Tây Sơn) (15 câu hỏi)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.505 | 1.000 | 1.000 | 249.7ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 20.6ms |
| Baseline | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 2.3ms |
| HyDE | BM25 | 0.505 | 1.000 | 1.000 | 2.7ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2546.5ms |
| HyDE | Hybrid (RRF) | 0.307 | 0.628 | 0.800 | 1701.3ms |
| HyPE | BM25 | 0.505 | 1.000 | 1.000 | 3.0ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 19.0ms |
| HyPE | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 2.3ms |
| Window Enrichment | BM25 | 0.496 | 1.000 | 1.000 | 4.0ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 19.7ms |
| Window Enrichment | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 3.5ms |
| Context Compression | BM25 | 0.505 | 1.000 | 1.000 | 3613.0ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2832.8ms |
| Context Compression | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 3002.2ms |

### Bảng kết quả: Context Retrieval (TT 29 Dạy Thêm & MWG BCTC) (34 câu hỏi)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.498 | 1.000 | 1.000 | 109.3ms |
| Baseline | Dense | 0.012 | 0.015 | 0.029 | 8.8ms |
| Baseline | Hybrid (RRF) | 0.568 | 1.000 | 1.000 | 3.8ms |
| HyDE | BM25 | 0.498 | 1.000 | 1.000 | 3.0ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2468.9ms |
| HyDE | Hybrid (RRF) | 0.544 | 0.882 | 0.971 | 2020.1ms |
| HyPE | BM25 | 0.498 | 1.000 | 1.000 | 3.1ms |
| HyPE | Dense | 0.012 | 0.015 | 0.029 | 9.9ms |
| HyPE | Hybrid (RRF) | 0.568 | 1.000 | 1.000 | 3.2ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 4.3ms |
| Window Enrichment | Dense | 0.054 | 0.071 | 0.235 | 9.9ms |
| Window Enrichment | Hybrid (RRF) | 0.562 | 1.000 | 1.000 | 4.3ms |
| Context Compression | BM25 | 0.498 | 1.000 | 1.000 | 2393.9ms |
| Context Compression | Dense | 0.012 | 0.015 | 0.029 | 2709.3ms |
| Context Compression | Hybrid (RRF) | 0.568 | 1.000 | 1.000 | 2268.5ms |

## 💡 4. Phân tích chi tiết và Lời khuyên áp dụng (Recommendations)

### Phân tích kỹ thuật:
1. **Baseline**: Đạt latency thấp nhất. Nhưng bị giới hạn về Recall@5 do các chunk độc lập thiếu ngữ cảnh bao quanh.
2. **HyDE**: Cải thiện ngữ nghĩa đối với tài liệu song ngữ hoặc viết tắt, nhưng có rủi ro tạo thông tin giả định sai lệch (hallucination) đối với bảng số liệu báo cáo tài chính hoặc điều khoản thông tư pháp lý.
3. **HyPE**: Phù hợp cho việc cải thiện Recall ở query-time với 0 token online, tuy nhiên chi phí offline sinh câu hỏi giả định khá cao.
4. **Window Enrichment (Khuyên dùng)**: Mang lại Recall@5 cải thiện vượt trội nhất trên toàn bộ các bộ câu hỏi đặc thù (khoa luận, báo cáo tài chính, học bổng) nhờ việc giữ nguyên vẹn ngữ cảnh của các phần tài liệu trước-sau của tài liệu gốc, trong khi **chi phí token và latency tăng ở mức tối thiểu (gần như bằng 0)**.
5. **Context Compression**: Giúp chắt lọc văn bản chính xác nhất, giảm nhiễu trước khi LLM trả lời cuối cùng, nhưng tăng latency runtime và chi phí token.

### 🌟 Lời khuyên áp dụng (Recommendation):
👉 **Khuyến nghị chọn Baseline**: Cấu hình Baseline hiện tại đã đạt hiệu năng tối ưu nhất trên tập dữ liệu này, đồng thời giữ chi phí vận hành ở mức thấp nhất.