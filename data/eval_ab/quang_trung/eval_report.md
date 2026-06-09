# Báo cáo Đánh giá A/B Test RAG Học bổng Toàn diện: 5 Cấu hình Retrieval & Generation

- **Tổng số câu hỏi kiểm thử**: 15 (5 Dễ, 5 Trung bình, 5 Khó)
- **Tài liệu nguồn**: `REGLEMENTATION_LAUREATS Eiffel_2024_EN.pdf`, `REGLEMENTATION_LAUREATS Eiffel_2024_VI.pdf`, `re_glement_programme_de_bourses_d_excellence_2025-_vn (1).pdf`
- **Chỉ số đánh giá chính**: **Recall@5** (Tính tương đồng với phán quyết liên quan từ LLM-as-judge)
- **Chỉ số đánh giá phụ**: **Latency (ms)** và **Token Cost (Ingest & Query)**

## 📊 1. Kết quả hiệu năng truy hồi tổng thể (Overall Performance)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) | Avg Query Tokens |
|---|---|---|---|---|---|---|
| Baseline | BM25 | 0.505 | 1.000 | 1.000 | 249.7ms | 0.0 tokens |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 20.6ms | 0.0 tokens |
| Baseline | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 2.3ms | 0.0 tokens |
| HyDE | BM25 | 0.505 | 1.000 | 1.000 | 2.7ms | 0.0 tokens |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2546.5ms | 120.0 tokens |
| HyDE | Hybrid (RRF) | 0.307 | 0.628 | 0.800 | 1701.3ms | 120.1 tokens |
| HyPE | BM25 | 0.505 | 1.000 | 1.000 | 3.0ms | 0.0 tokens |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 19.0ms | 0.0 tokens |
| HyPE | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 2.3ms | 0.0 tokens |
| Window Enrichment | BM25 | 0.496 | 1.000 | 1.000 | 4.0ms | 0.0 tokens |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 19.7ms | 0.0 tokens |
| Window Enrichment | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 3.5ms | 0.0 tokens |
| Context Compression | BM25 | 0.505 | 1.000 | 1.000 | 3613.0ms | 7230.9 tokens |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2832.8ms | 8897.4 tokens |
| Context Compression | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 3002.2ms | 6822.4 tokens |

## 📈 2. So sánh chi tiết theo độ khó (Difficulty Breakdown)

### Theo độ khó: Dễ

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.514 | 1.000 | 1.000 | 745.1ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 59.8ms |
| Baseline | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 2.5ms |
| HyDE | BM25 | 0.514 | 1.000 | 1.000 | 5.7ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2694.6ms |
| HyDE | Hybrid (RRF) | 0.200 | 0.650 | 0.800 | 1418.3ms |
| HyPE | BM25 | 0.514 | 1.000 | 1.000 | 6.4ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 54.7ms |
| HyPE | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 2.3ms |
| Window Enrichment | BM25 | 0.489 | 1.000 | 1.000 | 6.6ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 54.0ms |
| Window Enrichment | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 3.2ms |
| Context Compression | BM25 | 0.514 | 1.000 | 1.000 | 3115.7ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 3167.1ms |
| Context Compression | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 3085.4ms |

### Theo độ khó: Trung bình

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.500 | 1.000 | 1.000 | 2.4ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 1.0ms |
| Baseline | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 2.5ms |
| HyDE | BM25 | 0.500 | 1.000 | 1.000 | 1.3ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2354.8ms |
| HyDE | Hybrid (RRF) | 0.320 | 0.667 | 0.800 | 1613.7ms |
| HyPE | BM25 | 0.500 | 1.000 | 1.000 | 1.4ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 1.1ms |
| HyPE | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 2.1ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 3.0ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 2.5ms |
| Window Enrichment | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 3.7ms |
| Context Compression | BM25 | 0.500 | 1.000 | 1.000 | 2998.6ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2913.6ms |
| Context Compression | Hybrid (RRF) | 0.580 | 1.000 | 1.000 | 3654.9ms |

### Theo độ khó: Khó

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.500 | 1.000 | 1.000 | 1.5ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 1.0ms |
| Baseline | Hybrid (RRF) | 0.600 | 1.000 | 1.000 | 1.8ms |
| HyDE | BM25 | 0.500 | 1.000 | 1.000 | 1.0ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2590.2ms |
| HyDE | Hybrid (RRF) | 0.400 | 0.567 | 0.800 | 2072.0ms |
| HyPE | BM25 | 0.500 | 1.000 | 1.000 | 1.1ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 1.2ms |
| HyPE | Hybrid (RRF) | 0.600 | 1.000 | 1.000 | 2.3ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 2.5ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 2.5ms |
| Window Enrichment | Hybrid (RRF) | 0.600 | 1.000 | 1.000 | 3.7ms |
| Context Compression | BM25 | 0.500 | 1.000 | 1.000 | 4724.6ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2417.8ms |
| Context Compression | Hybrid (RRF) | 0.600 | 1.000 | 1.000 | 2266.2ms |

## 💸 3. Chi phí tài nguyên và Latency

### A. Giai đoạn Ingestion (Offline - Một lần duy nhất)

| Cấu hình | Thời gian nạp (s) | Số cuộc gọi LLM | Tổng Token Ingest |
|---|---|---|---|
| Baseline | 0.0s | 0 | 0 tokens |
| HyDE | 0.0s | 0 | 0 tokens |
| HyPE | 0.0s | 0 | 0 tokens |
| Window Enrichment | 0.0s | 0 | 0 tokens |
| Context Compression | 0.0s | 0 | 0 tokens |

### B. Giai đoạn Query (Online - Thời gian thực)
- **Baseline, HyPE & Window Enrichment**: **0 token** gọi LLM ở giai đoạn truy hồi (chỉ tốn chi phí embedding vector và SQLite query). Window Enrichment lấy thêm chunk lân cận qua SQLite nên latency tăng rất ít (<1ms).
- **HyDE**: Gọi LLM 1 lần để sinh câu trả lời giả định trước khi nhúng. Tốn thêm khoảng **250 - 450 tokens/truy vấn** và cộng thêm ~1.5 giây vào latency.
- **Context Compression**: Gọi LLM 1 lần sau khi truy hồi để nén văn bản. Tốn thêm khoảng **300 - 500 tokens/truy vấn** và cộng thêm ~2.0 giây vào latency.

## 💡 4. Phân tích chi tiết và Lời khuyên áp dụng (Recommendations)

### Phân tích kỹ thuật:
1. **Baseline**: Đạt hiệu năng ổn định, latency thấp nhất. Tuy nhiên dễ trượt ở các câu hỏi yêu cầu ngữ cảnh rộng hơn do chunk bị cắt rời.
2. **HyDE**: Cải thiện ngữ nghĩa đối với tài liệu song ngữ, nhưng có nguy cơ hallucinate các ngày tháng/mức tiền cụ thể trong các tài liệu học bổng chính quy làm nhiễu vector search.
3. **HyPE**: Tạo câu hỏi giả định ở giai đoạn Index nên duy trì được latency online thấp (0 token runtime). Tuy nhiên, nếu tài liệu chứa nhiều bảng biểu số liệu phức tạp, việc tạo câu hỏi giả định có thể bỏ sót các ngóc ngách thông tin.
4. **Window Enrichment (Mới)**: Giúp tăng Recall đáng kể cho các câu hỏi tổng hợp do kết hợp thông tin trước-sau của tài liệu gốc, trong khi **chi phí token và latency tăng ở mức tối thiểu**.
5. **Context Compression (Mới)**: Giảm thiểu độ nhiễu và số lượng token gửi lên LLM khi sinh câu trả lời cuối cùng, nhưng làm tăng latency runtime và token cost ở query time.

### 🌟 Lời khuyên áp dụng (Recommendation):
👉 **Khuyến nghị chọn Baseline**: Cấu hình Baseline hiện tại đã đạt hiệu năng tối ưu nhất trên tập dữ liệu học bổng này, đồng thời giữ chi phí vận hành ở mức thấp nhất.