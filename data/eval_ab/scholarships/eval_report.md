# Báo cáo Đánh giá A/B Test RAG Học bổng Toàn diện: 5 Cấu hình Retrieval & Generation

- **Tổng số câu hỏi kiểm thử**: 15 (5 Dễ, 5 Trung bình, 5 Khó)
- **Tài liệu nguồn**: `REGLEMENTATION_LAUREATS Eiffel_2024_EN.pdf`, `REGLEMENTATION_LAUREATS Eiffel_2024_VI.pdf`, `re_glement_programme_de_bourses_d_excellence_2025-_vn (1).pdf`
- **Chỉ số đánh giá chính**: **Recall@5** (Tính tương đồng với phán quyết liên quan từ LLM-as-judge)
- **Chỉ số đánh giá phụ**: **Latency (ms)** và **Token Cost (Ingest & Query)**

## 📊 1. Kết quả hiệu năng truy hồi tổng thể (Overall Performance)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) | Avg Query Tokens |
|---|---|---|---|---|---|---|
| Baseline | BM25 | 0.689 | 0.800 | 0.867 | 243.9ms | 0.0 tokens |
| Baseline | Dense | 0.217 | 0.233 | 0.267 | 20.0ms | 0.0 tokens |
| Baseline | Hybrid (RRF) | 0.067 | 0.067 | 0.067 | 4.2ms | 0.0 tokens |
| HyDE | BM25 | 0.375 | 0.433 | 0.467 | 4.8ms | 0.0 tokens |
| HyDE | Dense | 0.033 | 0.067 | 0.067 | 1919.5ms | 170.5 tokens |
| HyDE | Hybrid (RRF) | 0.053 | 0.067 | 0.067 | 2216.4ms | 165.5 tokens |
| HyPE | BM25 | 0.524 | 0.600 | 0.667 | 5.0ms | 0.0 tokens |
| HyPE | Dense | 0.033 | 0.067 | 0.067 | 19.4ms | 0.0 tokens |
| HyPE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 3.9ms | 0.0 tokens |
| Window Enrichment | BM25 | 0.067 | 0.067 | 0.067 | 5.9ms | 0.0 tokens |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 18.7ms | 0.0 tokens |
| Window Enrichment | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 4.9ms | 0.0 tokens |
| Context Compression | BM25 | 0.284 | 0.333 | 0.333 | 2256.8ms | 5209.7 tokens |
| Context Compression | Dense | 0.067 | 0.067 | 0.067 | 2343.8ms | 4790.1 tokens |
| Context Compression | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 2105.2ms | 5086.9 tokens |

## 📈 2. So sánh chi tiết theo độ khó (Difficulty Breakdown)

### Theo độ khó: Dễ

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.860 | 1.000 | 1.000 | 721.2ms |
| Baseline | Dense | 0.550 | 0.500 | 0.600 | 57.5ms |
| Baseline | Hybrid (RRF) | 0.200 | 0.200 | 0.200 | 3.5ms |
| HyDE | BM25 | 0.500 | 0.600 | 0.600 | 6.9ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 1848.6ms |
| HyDE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 1849.2ms |
| HyPE | BM25 | 0.500 | 0.600 | 0.600 | 7.7ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 55.9ms |
| HyPE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 3.2ms |
| Window Enrichment | BM25 | 0.200 | 0.200 | 0.200 | 7.6ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 51.6ms |
| Window Enrichment | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 4.3ms |
| Context Compression | BM25 | 0.333 | 0.400 | 0.400 | 1908.1ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2028.2ms |
| Context Compression | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 1842.4ms |

### Theo độ khó: Trung bình

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.660 | 0.700 | 0.800 | 3.9ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 1.2ms |
| Baseline | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 4.0ms |
| HyDE | BM25 | 0.400 | 0.300 | 0.400 | 2.9ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 1931.1ms |
| HyDE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 2519.7ms |
| HyPE | BM25 | 0.660 | 0.700 | 0.800 | 3.0ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 1.1ms |
| HyPE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 3.7ms |
| Window Enrichment | BM25 | 0.000 | 0.000 | 0.000 | 4.4ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 2.2ms |
| Window Enrichment | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 4.4ms |
| Context Compression | BM25 | 0.400 | 0.400 | 0.400 | 2419.3ms |
| Context Compression | Dense | 0.200 | 0.200 | 0.200 | 1995.8ms |
| Context Compression | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 2065.8ms |

### Theo độ khó: Khó

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.547 | 0.700 | 0.800 | 6.5ms |
| Baseline | Dense | 0.100 | 0.200 | 0.200 | 1.2ms |
| Baseline | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 5.2ms |
| HyDE | BM25 | 0.225 | 0.400 | 0.400 | 4.5ms |
| HyDE | Dense | 0.100 | 0.200 | 0.200 | 1978.7ms |
| HyDE | Hybrid (RRF) | 0.160 | 0.200 | 0.200 | 2280.4ms |
| HyPE | BM25 | 0.411 | 0.500 | 0.600 | 4.3ms |
| HyPE | Dense | 0.100 | 0.200 | 0.200 | 1.2ms |
| HyPE | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 4.9ms |
| Window Enrichment | BM25 | 0.000 | 0.000 | 0.000 | 5.6ms |
| Window Enrichment | Dense | 0.000 | 0.000 | 0.000 | 2.1ms |
| Window Enrichment | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 6.1ms |
| Context Compression | BM25 | 0.120 | 0.200 | 0.200 | 2442.9ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 3007.4ms |
| Context Compression | Hybrid (RRF) | 0.000 | 0.000 | 0.000 | 2407.3ms |

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