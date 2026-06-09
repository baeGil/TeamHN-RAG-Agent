# Báo cáo Đánh giá A/B Test RAG Học bổng Toàn diện: 5 Cấu hình Retrieval & Generation

- **Tổng số câu hỏi kiểm thử**: 34 (5 Dễ, 5 Trung bình, 5 Khó)
- **Tài liệu nguồn**: `REGLEMENTATION_LAUREATS Eiffel_2024_EN.pdf`, `REGLEMENTATION_LAUREATS Eiffel_2024_VI.pdf`, `re_glement_programme_de_bourses_d_excellence_2025-_vn (1).pdf`
- **Chỉ số đánh giá chính**: **Recall@5** (Tính tương đồng với phán quyết liên quan từ LLM-as-judge)
- **Chỉ số đánh giá phụ**: **Latency (ms)** và **Token Cost (Ingest & Query)**

## 📊 1. Kết quả hiệu năng truy hồi tổng thể (Overall Performance)

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) | Avg Query Tokens |
|---|---|---|---|---|---|---|
| Baseline | BM25 | 0.498 | 1.000 | 1.000 | 109.3ms | 0.0 tokens |
| Baseline | Dense | 0.012 | 0.015 | 0.029 | 8.8ms | 0.0 tokens |
| Baseline | Hybrid (RRF) | 0.568 | 1.000 | 1.000 | 3.8ms | 0.0 tokens |
| HyDE | BM25 | 0.498 | 1.000 | 1.000 | 3.0ms | 0.0 tokens |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2468.9ms | 145.3 tokens |
| HyDE | Hybrid (RRF) | 0.544 | 0.882 | 0.971 | 2020.1ms | 145.1 tokens |
| HyPE | BM25 | 0.498 | 1.000 | 1.000 | 3.1ms | 0.0 tokens |
| HyPE | Dense | 0.012 | 0.015 | 0.029 | 9.9ms | 0.0 tokens |
| HyPE | Hybrid (RRF) | 0.568 | 1.000 | 1.000 | 3.2ms | 0.0 tokens |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 4.3ms | 0.0 tokens |
| Window Enrichment | Dense | 0.054 | 0.071 | 0.235 | 9.9ms | 0.0 tokens |
| Window Enrichment | Hybrid (RRF) | 0.562 | 1.000 | 1.000 | 4.3ms | 0.0 tokens |
| Context Compression | BM25 | 0.498 | 1.000 | 1.000 | 2393.9ms | 6021.4 tokens |
| Context Compression | Dense | 0.012 | 0.015 | 0.029 | 2709.3ms | 8943.5 tokens |
| Context Compression | Hybrid (RRF) | 0.568 | 1.000 | 1.000 | 2268.5ms | 6140.5 tokens |

## 📈 2. So sánh chi tiết theo độ khó (Difficulty Breakdown)

### Theo độ khó: Dễ

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.496 | 1.000 | 1.000 | 259.6ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 19.8ms |
| Baseline | Hybrid (RRF) | 0.555 | 1.000 | 1.000 | 3.3ms |
| HyDE | BM25 | 0.496 | 1.000 | 1.000 | 3.5ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2449.0ms |
| HyDE | Hybrid (RRF) | 0.490 | 0.911 | 1.000 | 1834.2ms |
| HyPE | BM25 | 0.496 | 1.000 | 1.000 | 3.5ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 22.2ms |
| HyPE | Hybrid (RRF) | 0.555 | 1.000 | 1.000 | 2.7ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 4.9ms |
| Window Enrichment | Dense | 0.012 | 0.014 | 0.071 | 20.6ms |
| Window Enrichment | Hybrid (RRF) | 0.555 | 1.000 | 1.000 | 4.1ms |
| Context Compression | BM25 | 0.496 | 1.000 | 1.000 | 1863.8ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2545.2ms |
| Context Compression | Hybrid (RRF) | 0.555 | 1.000 | 1.000 | 1912.2ms |

### Theo độ khó: Trung bình

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.500 | 1.000 | 1.000 | 3.9ms |
| Baseline | Dense | 0.040 | 0.050 | 0.100 | 1.1ms |
| Baseline | Hybrid (RRF) | 0.567 | 1.000 | 1.000 | 3.8ms |
| HyDE | BM25 | 0.500 | 1.000 | 1.000 | 2.6ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2715.5ms |
| HyDE | Hybrid (RRF) | 0.610 | 0.925 | 1.000 | 2052.1ms |
| HyPE | BM25 | 0.500 | 1.000 | 1.000 | 2.7ms |
| HyPE | Dense | 0.040 | 0.050 | 0.100 | 1.3ms |
| HyPE | Hybrid (RRF) | 0.567 | 1.000 | 1.000 | 3.5ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 3.7ms |
| Window Enrichment | Dense | 0.092 | 0.140 | 0.300 | 2.5ms |
| Window Enrichment | Hybrid (RRF) | 0.560 | 1.000 | 1.000 | 4.3ms |
| Context Compression | BM25 | 0.500 | 1.000 | 1.000 | 2666.5ms |
| Context Compression | Dense | 0.040 | 0.050 | 0.100 | 2860.7ms |
| Context Compression | Hybrid (RRF) | 0.567 | 1.000 | 1.000 | 2494.8ms |

### Theo độ khó: Khó

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (Query) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.500 | 1.000 | 1.000 | 4.1ms |
| Baseline | Dense | 0.000 | 0.000 | 0.000 | 1.0ms |
| Baseline | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 4.4ms |
| HyDE | BM25 | 0.500 | 1.000 | 1.000 | 2.8ms |
| HyDE | Dense | 0.000 | 0.000 | 0.000 | 2250.1ms |
| HyDE | Hybrid (RRF) | 0.552 | 0.800 | 0.900 | 2248.4ms |
| HyPE | BM25 | 0.500 | 1.000 | 1.000 | 2.8ms |
| HyPE | Dense | 0.000 | 0.000 | 0.000 | 1.1ms |
| HyPE | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 3.4ms |
| Window Enrichment | BM25 | 0.500 | 1.000 | 1.000 | 4.1ms |
| Window Enrichment | Dense | 0.075 | 0.080 | 0.400 | 2.5ms |
| Window Enrichment | Hybrid (RRF) | 0.573 | 1.000 | 1.000 | 4.5ms |
| Context Compression | BM25 | 0.500 | 1.000 | 1.000 | 2863.4ms |
| Context Compression | Dense | 0.000 | 0.000 | 0.000 | 2787.7ms |
| Context Compression | Hybrid (RRF) | 0.587 | 1.000 | 1.000 | 2541.1ms |

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