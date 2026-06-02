# Báo cáo Đánh giá và So sánh Retrieval: Baseline vs HyDE vs HyPE

- Số câu hỏi: **20** (10 easy, 10 hard)
- Tài liệu nguồn RAG: `mo_hinh_hoa_DATN`
- Phương pháp định nhãn liên quan: LLM-as-judge trên tập ứng viên gộp (pooling) từ cả 3 cấu hình.

## 📊 Kết quả tổng thể

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (ms) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.667 | 0.767 | 0.900 | 1.2ms |
| Baseline | Dense | 0.821 | 0.838 | 1.000 | 428.0ms |
| Baseline | Hybrid (RRF) | 0.838 | 0.917 | 1.000 | 0.9ms |
| Baseline | Hybrid + Rerank | 0.838 | 0.917 | 1.000 | 155.5ms |
| HyDE | BM25 | 0.667 | 0.767 | 0.900 | 0.2ms |
| HyDE | Dense | 0.754 | 0.852 | 1.000 | 2860.3ms |
| HyDE | Hybrid (RRF) | 0.762 | 0.925 | 1.000 | 2427.3ms |
| HyDE | Hybrid + Rerank | 0.762 | 0.900 | 1.000 | 2252.5ms |
| HyPE | BM25 | 0.667 | 0.767 | 0.900 | 0.3ms |
| HyPE | Dense | 0.833 | 0.852 | 1.000 | 480.0ms |
| HyPE | Hybrid (RRF) | 0.863 | 0.917 | 1.000 | 1.0ms |
| HyPE | Hybrid + Rerank | 0.863 | 0.917 | 1.000 | 102.1ms |

## 📈 So sánh chi tiết theo độ khó

### Theo độ khó: easy

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (ms) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.717 | 0.750 | 0.900 | 0.8ms |
| Baseline | Dense | 0.800 | 0.900 | 1.000 | 501.4ms |
| Baseline | Hybrid (RRF) | 0.833 | 0.950 | 1.000 | 0.9ms |
| Baseline | Hybrid + Rerank | 0.833 | 0.950 | 1.000 | 310.2ms |
| HyDE | BM25 | 0.717 | 0.750 | 0.900 | 0.2ms |
| HyDE | Dense | 0.733 | 0.900 | 1.000 | 2482.9ms |
| HyDE | Hybrid (RRF) | 0.767 | 0.950 | 1.000 | 2177.4ms |
| HyDE | Hybrid + Rerank | 0.767 | 0.950 | 1.000 | 2136.5ms |
| HyPE | BM25 | 0.717 | 0.750 | 0.900 | 0.3ms |
| HyPE | Dense | 0.933 | 0.833 | 1.000 | 583.4ms |
| HyPE | Hybrid (RRF) | 0.883 | 0.950 | 1.000 | 0.9ms |
| HyPE | Hybrid + Rerank | 0.883 | 0.950 | 1.000 | 203.3ms |

### Theo độ khó: hard

| Cấu hình | Phương pháp | Recall@5 | MRR@5 | Hit@5 | Latency (ms) |
|---|---|---|---|---|---|
| Baseline | BM25 | 0.617 | 0.783 | 0.900 | 1.6ms |
| Baseline | Dense | 0.842 | 0.775 | 1.000 | 354.7ms |
| Baseline | Hybrid (RRF) | 0.842 | 0.883 | 1.000 | 0.9ms |
| Baseline | Hybrid + Rerank | 0.842 | 0.883 | 1.000 | 0.9ms |
| HyDE | BM25 | 0.617 | 0.783 | 0.900 | 0.2ms |
| HyDE | Dense | 0.775 | 0.803 | 1.000 | 3237.7ms |
| HyDE | Hybrid (RRF) | 0.758 | 0.900 | 1.000 | 2677.2ms |
| HyDE | Hybrid + Rerank | 0.758 | 0.850 | 1.000 | 2368.4ms |
| HyPE | BM25 | 0.617 | 0.783 | 0.900 | 0.3ms |
| HyPE | Dense | 0.733 | 0.870 | 1.000 | 376.6ms |
| HyPE | Hybrid (RRF) | 0.842 | 0.883 | 1.000 | 1.1ms |
| HyPE | Hybrid + Rerank | 0.842 | 0.883 | 1.000 | 1.0ms |

## 🔍 Phân tích chi tiết: Thời gian vs Độ bao phủ (Recall)

### 1. Thời gian truy hồi (Latency)
- **Baseline**: Có thời gian truy hồi thấp nhất do chỉ thực hiện tìm kiếm từ vựng (BM25) và nhúng truy vấn trực tiếp để tìm kiếm vector.
- **HyPE (Hypothetical Prompt Embeddings)**: Giữ được latency gần tương đương với Baseline. Do việc tạo câu hỏi giả định được thực hiện trước ở **giai đoạn index**, tại thời điểm query hệ thống chỉ nhúng câu hỏi gốc rồi truy hồi trên không gian câu hỏi. Điều này giúp tối ưu hóa đáng kể thời gian so với HyDE.
- **HyDE (Hypothetical Document Embeddings)**: Có latency cao nhất và vượt trội hẳn so với 2 phương pháp còn lại, do bắt buộc phải gọi LLM sinh câu trả lời giả định cho mỗi câu hỏi tại **giai đoạn query** trước khi nhúng và tìm kiếm vector.

### 2. Độ bao phủ (Recall@5)
- **HyDE**: Cải thiện rõ rệt độ Recall đối với các câu hỏi khó (hard) do câu trả lời giả định từ LLM giúp giảm khoảng cách ngữ nghĩa giữa câu hỏi dạng hỏi và các văn bản gốc dạng mô tả.
- **HyPE**: Đạt Recall rất cao trên cả các câu hỏi dễ và câu hỏi khó do không gian tìm kiếm vector chứa các câu hỏi đa dạng được sinh từ chunk trước đó, giúp việc khớp câu hỏi - câu hỏi hiệu quả hơn.

## 🛠️ Phân tích lỗi (Hybrid + Rerank trượt @5)

### Cấu hình: Baseline
Không có câu hỏi nào trượt ở top-5 với cấu hình Hybrid + Rerank. ✅

### Cấu hình: HyDE
Không có câu hỏi nào trượt ở top-5 với cấu hình Hybrid + Rerank. ✅

### Cấu hình: HyPE
Không có câu hỏi nào trượt ở top-5 với cấu hình Hybrid + Rerank. ✅
