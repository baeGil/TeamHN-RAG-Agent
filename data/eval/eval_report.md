# Báo cáo Đánh giá Retrieval (k=5)

- Số câu hỏi: **20**
- Tài liệu: mô_hình_hóa_ĐATN
- Phương pháp định nhãn liên quan: LLM-as-judge trên tập ứng viên gộp (pooling).

## Kết quả tổng thể

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.738 | 0.850 | 0.950 |
| Dense | 0.750 | 0.829 | 0.950 |
| Hybrid (RRF) | 0.793 | 0.967 | 1.000 |

## Theo độ khó: easy

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.743 | 0.750 | 0.900 |
| Dense | 0.857 | 0.808 | 1.000 |
| Hybrid (RRF) | 0.843 | 0.933 | 1.000 |

## Theo độ khó: hard

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.732 | 0.950 | 1.000 |
| Dense | 0.643 | 0.850 | 0.900 |
| Hybrid (RRF) | 0.743 | 1.000 | 1.000 |

## Phân tích lỗi (Hybrid (RRF) trượt @5)

Không có câu hỏi nào trượt ở top-5 với cấu hình tốt nhất. ✅