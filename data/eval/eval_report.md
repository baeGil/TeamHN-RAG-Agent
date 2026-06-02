# Báo cáo Đánh giá Retrieval (k=5)

- Số câu hỏi: **20**
- Tài liệu: mô_hình_hóa_ĐATN
- Phương pháp định nhãn liên quan: LLM-as-judge trên tập ứng viên gộp (pooling).

## Kết quả tổng thể

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.797 | 0.754 | 0.900 |
| Dense | 0.880 | 0.850 | 1.000 |
| Hybrid (RRF) | 0.845 | 0.852 | 1.000 |
| Hybrid + Rerank | 0.845 | 0.863 | 1.000 |

## Theo độ khó: easy

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.850 | 0.733 | 0.900 |
| Dense | 0.917 | 0.875 | 1.000 |
| Hybrid (RRF) | 0.917 | 0.900 | 1.000 |
| Hybrid + Rerank | 0.917 | 0.950 | 1.000 |

## Theo độ khó: hard

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.743 | 0.775 | 0.900 |
| Dense | 0.843 | 0.825 | 1.000 |
| Hybrid (RRF) | 0.773 | 0.803 | 1.000 |
| Hybrid + Rerank | 0.773 | 0.775 | 1.000 |

## Phân tích lỗi (Hybrid + Rerank trượt @5)

Không có câu hỏi nào trượt ở top-5 với cấu hình tốt nhất. ✅