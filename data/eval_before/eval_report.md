# Báo cáo Đánh giá Retrieval (k=5)

- Số câu hỏi: **20**
- Tài liệu: mô_hình_hóa_ĐATN
- Phương pháp định nhãn liên quan: LLM-as-judge trên tập ứng viên gộp (pooling).

## Kết quả tổng thể

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.740 | 0.692 | 0.900 |
| Dense | 0.813 | 0.792 | 0.950 |
| Hybrid (RRF) | 0.923 | 0.825 | 1.000 |
| Hybrid + Rerank | 0.857 | 0.850 | 0.950 |

## Theo độ khó: easy

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.850 | 0.700 | 0.900 |
| Dense | 0.800 | 0.833 | 0.900 |
| Hybrid (RRF) | 0.950 | 0.883 | 1.000 |
| Hybrid + Rerank | 0.950 | 1.000 | 1.000 |

## Theo độ khó: hard

| Phương pháp | Recall@5 | MRR@5 | Hit@5 |
|---|---|---|---|
| BM25 | 0.630 | 0.683 | 0.900 |
| Dense | 0.827 | 0.750 | 1.000 |
| Hybrid (RRF) | 0.897 | 0.767 | 1.000 |
| Hybrid + Rerank | 0.763 | 0.700 | 0.900 |

## Phân tích lỗi (Hybrid + Rerank trượt @5)

- **h2** (hard): Trong công thức `S(c) = C1 * (24 - Nobs(c)) / 24 + (1 - C1) * dmin(c, Oc) / 3`, điều gì xảy ra với `S(c)` khi `C1 -> 1`? Khi đó thuật toán ưu tiên yếu tố nào hơn?
  - top5 = [3, 6, 8, 12, 13], số đoạn liên quan = 1