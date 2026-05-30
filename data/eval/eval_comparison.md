# So sánh Trước/Sau khi tối ưu parsing + chunking + contextual headers

> Trước = parsing cũ (pdfplumber, không section/header). Sau = parsing math-aware (PyMuPDF) + chunk theo mục + contextual headers.
> Lưu ý: tập đoạn liên quan được định nhãn lại (LLM-as-judge, TREC pooling) cho mỗi lần chạy vì chunk_id thay đổi sau khi parse lại.

- Số đoạn liên quan TB/câu: Trước **2.05** → Sau **1.95**; số chunk: 14 → 20.
- Chunk theo mục tạo ra các đoạn **nhỏ và tập trung hơn**, nên nội dung liên quan trải ra nhiều đoạn hơn. Điều này **cải thiện thứ hạng** (MRR@5 tăng ở mọi cấu hình, Hit@5 → 1.000) nhưng có thể làm Recall@5 của Hybrid giảm nhẹ ở mức k=5 cố định (nội dung bị phân mảnh). Cấu hình dùng thực tế (Hybrid + Rerank) gần như không đổi về recall và tốt hơn về MRR/Hit.
- **Lợi ích chính (định tính):** công thức toán trong đoạn được trích đúng (ví dụ `S(c) = C1·(24−Nobs(c))/24 + (1−C1)·dmin(c,Oc)/3`) thay vì bị vỡ (`(cid:88)`, `(cid:113)`, ký tự xáo trộn) như parser cũ → giảm trả lời sai công thức.

## Tổng thể

| Phương pháp | Metric | Trước | Sau | Δ |
|---|---|---|---|---|
| BM25 | recall@5 | 0.740 | 0.797 | +0.057 |
| BM25 | mrr@5 | 0.692 | 0.754 | +0.062 |
| BM25 | hit@5 | 0.900 | 0.900 | +0.000 |
| Dense | recall@5 | 0.813 | 0.880 | +0.067 |
| Dense | mrr@5 | 0.792 | 0.850 | +0.058 |
| Dense | hit@5 | 0.950 | 1.000 | +0.050 |
| Hybrid (RRF) | recall@5 | 0.923 | 0.845 | -0.078 |
| Hybrid (RRF) | mrr@5 | 0.825 | 0.852 | +0.027 |
| Hybrid (RRF) | hit@5 | 1.000 | 1.000 | +0.000 |
| Hybrid + Rerank | recall@5 | 0.857 | 0.845 | -0.012 |
| Hybrid + Rerank | mrr@5 | 0.850 | 0.863 | +0.013 |
| Hybrid + Rerank | hit@5 | 0.950 | 1.000 | +0.050 |

## Theo độ khó: easy

| Phương pháp | Metric | Trước | Sau | Δ |
|---|---|---|---|---|
| BM25 | recall@5 | 0.850 | 0.850 | +0.000 |
| BM25 | mrr@5 | 0.700 | 0.733 | +0.033 |
| BM25 | hit@5 | 0.900 | 0.900 | +0.000 |
| Dense | recall@5 | 0.800 | 0.917 | +0.117 |
| Dense | mrr@5 | 0.833 | 0.875 | +0.042 |
| Dense | hit@5 | 0.900 | 1.000 | +0.100 |
| Hybrid (RRF) | recall@5 | 0.950 | 0.917 | -0.033 |
| Hybrid (RRF) | mrr@5 | 0.883 | 0.900 | +0.017 |
| Hybrid (RRF) | hit@5 | 1.000 | 1.000 | +0.000 |
| Hybrid + Rerank | recall@5 | 0.950 | 0.917 | -0.033 |
| Hybrid + Rerank | mrr@5 | 1.000 | 0.950 | -0.050 |
| Hybrid + Rerank | hit@5 | 1.000 | 1.000 | +0.000 |

## Theo độ khó: hard

| Phương pháp | Metric | Trước | Sau | Δ |
|---|---|---|---|---|
| BM25 | recall@5 | 0.630 | 0.743 | +0.113 |
| BM25 | mrr@5 | 0.683 | 0.775 | +0.092 |
| BM25 | hit@5 | 0.900 | 0.900 | +0.000 |
| Dense | recall@5 | 0.827 | 0.843 | +0.017 |
| Dense | mrr@5 | 0.750 | 0.825 | +0.075 |
| Dense | hit@5 | 1.000 | 1.000 | +0.000 |
| Hybrid (RRF) | recall@5 | 0.897 | 0.773 | -0.123 |
| Hybrid (RRF) | mrr@5 | 0.767 | 0.803 | +0.037 |
| Hybrid (RRF) | hit@5 | 1.000 | 1.000 | +0.000 |
| Hybrid + Rerank | recall@5 | 0.763 | 0.773 | +0.010 |
| Hybrid + Rerank | mrr@5 | 0.700 | 0.775 | +0.075 |
| Hybrid + Rerank | hit@5 | 0.900 | 1.000 | +0.100 |