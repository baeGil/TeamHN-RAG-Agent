# So sánh Trước/Sau khi tối ưu parsing + chunking + contextual headers

> Trước = parsing cũ (pdfplumber, không section/header). Sau = parsing math-aware (PyMuPDF) + chunk theo mục + contextual headers.
> Lưu ý: tập đoạn liên quan được định nhãn lại (LLM-as-judge, TREC pooling) cho mỗi lần chạy vì chunk_id thay đổi sau khi parse lại.

- Số đoạn liên quan TB/câu: Trước **2.05** → Sau **2.75**.
- Chunk theo mục tạo ra các đoạn **nhỏ và tập trung hơn**, nên nội dung liên quan trải ra nhiều đoạn hơn. Điều này **cải thiện thứ hạng** (MRR@5 tăng ở mọi cấu hình, Hit@5 → 1.000) nhưng có thể làm Recall@5 của Hybrid giảm nhẹ ở mức k=5 cố định (nội dung bị phân mảnh). Cấu hình dùng thực tế (Hybrid + Rerank) gần như không đổi về recall và tốt hơn về MRR/Hit.
- **Lợi ích chính (định tính):** công thức toán trong đoạn được trích đúng (ví dụ `S(c) = C1·(24−Nobs(c))/24 + (1−C1)·dmin(c,Oc)/3`) thay vì bị vỡ (`(cid:88)`, `(cid:113)`, ký tự xáo trộn) như parser cũ → giảm trả lời sai công thức.

## Tổng thể

| Phương pháp | Metric | Trước | Sau | Δ |
|---|---|---|---|---|
| BM25 | recall@5 | 0.740 | 0.738 | -0.002 |
| BM25 | mrr@5 | 0.692 | 0.850 | +0.158 |
| BM25 | hit@5 | 0.900 | 0.950 | +0.050 |
| Dense | recall@5 | 0.813 | 0.750 | -0.063 |
| Dense | mrr@5 | 0.792 | 0.829 | +0.037 |
| Dense | hit@5 | 0.950 | 0.950 | +0.000 |
| Hybrid (RRF) | recall@5 | 0.923 | 0.793 | -0.130 |
| Hybrid (RRF) | mrr@5 | 0.825 | 0.967 | +0.142 |
| Hybrid (RRF) | hit@5 | 1.000 | 1.000 | +0.000 |

## Theo độ khó: easy

| Phương pháp | Metric | Trước | Sau | Δ |
|---|---|---|---|---|
| BM25 | recall@5 | 0.850 | 0.743 | -0.107 |
| BM25 | mrr@5 | 0.700 | 0.750 | +0.050 |
| BM25 | hit@5 | 0.900 | 0.900 | +0.000 |
| Dense | recall@5 | 0.800 | 0.857 | +0.057 |
| Dense | mrr@5 | 0.833 | 0.808 | -0.025 |
| Dense | hit@5 | 0.900 | 1.000 | +0.100 |
| Hybrid (RRF) | recall@5 | 0.950 | 0.843 | -0.107 |
| Hybrid (RRF) | mrr@5 | 0.883 | 0.933 | +0.050 |
| Hybrid (RRF) | hit@5 | 1.000 | 1.000 | +0.000 |

## Theo độ khó: hard

| Phương pháp | Metric | Trước | Sau | Δ |
|---|---|---|---|---|
| BM25 | recall@5 | 0.630 | 0.732 | +0.102 |
| BM25 | mrr@5 | 0.683 | 0.950 | +0.267 |
| BM25 | hit@5 | 0.900 | 1.000 | +0.100 |
| Dense | recall@5 | 0.827 | 0.643 | -0.183 |
| Dense | mrr@5 | 0.750 | 0.850 | +0.100 |
| Dense | hit@5 | 1.000 | 0.900 | -0.100 |
| Hybrid (RRF) | recall@5 | 0.897 | 0.743 | -0.153 |
| Hybrid (RRF) | mrr@5 | 0.767 | 1.000 | +0.233 |
| Hybrid (RRF) | hit@5 | 1.000 | 1.000 | +0.000 |