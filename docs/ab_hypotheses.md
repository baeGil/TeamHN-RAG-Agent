# A/B Test — Giả thuyết tích hợp từng kỹ thuật

## Baseline hiện tại

Pipeline: Reducto agentic parse → BM25 + Dense + RRF + Reranker → LLM agent (router / planner / sufficiency / verify).

**Failure modes còn lại:** query ngắn/mơ hồ thiếu tín hiệu lexical; evidence bị phân mảnh qua nhiều chunk; top-k có chunk gần-duplicate; multi-hop cần ngữ cảnh trải trên nhiều đoạn liên tiếp.

---

## T1 — Query Transformation

### Giả thuyết
Query người dùng thường ngắn, thiếu ngữ cảnh, hoặc dùng ngôn ngữ hỏi trong khi corpus dùng ngôn ngữ trả lời. Một rewrite rõ hơn hoặc một step-back rộng hơn sẽ tăng tín hiệu lexical và semantic, giúp cả BM25 và dense retriever match đúng đoạn hơn — đặc biệt với câu hỏi ngắn hoặc implicit.

**Vùng tác động:** query-time thuần, không đụng Reducto → vùng test sạch nhất.

### Failure mode nhắm vào
- Query mơ hồ hoặc quá ngắn bị miss bởi BM25.
- Câu hỏi implicit cần step-back để lấy bối cảnh nền trước khi drill-down.

### Metrics cần so sánh

| Metric | Lý do |
|--------|-------|
| **Recall@5** (primary) | Đo trực tiếp tỷ lệ chunk liên quan được tìm thấy — kỳ vọng tăng nhờ query rõ hơn. |
| **MRR@5** | Đo thứ hạng chunk đúng đầu tiên — rewrite tốt nên đẩy chunk đúng lên cao hơn. |
| **Tail-query Hit@5** (hard set) | Câu hỏi khó/multi-hop thường hưởng lợi nhiều nhất từ query expansion. |
| **Latency delta (p95)** | Mỗi variant thêm 1 LLM call — cần đo cost tradeoff. |


---

## T4 — Contextual Chunk Headers

### Giả thuyết
Chunk hiện tại đã prepend `title › section` nhờ `_index_text`, nhưng chỉ ở dạng path ngắn. Nâng lên thành header đầy đủ hơn (domain, summary một câu, page range) giúp BM25 và dense vector phản ánh cả **vị trí logic** lẫn nội dung chunk. Chunk có header sẽ ít bị "mù ngữ cảnh" hơn, đặc biệt khi chunk rất ngắn hoặc bị cắt rời khỏi section mẹ.

**Vùng tác động:** thay đổi index-text → cần re-index riêng; không đụng Reducto embed_text (dense vẫn dùng embed_text gốc của Reducto).

### Failure mode nhắm vào
- Chunk ngắn/fragment không tự giải thích được chủ đề mà không có header.
- BM25 miss vì không thấy từ khoá của document-level context.

### Metrics cần so sánh

| Metric | Lý do |
|--------|-------|
| **Context Precision@5** | Đo tỷ lệ chunk trả về thực sự liên quan — header tốt giảm noise chunk cùng từ khoá nhưng khác ý. |
| **MRR@5** | Header giúp chunk đúng được rank cao hơn trong BM25. |
| **Citation accuracy** | Chunk có context đúng → agent cite đúng section/trang hơn. |
| **Recall@5** | Baseline để xác nhận header không làm giảm recall do BM25 bị nhiễu bởi boilerplate. |


---

## T8 — Relevant Segment Extraction (RSE)

### Giả thuyết
Top-k hiện tại trả về các chunk rời rạc. Với tài liệu kỹ thuật/pháp lý, evidence thường cluster theo đoạn liên tiếp — một công thức hay một điều khoản thường cần vài chunk liền nhau để đọc trọn vẹn. RSE gom các chunk liên tiếp có score cao thành segment mạch lạc, giúp LLM đọc context có đầu-giữa-cuối thay vì list mảnh vụn.

**Vùng tác động:** post-rerank, query-time thuần → không đụng Reducto; Reducto làm chunk boundary tốt nên RSE càng dễ ghép đúng.

### Failure mode nhắm vào
- Multi-hop query cần evidence trải trên nhiều chunk liên tiếp.
- Top-k trả về gần-duplicate hoặc thiếu chunk "cầu nối" giữa 2 chunk liên quan.

### Metrics cần so sánh

| Metric | Lý do |
|--------|-------|
| **Recall@5** (primary) | RSE mở rộng context sang chunk lân cận → kỳ vọng tăng recall với multi-hop. |
| **Faithfulness / Groundedness** | LLM đọc segment mạch lạc → bám nguồn tốt hơn, ít hallucinate hơn. |
| **Redundancy ratio** | Đo xem RSE có đưa chunk gần-duplicate vào không — kỳ vọng giảm redundancy so với top-k naive. |
| **nDCG@5** | Đo ordering + coverage tổng thể tốt hơn Recall/MRR đơn thuần cho segment assembly. |


---

## T10 — Hierarchical Indices

### Giả thuyết
Flat index hiện tại quét toàn bộ chunk pool. Với tài liệu tài chính nhiều trang (báo cáo MWG 26 trang) hoặc văn bản pháp lý có cấu trúc rõ, một tầng coarse (summary/section) có thể thu hẹp search space trước, rồi mới drill-down vào chunk chi tiết. Điều này giúp tăng precision và giảm false positive từ các section không liên quan.

**Vùng tác động:** thêm parent index in-memory, không thay đổi chunk DB; Reducto làm parse/chunk tốt nên parent node được xây từ đoạn đã có cấu trúc rõ.

### Failure mode nhắm vào
- Query rơi vào section sai do từ khoá trùng nhau giữa các mục (e.g., "doanh thu" xuất hiện ở nhiều section khác nhau).
- Corpus lớn có nhiều section — flat index trả về mix từ nhiều nơi, recall tốt nhưng precision thấp.

### Metrics cần so sánh

| Metric | Lý do |
|--------|-------|
| **Context Precision@5** (primary) | Kỳ vọng tăng nhờ coarse-to-fine filter giảm chunk từ section sai. |
| **MRR@5** | Parent boost kéo chunk đúng section lên rank cao hơn. |
| **Recall@5** | Cần monitor — nếu parent summary drift thì có thể bỏ sót section đúng. |
| **Tail-query Hit@5** (hard set) | Query cần drill-down section cụ thể hưởng lợi nhiều nhất. |


---

## Tổng hợp so sánh

| Kỹ thuật | Failure mode chính | Primary metric | Kỳ vọng | Rủi ro chính |
|----------|--------------------|----------------|---------|--------------|
| T1 Query Transform | Query ngắn/mơ hồ | Recall@5, Tail Hit@5 | +recall trên hard/implicit | Latency tăng theo số variant |
| T4 Contextual Headers | Chunk thiếu ngữ cảnh | Context Precision@5, Citation accuracy | +precision, +citation | Re-index cost; BM25 nhiễu nếu header dài |
| T8 RSE | Evidence phân mảnh, multi-hop | Recall@5, Faithfulness | +recall multi-hop, -redundancy | Top-k nhỏ không đủ chunk để ghép segment |
| T10 Hierarchical | Section sai, corpus rộng | Context Precision@5, MRR@5 | +precision | Parent summary drift; ingest cost parent embed |

**Quy tắc đo lường:** giữ cố định embedding model, reranker, generator và eval set. Bật từng kỹ thuật một, không bật đồng thời. Nhìn tail queries (hard set) thay vì chỉ nhìn average.
