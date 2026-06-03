# RAG Technique Summary

Tài liệu này gom toàn bộ ý tưởng cốt lõi trong `improvement/` thành một bản đọc nhanh để thuyết trình.

## 5 Ý Cần Nhớ Trước

1. `Query enhancement` làm query tốt hơn trước khi retrieve.
2. `Context enrichment` làm chunk và context "có nghĩa" hơn.
3. `Advanced retrieval` làm candidate selection thông minh hơn.
4. `Adaptive retrieval` chọn chiến lược đúng cho từng loại query.
5. Với PDF path hiện tại, `Reducto` đã bao phủ một phần ingest/chunking, nên không phải technique nào cũng là experiment sạch.

## Bản Đồ Nhanh

- **Chapter 1: Query enhancement**
  - Query transformation
  - HyDE
  - HyPE
- **Chapter 2: Context enrichment**
  - Contextual chunk headers
  - Semantic chunking
  - Context window enhancement
  - Contextual compression
  - Relevant segment extraction
  - Document augmentation
- **Chapter 3: Advanced retrieval**
  - Hierarchical indices
  - Dartboard retrieval
- **Chapter 4: Iterative techniques**
  - Adaptive retrieval

## Chapter 1: Query Enhancement

### 1) Query Transformation

**Core idea**
- Viết lại query cho rõ hơn, rộng hơn, hoặc tách query phức tạp thành nhiều sub-queries.
- Gồm 3 biến thể hay gặp:
  - `Rewrite`: làm query cụ thể hơn.
  - `Step-back`: lùi lên câu hỏi khái quát hơn để lấy bối cảnh.
  - `Decomposition`: tách query nhiều ý thành các query con.

**Khi nên dùng**
- Query quá ngắn.
- Query mơ hồ hoặc thiếu entity.
- Query multi-hop, nhiều ràng buộc, nhiều khía cạnh.

**Vì sao hiệu quả**
- Query người dùng thường khác cách tài liệu viết.
- Viết lại query giúp tăng overlap giữa query và corpus.

**Reducto overlap**
- Không có.
- Đây là vùng test sạch nhất.

**Câu nói dễ thuyết trình**
- "Thay vì hỏi một câu mơ hồ, ta biến nó thành một hoặc nhiều câu dễ tìm hơn."

### 2) HyDE

**Core idea**
- LLM tạo một tài liệu hoặc câu trả lời giả định cho query.
- Sau đó embed bản giả định này để đi tìm tài liệu thật.

**Khi nên dùng**
- Query ngắn nhưng cần tìm tài liệu dài, giàu ngữ cảnh.
- Dense retrieval theo query literal chưa đủ tốt.

**Vì sao hiệu quả**
- Giảm semantic gap giữa ngôn ngữ hỏi và ngôn ngữ trả lời.

**Trade-off**
- Tốn runtime cost vì phải gọi LLM ở query-time.
- Có nguy cơ hallucination trong bản giả định.

**Reducto overlap**
- Không có.

**Câu nói dễ thuyết trình**
- "HyDE không tìm tài liệu bằng câu hỏi thật, mà bằng một câu trả lời giả định."

### 3) HyPE

**Core idea**
- Ở offline indexing, LLM sinh nhiều câu hỏi giả định cho mỗi chunk.
- Query của user sẽ được match với các question embeddings đó.
- Đây là question-to-question retrieval.

**Khi nên dùng**
- Corpus ổn định.
- Muốn dồn chi phí sang indexing để query-time nhanh hơn.

**Vì sao hiệu quả**
- Query của user và embedding trong index đều có cùng style là câu hỏi.

**Trade-off**
- Indexing nặng hơn.
- Cần prompt quality tốt để phủ được đủ kiểu câu hỏi.

**Reducto overlap**
- Không trực tiếp.
- Nhưng Reducto đã làm ingest nặng hơn, nên phải cân nhắc tổng chi phí.

**Câu nói dễ thuyết trình**
- "HyPE chuẩn bị sẵn câu hỏi cho tài liệu, để lúc user hỏi thì hệ thống chỉ cần tìm câu hỏi gần nhất."

## Chapter 2: Context Enrichment

### 4) Contextual Chunk Headers

**Core idea**
- Thêm document-level context và section-level context vào trước chunk trước khi embed.
- Mục tiêu là chunk không còn "trơ".

**Khi nên dùng**
- PDF nhiều section.
- Chunk đúng nội dung nhưng thiếu ngữ cảnh.

**Vì sao hiệu quả**
- Chunk mang theo "ai tôi là" thay vì chỉ có text thô.
- BM25 và dense embedding đều hiểu chunk tốt hơn.

**Reducto overlap**
- Có overlap một phần.
- Hệ hiện tại đã prepend `title › section`.

**Câu nói dễ thuyết trình**
- "Chunk không chỉ cần đúng nội dung, mà còn phải mang theo đúng bối cảnh."

### 5) Semantic Chunking

**Core idea**
- Cắt tài liệu theo ranh giới ngữ nghĩa thay vì cắt theo độ dài cố định.

**Khi nên dùng**
- Long docs.
- Nội dung có cấu trúc ý rõ ràng.

**Vì sao hiệu quả**
- Không cắt ngang một lập luận hoặc một công thức giữa chừng.

**Trade-off**
- Cần nhiều công sức hơn chunking đơn giản.
- Chất lượng phụ thuộc cách xác định breakpoint.

**Reducto overlap**
- Có overlap một phần mạnh.
- Reducto đã chunk theo `page_sections`, nên nếu test trên PDF path hiện tại thì tín hiệu sẽ bị trộn.

**Câu nói dễ thuyết trình**
- "Semantic chunking đảm bảo mỗi chunk chứa trọn một ý."

### 6) Context Window Enhancement

**Core idea**
- Sau khi retrieve chunk chính, kéo thêm chunk hàng xóm trước/sau nó.
- Mục tiêu là phục hồi ngữ cảnh bị cắt.

**Khi nên dùng**
- Câu trả lời trải trên nhiều chunk liên tiếp.
- Cần gỡ đại từ, ký hiệu, hoặc câu nối.

**Vì sao hiệu quả**
- Một chunk đúng nhưng thiếu 1-2 câu lân cận thì vẫn chưa đủ để trả lời tốt.

**Trade-off**
- Tăng token usage.
- Cần dedupe overlap.

**Reducto overlap**
- Không trực tiếp.
- Nhưng chunk boundary tốt hơn thì window enhancement càng hiệu quả.

**Câu nói dễ thuyết trình**
- "Retrieve đúng chunk chưa đủ, cần kéo thêm hàng xóm để đủ ngữ cảnh."

### 7) Contextual Compression

**Core idea**
- Retrieve rộng trước, rồi dùng compressor để giữ lại phần thật sự liên quan.

**Khi nên dùng**
- Context dài.
- Noise nhiều.
- Token budget hạn chế.

**Vì sao hiệu quả**
- Giữ evidence chính, bỏ bớt phần thừa trước khi vào LLM.

**Trade-off**
- Compressor quá aggressive có thể làm mất chi tiết quan trọng.

**Reducto overlap**
- Không trực tiếp.

**Câu nói dễ thuyết trình**
- "Compression không thay retrieval, nó làm sạch context trước khi vào model."

### 8) Relevant Segment Extraction

**Core idea**
- Không trả về các chunk rời rạc, mà ghép các chunk liên tiếp thành một segment có nghĩa.

**Khi nên dùng**
- Multi-hop.
- Long answer.
- Câu trả lời nằm rải qua nhiều đoạn gần nhau.

**Vì sao hiệu quả**
- LLM thích một đoạn mạch lạc hơn là một list mảnh vụn.

**Trade-off**
- Phụ thuộc vào order và quality của chunk boundary.

**Reducto overlap**
- Không trực tiếp.

**Câu nói dễ thuyết trình**
- "RSE biến top-k chunks thành một đoạn văn có đầu, giữa, cuối."

### 9) Document Augmentation

**Core idea**
- Sinh thêm câu hỏi hoặc text kiểu query cho từng document/chunk.
- Mục tiêu là làm document dễ tìm hơn trong embedding space.

**Khi nên dùng**
- Query phrasing khác mạnh với text gốc.
- Corpus có nhiều phrasing sparse.

**Vì sao hiệu quả**
- Tăng "bề mặt truy hồi" của tài liệu.

**Trade-off**
- Có thể thêm noise nếu augment quá tay.

**Reducto overlap**
- Có overlap một phần vì Reducto đã có `embed_text`.

**Câu nói dễ thuyết trình**
- "Ta không chỉ lưu tài liệu, mà còn tạo thêm những câu hỏi mà tài liệu đó có thể trả lời."

## Chapter 3: Advanced Retrieval

### 10) Hierarchical Indices

**Core idea**
- Tổ chức retrieval theo nhiều tầng: summary -> section -> chunk.
- Query đi từ coarse đến fine.

**Khi nên dùng**
- Corpus lớn.
- Cần thu hẹp search space trước khi drill-down.

**Vì sao hiệu quả**
- Không cần quét toàn bộ chunk chi tiết ngay từ đầu.

**Trade-off**
- Index phức tạp hơn.
- Summary layer có thể drift.

**Reducto overlap**
- Không trực tiếp.

**Câu nói dễ thuyết trình**
- "Hierarchical indices là cách tìm đúng vùng tài liệu trước, rồi mới tìm đúng đoạn."

### 11) Dartboard Retrieval

**Core idea**
- Chọn top-k sao cho vừa relevant vừa diverse.
- Tránh kết quả bị lặp ý.

**Khi nên dùng**
- Top-k quá trùng lặp.
- Cần nhiều góc nhìn.

**Vì sao hiệu quả**
- Không phải chunk có similarity cao nhất luôn là chunk hữu ích nhất.

**Trade-off**
- Hy sinh một phần best-match tuyệt đối để đổi lấy diversity.

**Reducto overlap**
- Không trực tiếp.

**Câu nói dễ thuyết trình**
- "Dartboard giữ cho top-k vừa đúng vừa không bị một ý lặp lại quá nhiều."

## Chapter 4: Iterative Techniques

### 12) Adaptive Retrieval

**Core idea**
- Phân loại query trước, rồi chọn strategy phù hợp.
- Biến RAG thành một policy engine.

**Query types thường dùng**
- `Factual`
- `Analytical`
- `Opinion / compare`
- `Contextual`

**Strategy gợi ý**
- Factual: rewrite nhẹ + hybrid retrieval.
- Analytical: decomposition + RSE + compression.
- Opinion/compare: dartboard + diversity.
- Contextual: window enhancement + session history.

**Khi nên dùng**
- Khi baseline retrieval đã ổn.
- Khi muốn tối ưu theo loại query thay vì một công thức cho tất cả.

**Trade-off**
- Cần classifier và logging tốt.
- Route sai sẽ làm degrade chất lượng.

**Reducto overlap**
- Không có.

**Câu nói dễ thuyết trình**
- "Adaptive retrieval là bước biến RAG từ một pipeline cố định thành một policy engine."

## Reducto Và Kỹ Thuật Nào Test Sạch

### Test sạch nhất
- Query transformation
- HyDE
- HyPE
- Adaptive retrieval
- Hierarchical indices
- Dartboard retrieval
- Context window enhancement
- Contextual compression
- Relevant segment extraction

### Có overlap, cần ablate cẩn thận
- Contextual chunk headers
- Semantic chunking
- Document augmentation

### Lưu ý khi nói với team
- Nếu technique chỉ cải thiện vì chunk đẹp hơn, hãy kiểm tra Reducto đã làm hộ phần đó chưa.
- Nếu muốn đo công bằng, giữ cố định:
  - parse layer
  - embedding model
  - reranker
  - eval set

## Trình Tự Thuyết Trình Gợi Ý

1. Nói baseline hiện tại của team đang có gì.
2. Giải thích Reducto đã làm gì và overlap ở đâu.
3. Đi qua query enhancement trước vì đây là test sạch nhất.
4. Sang context enrichment để giải thích cách làm chunk "có ngữ cảnh".
5. Kết thúc bằng advanced retrieval và adaptive retrieval như lớp policy cuối.
6. Chốt bằng cách đo, A/B test, và rollout.

## One-liner Cho Mỗi Nhóm

- **Query enhancement:** làm query tốt hơn.
- **Context enrichment:** làm chunk và context giàu nghĩa hơn.
- **Advanced retrieval:** chọn candidate thông minh hơn.
- **Adaptive retrieval:** chọn chiến lược đúng cho từng query.

## Local References

- `improvement/query-enhancement/query-transformation.md`
- `improvement/query-enhancement/hyde.md`
- `improvement/query-enhancement/hype.md`
- `improvement/context-enrichment/contextual-chunk-headers.md`
- `improvement/context-enrichment/semantic-chunking.md`
- `improvement/context-enrichment/context-window-enhancement.md`
- `improvement/context-enrichment/contextual-compression.md`
- `improvement/context-enrichment/relevant-segment-extraction.md`
- `improvement/context-enrichment/document-augmentation.md`
- `improvement/advanced-retrieval/hierarchical-indices.md`
- `improvement/advanced-retrieval/dartboard-retrieval.md`
- `improvement/iterative-techniques/adaptive-retrieval.md`
