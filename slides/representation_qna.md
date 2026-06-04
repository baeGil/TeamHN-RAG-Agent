# Chuẩn bị hỏi đáp — Hệ thống RAG Agent Tiếng Việt

---

## Slide 1 — Tổng quan

- **Tại sao chọn gpt-4o-mini thay vì gpt-4o?**: gpt-4o-mini nhanh hơn 3–5×, rẻ hơn ~10×. Trong RAG, LLM đã có context từ retrieved chunks nên không cần "trí tuệ" model lớn — chủ yếu cần tốc độ parse JSON và viết câu trả lời coherent từ context đã cho. Config cho phép override `LLM_MODEL` nếu cần model mạnh hơn.

- **Tại sao dùng text-embedding-3-small mà không phải text-embedding-3-large?**: `-small` tạo vector 1536 chiều, đủ cho semantic search. `-large` tạo 3072 chiều — tốn gấp đôi chi phí API, RAM, và storage Turbovec (768 bytes → 1536 bytes/vector), nhưng cải thiện recall thực tế nhỏ ở task retrieval. Với corpus học thuật tiếng Việt cỡ vừa, 1536 chiều là điểm ngọt giữa chất lượng và chi phí.

- **Node nào là bottleneck về thời gian trong toàn pipeline?**: Reranker (~50–100ms) là bottleneck trong retrieval pipeline. LLM calls (~500–2000ms tổng) là bottleneck tổng thể, đặc biệt khi có replanning (mỗi vòng thêm nhiều LLM calls). BM25 (~1ms), Dense search (~0.5ms), RRF (~0.1ms) đều không đáng kể.

---

## Slide 2 — Kiến trúc Tổng quan

- **Tại sao không dùng LangChain/LangGraph thay vì tự xây?**: LangGraph thêm lớp abstraction ẩn logic flow, khó debug khi agent làm sai. Với graph tự viết: mỗi node là plain Python function, mỗi transition là if-else rõ ràng, có thể đặt log ở bất cứ đâu. Không phụ thuộc vào API của framework bên thứ 3 có thể thay đổi.

- **SQLite WAL mode là gì, tại sao không dùng PostgreSQL?**: WAL (Write-Ahead Logging) cho phép multiple readers đọc đồng thời với 1 writer mà không block nhau. Phù hợp vì chat thread đọc index trong khi upload thread đang ghi. SQLite phù hợp cho deployment 1 máy: không cần server riêng, dữ liệu là 1 file duy nhất dễ backup. PostgreSQL cần thiết khi scale nhiều máy hoặc nhiều writer đồng thời. (Nhưng nếu có nhiều hơn 1 luồng ghi cùng lúc thì sqlite vẫn sẽ block)

- **Nếu OpenAI API down hoàn toàn, hệ thống xử lý thế nào?**: [Chưa có] Áp dụng mẫu thiết kế Circuit Breaker (Cầu dao tự ngắt): Nếu hệ thống gọi OpenAI lỗi liên tục 5 lần, cầu dao sẽ tự ngắt kết nối tới OpenAI trong 30 giây để tránh làm nghẽn hệ thống và Tích hợp Fallback Model chạy Local: Hệ thống sẽ tự động chuyển hướng (Reroute) yêu cầu sang một mô hình mã nguồn mở chạy local (như Llama 3 hoặc Qwen qua Ollama) để duy trì tính liên tục của dịch vụ, dù câu trả lời có thể ngắn hơn."

- **"Air-gapped capable" — khi nào không hoạt động offline?**: Sau khi index đã được build (embeddings cached, Turbovec + BM25 index lưu trên disk), retrieval pipeline chạy hoàn toàn offline. Không hoạt động offline khi: (1) cần embed text mới (gọi OpenAI Embedding API), (2) cần LLM generate câu trả lời (gọi OpenAI Chat API), (3) upload document mới chưa cache embedding.

---

## Slide 3 — Luồng Người dùng

- **SSE events lifecycle**: 1. `thinking`: Bật trạng thái chờ trên màn hình (Loading UX). 2. `route`: Phân loại câu hỏi để định hướng xử lý. 3. `plan`: Chia nhỏ câu hỏi phức tạp thành các câu hỏi phụ. 4. `retrieved`: Lấy và hiển thị các đoạn trích tài liệu tìm được. 5. `token`: Đổ chữ câu trả lời theo thời gian thực (Streaming). 6. `final`: Hoàn thành câu trả lời và đóng kết nối.

- **Tại sao max upload size = 5MB**: Ước lượng 100–150 trang tài liệu chữ tối đa → tiết kiệm tài nguyên. Nếu vượt → backend trả về 413 error.

- **SSE là gì?**: Hệ thống tạo ra một "đường ống" một chiều. Server liên tục đẩy dữ liệu mới xuống trình duyệt ngay khi có sự kiện phát sinh, mà trình duyệt không cần phải gửi đi gửi lại yêu cầu. Client chỉ yêu cầu 1 lần duy nhất + server giữ kết nối luôn mở + server chủ động stream dữ liệu.

- **Tại sao SSE mà không phải WebSocket?**: Hướng chuyển dữ liệu của SSE là 1 chiều (server → client) trong khi WebSocket là 2 chiều. Chọn SSE vì nó nhẹ, chạy trên nền HTTP, không lo bị tường lửa chặn và dễ code. Chọn WebSocket khi làm game online hoặc ứng dụng chat real-time giữa người với người, nơi cả hai bên liên tục bắn dữ liệu qua lại đồng thời.

- **Processing dùng poll mà không phải SSE?**: Giai đoạn Processing là tác vụ bất đồng bộ xảy ra một lần cho mỗi tài liệu và chỉ cần trả về một kết quả cuối cùng (ready hoặc failed). FE thỉnh thoảng gửi request để check trạng thái là đủ, không cần duy trì kết nối liên tục.

- **Tại sao fetch 6 messages từ DB nhưng chỉ đưa 4 vào LLM?**: Từ `config.py`: `HISTORY_WINDOW=6` (fetch từ DB), nhưng `_history_msgs()` dùng `window=4` khi truyền vào agent. Lấy 6 để có buffer khi có summary message, sau đó chỉ đưa 4 message gần nhất vào prompt để giữ context ngắn gọn.

- **Sau khi reload trang, frontend biết câu trả lời đang trong tiến trình thế nào?**: Frontend gọi `getSession()` → lấy messages từ DB → nếu message có `status=processing` → poll `/api` mỗi 2 giây cho đến khi `status=complete`. Mỗi agent event đã được lưu vào `trace_snapshot` trong DB nên sau reload vẫn thấy được tiến trình đã xảy ra.

---

## Slide 4 — Ingestion: PDF Parsing

- **Biết PDF lỗi cấu trúc như vậy thì hệ thống giải quyết thế nào?**: Sử dụng các bộ công cụ layout-aware parsing, mặc dù PyMuPDF không phải công cụ như vậy nhưng hệ sinh thái của nó thì có: pymupdf4llm,... Bên cạnh đó hệ thống tự xây dựng font remap, fraction reconstruction, và geometric space rebuild.

- **Tại sao không dùng pymupdf4llm ngay từ đầu thay vì tự viết extraction?**: [Chưa có so sánh trực tiếp] pymupdf4llm là wrapper high-level, nhưng hệ thống cần kiểm soát cấp thấp: custom font remap cho LaTeX math, fraction bar detection từ vector drawings, và geometric space rebuild. Các wrapper high-level không expose những API cấp thấp của fitz mà hệ thống cần.

- **Nếu PDF có layout 2 cột (double-column), PyMuPDF xử lý thế nào?**: [Chưa có] PyMuPDF đọc theo thứ tự tọa độ — với layout 2 cột, text có thể bị trộn lẫn giữa cột trái và cột phải. Chưa có cơ chế detect và xử lý multi-column layout. Đây là hạn chế hiện tại cho tài liệu paper khoa học dạng 2 cột.

- **Khi nào VLM fallback được trigger trong chế độ `auto`?**: Từ `pdf_extract.py`: sau khi extract text từ 1 trang và áp dụng boilerplate filter, nếu tổng ký tự < 3 thì trang đó được đánh dấu "empty" → đưa vào danh sách `empty_pages` → gọi VLM để render trang đó thành PNG và transcribe.

---

## Slide 5 — PyMuPDF: Font Remap

- **Hệ thống có map được các font khác ngoài CMEX10 không?**: Từ `pdf_extract.py`: `_MATH_OP_FONT_HINTS = ("CMEX", "CMMI", "CMSY", "MSAM", "MSBM")` — hỗ trợ 5 họ font LaTeX math phổ biến. Tuy nhiên bảng remap `_CMEX_REMAP` hiện chỉ có 6 ký tự: `{"X": "∑", "P": "∑", "q": "√", "Y": "∏", "Z": "∫", "R": "∫"}` — áp dụng chung cho tất cả font math được detect.

- **Tại sao biết được là font nào?**: Hệ thống phát hiện tên font nhờ file PDF luôn lưu sẵn thông tin này để trình duyệt hiển thị đúng. PyMuPDF `page.get_text("rawdict")` trả về từng span với `"font"` key chứa tên font đầy đủ.

- **Nếu font không phải LaTeX math nhưng tên chứa "CMEX" thì sao?**: `_is_math_font()` check chuỗi con trong tên font. Trường hợp false positive (font thường tên chứa "CMEX") sẽ bị remap nhầm — nhưng trong thực tế cực kỳ hiếm vì CMEX là prefix chuẩn của LaTeX Computer Modern Expanded fonts.

- **Bảng remap hiện chỉ có 6 ký tự, có đủ không?**: Với các toán tử lớn (large operators) phổ biến nhất trong tài liệu học thuật Việt Nam: ∑ (tổng), √ (căn), ∏ (tích), ∫ (tích phân) — 6 ký tự cover phần lớn trường hợp. Các ký hiệu khác (α, β, γ,...) thường được LaTeX lưu dưới Unicode đúng trong PDF hoặc nằm ở CMMI/CMSY — hiện chưa có remap riêng cho các font đó.

---

## Slide 6 — PyMuPDF: Fraction Reconstruction

- **Mất hoàn toàn cấu trúc phân số**: Các công cụ đọc PDF thông thường chỉ quét tìm các ký tự text. Vì đường gạch ngang không phải là text nên nó bị bỏ qua hoàn toàn.

- **Tại sao chọn các con số fix cứng như 14px, 60% trang, hay 6px?**: Các thông số này là ngưỡng thực nghiệm dựa trên độ phân giải chuẩn của tài liệu PDF học thuật (thường là 72 DPI hoặc 300 DPI). Để hệ thống linh hoạt hơn, có thể chuyển thành tỷ lệ tương đối dựa trên font_size của đoạn văn bản đó để tự động co giãn.

- **Thay vì tìm horizontal line thủ công, sao không dùng pdfminer hay camelot?**: pdfminer không detect vector drawings là fraction bars. camelot/tabula chuyên cho bảng biểu (table), không xử lý fraction trong công thức inline. PyMuPDF có API `page.get_drawings()` cho phép truy cập trực tiếp vào vector graphics — đây là cơ sở để detect fraction bar một cách deterministic, không cần ML.

- **Nếu phân số lồng nhau (nested fractions) thì sao?**: [Chưa có] Hiện tại `_build_fractions()` xử lý từng bar độc lập. Với nested fractions, thuật toán chỉ xử lý được lớp ngoài cùng — lớp trong cũng sẽ được detect như một bar riêng. Kết quả có thể là `((a)/(b))/(c)` thay vì LaTeX `\frac{\frac{a}{b}}{c}` — vẫn đọc được nhưng không hoàn hảo.

- **Làm sao tránh false positive — đường line watermark bị nhầm thành fraction bar?**: `_fraction_bars()` lọc: chiều rộng bar phải >= 14px và <= 60% chiều rộng trang (loại watermark/border toàn trang). `_build_fractions()` thêm: check `embedded` (tránh overlines), check khoảng cách tối thiểu 6px giữa numerator và denominator, giới hạn số chars trong num/den <= 40.

---

## Slide 7 — PyMuPDF: Geometric Space

- **Tại sao trung bình dùng median không phải mean?**: Giá trị trung vị giúp hệ thống không bị nhiễu bởi các khoảng cách siêu lớn (như khoảng cách giữa 2 đoạn văn hoặc khoảng cách thụt lề đầu dòng).

- **Tại sao ngưỡng 60%?**: Khoảng cách giữa các ký tự trong cùng một từ thường rất nhỏ (dưới 20% chiều rộng ký tự). Khi khoảng cách vọt lên trên 60%, đó gần như chắc chắn là ranh giới giữa hai từ (word boundary).

- **Nếu trang vừa có chữ tiêu đề to (24px) vừa có chữ nội dung nhỏ (10px), median bị lệch — xử lý thế nào?**: [Chưa có] Đúng là tính chung cho cả trang sẽ bị lệch khi có nhiều cỡ font. Để tối ưu, có thể nhóm các ký tự cùng font-size lại thành từng cụm riêng, sau đó tính median space width độc lập cho từng nhóm. Nhờ vậy chữ to hay nhỏ đều được chèn khoảng cách chính xác theo tỷ lệ riêng.

- **Với tài liệu căn lề đều hai bên (Justified), khoảng cách bị kéo giãn — thuật toán có nhầm không?**: Với văn bản justified, khoảng cách giữa các từ lớn hơn bình thường nhưng khoảng cách giữa các ký tự trong cùng một từ vẫn rất nhỏ. Thuật toán dựa trên sự thay đổi ĐỘT NGỘT giữa khoảng cách ký tự nội bộ và khoảng cách từ, nên justified text chỉ làm gap rõ hơn, không chèn thừa space vào giữa từ.

- **Hệ thống có xử lý được text bị xoay 90° (rotated text) không?**: [Chưa có] `_collect_chars()` đọc coordinates nhưng không có bước xử lý rotation. Text bị xoay 90° sẽ có tọa độ x, y hoán đổi — thuật toán sort theo x0 và tính gap sẽ cho kết quả sai. Đây là edge case chưa được handle.

---

## Slide 8 — Heading Detection & Boilerplate Removal

- **Tại sao ngưỡng boilerplate là 50% số trang?**: `max(3, int(n_pages * 0.5))` trong `_boilerplate_lines()`. 50% đảm bảo chỉ loại bỏ những gì thực sự lặp lại ở đa số trang. Nếu đặt thấp hơn (20%) sẽ loại nhầm nội dung quan trọng xuất hiện nhiều trang. Threshold tối thiểu 3 để tránh flag trên tài liệu quá ngắn.

- **Nếu tài liệu chỉ có 2 trang, boilerplate detection có hoạt động không?**: Từ `_boilerplate_lines()`: `if n_pages < 3: return set()` — với tài liệu ≤ 2 trang, không chạy detection, giữ nguyên tất cả content. Safety guard vì với 2 trang không đủ dữ liệu thống kê để phân biệt boilerplate.

- **Heading detection fail với những trường hợp nào?**: Regex `_HEADING_RE` chỉ match format số (1.2.3), Roman numeral (I., VI.), Bước N:, và một số từ khóa cố định. Fail với: (1) heading dùng chữ cái (A., B., C.), (2) heading tiếng Việt không có số và không nằm trong danh sách keyword, (3) heading chữ thường hoặc font size không lớn hơn 1.15× median.

- **Điều kiện "chữ lớn hơn bình thường ≥ 1.15×" — median được tính như thế nào?**: `_median_font_size(chars)` tính median của tất cả font size trên trang. Heading check: `size >= max(median_size * 1.15, 11.0)`. Ngưỡng tối thiểu 11pt để không nhầm text nhỏ trong document không có heading.

- **Tại sao heading bị loại khỏi block text thay vì giữ lại?**: Heading được extract làm `section` label — không nằm trong `text` của Block. Khi search, chunk header `"Tên tài liệu › Tên section"` được prepend vào text để embed. Heading bản thân không chứa nội dung cần retrieve — nó chỉ là "địa chỉ" ngữ cảnh.

---

## Slide 9 — VLM Fallback

- **Tại sao dùng gpt-4o-mini vision thay vì Tesseract OCR (miễn phí, local)?**: Tesseract chỉ làm OCR (nhận dạng ký tự) — không hiểu cấu trúc, không xử lý công thức toán hay bảng phức tạp. gpt-4o-mini vision trả về Markdown có cấu trúc: `| col1 | col2 |` cho bảng, `$\frac{a}{b}$` cho công thức — đúng format để chunker xử lý tiếp. Tradeoff: tốn tiền mỗi trang nhưng chất lượng cao hơn nhiều.

- **Khi `VLM_PARSE=auto`, ngưỡng "trang trống" là gì chính xác?**: Từ `pdf_extract.py`: sau boilerplate removal, nếu `sum(len(t) for t, _ in cleaned) < 3` thì trang đó được xếp vào `empty_pages`. Con số 3 ký tự để bắt các trang chỉ còn số trang sau khi boilerplate filter.

- **VLM fallback có xử lý được chữ viết tay không?**: Có thể nhận dạng được trong nhiều trường hợp chữ viết rõ ràng — gpt-4o-mini vision được train trên large multimodal corpus. Tuy nhiên chữ viết tay phức tạp hoặc kém rõ vẫn có thể bị nhầm. Hệ thống không có bước post-process đặc biệt cho handwriting.

- **Chi phí VLM cho tài liệu scan 50 trang là bao nhiêu?**: Mỗi trang ~3.500đ (estimate từ slide) → 50 trang ≈ 175.000đ. Với `VLM_PARSE=auto`, chỉ trang empty mới dùng VLM — PDF scan thường có 100% trang empty → toàn bộ 50 trang dùng VLM. Cache bằng SHA-256 hash của PNG bytes → nếu cùng 1 trang upload lại sẽ không gọi API lần 2.

- **Config `VLM_PARSE=on` khác `auto` thế nào?**: `auto` = chỉ dùng VLM cho trang empty (text < 3 chars). `on` = dùng VLM cho mọi trang kể cả trang đã có text — hữu ích khi PyMuPDF extract được text nhưng layout vẫn sai (ví dụ 2-column confusion). `off` = không dùng VLM, chỉ PyMuPDF local.

---

## Slide 10 — Reducto Integration

- **Reducto khác gì VLM fallback — cả hai đều dùng AI?**: VLM fallback tự chụp ảnh từng trang rồi gọi gpt-4o-mini vision với prompt đơn giản "chuyển thành Markdown". Reducto là dịch vụ chuyên biệt dùng pipeline riêng: layout detection, reading order analysis, table structure recognition. `agentic` mode còn AI-review từng element. Reducto xử lý tốt hơn với layout phức tạp (multi-column, nested tables).

- **Tại sao Reducto trả về `embed_text` riêng thay vì raw text?**: Reducto generate bản tóm tắt semantic của mỗi block — ví dụ "Bảng so sánh điểm chuẩn 5 năm theo ngành" thay vì raw table data. Khi dùng `embed_text` để embed, semantic search hoạt động tốt hơn vì vector đại diện cho ý nghĩa. Hệ thống ưu tiên dùng `embed_text` nếu có, fallback về raw text nếu không.

- **Nếu REDUCTO_API_KEY hết quota thì document ở trạng thái nào?**: [Chưa có] Exception từ Reducto API propagate lên, document sẽ ở trạng thái `failed`. Không có xử lý đặc biệt cho quota exceeded hay fallback về PyMuPDF khi Reducto fail.

- **`REDUCTO_FILTER_BLOCKS` default là gì và tại sao?**: Default `"Header,Footer,Page Number"` — Reducto tự detect các element type này và filter ra trước khi trả về content. Tương đương với boilerplate removal của PyMuPDF nhưng chính xác hơn vì AI-driven thay vì heuristic tần suất.

---

## Slide 11 — Underthesea: Tokenizer tiếng Việt

- **Underthesea dùng algorithm nào để tách từ?**: Underthesea dùng CRF (Conditional Random Field) hoặc BiLSTM model được train trên corpus tiếng Việt annotated. Đây là sequence labeling — mỗi syllable được gán nhãn B (Begin), I (Inside), O (Outside word boundary). Model học pattern từ ngữ cảnh xung quanh.

- **Tại sao Underthesea được load lazy (lần đầu tiên gọi)?**: Từ `vn_text.py`: dùng double-checked locking. Underthesea là heavy import (~100–200MB model). Lazy load giúp server respond nhanh hơn khi khởi động — model chỉ load khi cần, và được force-init một lần trên main thread để tránh race condition.

- **Tại sao tách compound (underscore/camelCase)?**: Code variable `compute_loss` có thể xuất hiện trong document nhưng user query là "compute loss" (có space). Nếu chỉ lưu `compute_loss` → BM25 không match. Hệ thống emit cả 3: `compute_loss`, `compute`, `loss` → match được cả hai cách viết. Domain-specific optimization cho tài liệu có code.

- **Cache `_tok_cached` 4096 entries có đủ không?**: `@lru_cache(maxsize=4096)` cache theo text string. Phần lớn chunks được tokenize một lần khi ingest và lưu trong BM25 index. Khi query thì chỉ tokenize query string ngắn. 4096 entries rất đủ cho use case thực tế.

- **Nếu Underthesea không cài được, hệ thống có fallback không?**: [Chưa có] Nếu `from underthesea import word_tokenize` fail thì BM25 tokenization crash. Không có fallback về simple whitespace split. Trong Docker image, Underthesea đã install sẵn nên không phải vấn đề ở deployment thực tế.

---

## Slide 12 — Chunking: Tại sao cần?

- **"Lost in the middle" problem là gì?**: Nghiên cứu về attention mechanism của LLM cho thấy khi context window rất dài, model có xu hướng bỏ qua thông tin ở giữa — chỉ chú ý tốt ở đầu và cuối context. Nếu chunk quá lớn nhét nhiều chủ đề vào, thông tin ở giữa có thể bị LLM bỏ qua khi generate câu trả lời.

- **Tại sao overlap 200 ký tự thay vì 100 hay 300?**: 200 ký tự ≈ 1–2 câu tiếng Việt — đủ để giữ context ở ranh giới giữa 2 chunks mà không làm chunk quá nhiều nội dung trùng lặp. 100 ký tự quá ít, ranh giới vẫn bị mất. 300 ký tự thì 30% chunk là overlap — lãng phí index space. Config cho phép override qua `CHUNK_OVERLAP`.

- **Với `CHUNK_MAX_CHARS=1000`, một chunk trung bình chứa bao nhiêu câu?**: Câu tiếng Việt trung bình khoảng 100–150 ký tự → một chunk ≈ 6–10 câu. Đủ chứa 1–2 đoạn văn hoặc 1 ý hoàn chỉnh. Không quá dài để vector bị loãng, không quá ngắn để thiếu ngữ cảnh.

- **Tại sao không dùng semantic chunking (chia theo ý nghĩa)?**: Semantic chunking cần LLM hoặc embedding để detect topic boundaries — tốn chi phí per-chunk và tăng ingest time đáng kể. Approach hiện tại (structural + sentence + character count) là free, deterministic, đủ tốt cho corpus học thuật có cấu trúc rõ ràng. Semantic chunking được đề cập trong Future Roadmap.

---

## Slide 13 — Chunking: Structural Split

- **Làm sao hệ thống biết một dòng là công thức toán?**: Từ `chunker.py`: `_FORMULA_RE` match: (1) ký hiệu toán trong `_MATH_SYMBOLS` (∑√∏∫≤≥≠⇒→←±·×÷∈∀∅∩∪⊆≡∇∂), (2) pattern `)/(`  (phân số sau fraction reconstruction), (3) ký tự `$` (LaTeX delimiter), (4) `" | "` (Markdown table pipe). Bất kỳ dòng nào chứa ít nhất 1 pattern → atomic.

- **Bảng Markdown được detect như thế nào?**: `_is_atomic_line()` check `" | " in s` — dòng chứa pipe với space → là bảng Markdown row → atomic. Ngoài ra check `s.startswith("[BẢNG]")` — tag đặc biệt từ Reducto parser.

- **Nếu công thức quá dài (> 1000 ký tự) thì sao?**: Trong `_pack()`: atomic units giữ nguyên kể cả khi vượt `max_chars`. Không có size check với atomic lines. Công thức rất dài sẽ tạo ra chunk > 1000 ký tự, nhưng đây là đúng behavior — không thể cắt ngang công thức.

- **Nếu tài liệu không có công thức toán gì, structural split có tác dụng không?**: Vẫn có ích: nếu có bảng Markdown (`|`), chúng được giữ nguyên. Với pure text, tất cả đều là prose → `_split_units()` gom thành sentences bình thường. Structural split gracefully degrade về sentence splitting khi không có math/table.

---

## Slide 14 — Chunking: Sentence Segmentation & Smart Packing

- **Nếu 1 câu đơn dài hơn 1000 ký tự, xử lý thế nào?**: `_pack()`: `if len(text) > max_chars` → gọi `_smart_split_long()`. Hàm này tìm điểm cắt tại `,;)` sau 40% của max_chars — cắt tại ranh giới logic, không cắt giữa từ. Nếu không tìm được điểm cắt hợp lệ, cắt cứng tại max_chars.

- **Regex bảo vệ viết tắt hiện tại cover được gì?**: `_ABBREV_RE` match pattern `[Chữ_hoa][chữ_thường]*\.` + space — detect viết tắt kiểu "U.S.", "e.g.", "T.P.", "GS.TS." Không cover dấu chấm sau số (ví dụ "mục 3.4.") hay viết tắt toàn chữ hoa ("TPHCM. Theo..."). Đây là heuristic, không phải comprehensive solution.

- **Overlap 200 ký tự implement ở cấp câu hay ký tự?**: `_overlap_tail()` lấy các **câu** từ cuối chunk hiện tại sao cho tổng ≤ overlap_chars. Sentence-level overlap — câu không bị chặt giữa chừng trong phần overlap.

- **Tại sao không dùng Underthesea's `sent_tokenize()` thay vì regex tự viết?**: Underthesea `sent_tokenize()` nặng hơn và có latency cao hơn khi gọi nhiều lần trên các block text nhỏ. Regex-based split với abbreviation protection đủ chính xác cho văn bản học thuật có dấu câu chuẩn. Underthesea phù hợp hơn cho document dài cần tokenize 1 lần.

---

## Slide 15 — Contextual Chunk Headers

- **Nếu tài liệu không có section headers, chunk header trông như thế nào?**: Từ `store.py` → `_index_text()`: `head = " › ".join(p for p in (title, section) if p)`. Nếu `section=None`, header chỉ là tên tài liệu: `"Tên file.pdf\n[chunk text]"`. Nếu cả hai đều None — không có header, chỉ raw text.

- **Tại sao không dùng LLM để generate context cho từng chunk như Anthropic's Contextual Retrieval?**: LLM-based contextual retrieval cần 1 LLM call per chunk — document 200 trang × 5 chunks/trang = 1000 LLM calls → rất tốn kém. Approach của dự án dùng structural metadata (title + heading) extract free từ PDF — gần như 0 cost, đủ tốt khi heading rõ ràng.

- **Header có được lưu trong DB hay chỉ trong search index?**: Header là `embed_text` (text dùng để embed và index BM25), còn `text` (display text) là raw chunk không có header. Response trả về `text` sạch để hiển thị citation cho người dùng. Header chỉ tồn tại trong Turbovec vector và BM25 token list.

- **`_INDEX_VERSION = 4` trong store.py là gì?**: Khi tokenization logic hay `_index_text()` format thay đổi, bump version này → khi startup so sánh với version trong `index_meta.json`, nếu khác → force rebuild toàn bộ index. Đảm bảo không bị stale index sau update code.

---

## Slide 16 — Embedding: OpenAI + L2 Normalize

- **Tại sao không dùng embedding model local như bge-m3?**: Local model như `BAAI/bge-m3` cần PyTorch, RAM ~1–2GB, inference chậm hơn trên CPU-only. Với tập tài liệu vừa, OpenAI API + embedding cache là đơn giản hơn. bge-m3 được đề cập trong Future Roadmap khi cần giảm cost hoặc tăng privacy.

- **Cache embedding dùng SHA-256 key — nếu model thay đổi mà key không đổi, có dùng nhầm vector cũ không?**: Không. Từ `embeddings.py`: key = `SHA256(f"{embed_model}::{text}")` — model name được include trong hash. Thay `EMBED_MODEL` → key khác hoàn toàn → cache miss → gọi API lại với model mới. An toàn.

- **L2 normalization tại sao giúp dot product = cosine similarity?**: Cosine similarity(A,B) = A·B / (‖A‖ × ‖B‖). Sau L2 normalize: ‖A‖ = ‖B‖ = 1 → cosine = A·B. Turbovec dùng SIMD-optimized dot product vì nó nhanh hơn cosine với phép chia. Normalize trước giúp tận dụng tối ưu này mà không thay đổi semantic ranking.

- **Embedding được store dạng gì trong SQLite cache?**: `np.ndarray` float32 serialize thành raw bytes (`vec.astype(np.float32).tobytes()`) → lưu dưới dạng BLOB. Mỗi vector 1536 × 4 bytes = 6144 bytes ≈ 6KB per cache entry.

---

## Slide 17 — Turbovec & TurboQuant

- **TurboQuant tính bucket ngưỡng như thế nào mà không cần training data?**: Sau khi xoay ngẫu nhiên, mỗi tọa độ tuân theo phân phối Gaussian N(0, 1/D). Phân phối Gaussian có analytical form → có thể tính sẵn N điểm phân vị (quantile points) bằng công thức toán học (inverse CDF của Gaussian). Với 4-bit = 16 buckets: tính 15 ngưỡng từ toán học, không cần xem data. Đây là điểm cốt lõi khác với PQ cần k-means.

- **Tại sao xoay ngẫu nhiên giúp đạt phân phối Gaussian?**: Kết quả từ concentration of measure và Johnson-Lindenstrauss lemma. Khi chiếu dữ liệu lên một hướng ngẫu nhiên trong không gian chiều cao, dù data gốc có phân phối bất kỳ, projection sẽ gần với Gaussian theo Central Limit Theorem effect. Trong không gian 1536 chiều, hiệu ứng này rất mạnh.

- **"Bù bias" từ RaBitQ (SIGMOD 2024) cụ thể là gì?**: RaBitQ nhận thấy lượng tử hóa gây ra systematic bias trong inner product estimation. Giải pháp: lưu thêm thông tin về norm gốc của vector trước khi quantize, dùng để correct bias khi tính similarity. Với 4-bit quantization, correction này cải thiện recall lên ~1–3%.

- **Thêm vector mới sau khi index đã build, có cần rebuild không?**: Không. Thêm document → embed → `vector.add()` để append vào Turbovec index trong memory và persist lại. BM25 cũng append và mark dirty (rebuild lazy). Không có rebuild toàn bộ index.

- **So sánh Turbovec với FAISS IndexFlatIP (exact search)?**: FAISS IndexFlatIP dùng float32 full precision, recall 100% exact. Turbovec 4-bit: recall ~99%, nhưng 8× ít RAM hơn, SIMD-optimized nhanh hơn trên ARM (NEON). Với corpus nhỏ-vừa, ~1% recall loss là chấp nhận được.

---

## Slide 18 — Exhaustive SIMD Search

- **Khi nào mới cần chuyển từ exhaustive sang ANN (HNSW/IVF)?**: Theo benchmark trong slide: 100K vectors ~3ms, 1M vectors ~30ms. Nếu corpus > 500K chunks và latency < 10ms là hard requirement → cần HNSW. Với use case hiện tại (vài nghìn đến vài chục nghìn chunks), exhaustive hoàn toàn đủ và không phải bottleneck.

- **AVX-512 vs AVX2 vs NEON — Turbovec detect hardware tự động không?**: Turbovec (viết bằng Rust) dùng compile-time feature detection hoặc runtime CPUID check. Trên Apple Silicon (ARM) → tự động dùng NEON 128-bit SIMD. Trên x86 → kiểm tra AVX2/AVX-512. Người dùng không cần config gì.

- **"Recall 100% trên dữ liệu nén" — không phải 100% so với exact search?**: Đúng. "Recall 100%" ở đây nghĩa là exhaustive scan — không bỏ sót chunk nào trong index (không dùng approximation để skip candidates). Nhưng vì vector đã quantize 4-bit, ranking có thể khác đôi chút so với exact float32. Recall@K so với exact search là ~99%, không phải 100%.

- **So với FAISS IVF (Inverted File Index), tradeoff là gì?**: FAISS IVF phân cluster → search O(N/nlist), nhanh hơn exhaustive nhưng cần training (k-means), recall thấp hơn (~95–98%), và thêm/xóa vector phức tạp hơn. Turbovec exhaustive phù hợp cho corpus động (upload/xóa tài liệu thường xuyên) vì không cần rebuild index structure.

---

## Slide 19 — BM25 Keyword Search

- **Giá trị k1=1.5, b=0.75 mặc định có được tune cho tiếng Việt không?**: [Chưa có] Dùng default của `rank_bm25` library — giá trị empirically validated trên nhiều dataset tiếng Anh. Chưa có tuning experiment riêng cho tiếng Việt. Về lý thuyết, tiếng Việt có câu ngắn hơn tiếng Anh, b (length normalization) có thể cần điều chỉnh, nhưng cần benchmark để confirm.

- **BM25 có hiểu "ngưỡng điểm" = "điểm chuẩn" không?**: Không — BM25 là exact token matching với IDF weighting. Đây chính là lý do cần Dense embedding (hiểu ngữ nghĩa) và Hybrid search. BM25 giỏi exact match, Dense giỏi semantic match — cả hai bổ sung cho nhau.

- **Khi thêm document mới, BM25 index rebuild như thế nào?**: Từ `bm25_index.py`: `add()` append chunk_id và tokens vào list, mark `_dirty=True`. Khi `search()` được gọi tiếp theo → detect `_dirty` → gọi `_rebuild()` → tạo `BM25Okapi(self._corpus_tokens)` mới. Lazy rebuild — không rebuild ngay khi add, chỉ rebuild khi cần search.

- **BM25 có persist ra disk không?**: Có. `save()` dùng pickle serialize `chunk_ids` và `corpus_tokens` ra `bm25.pkl`. Khi khởi động lại: `BM25Index.load(path)` đọc pickle, set `_dirty=True` → rebuild BM25Okapi lần đầu tiên search. Token list được lưu, không cần retokenize toàn bộ corpus.

---

## Slide 20 — Hybrid Search & RRF

- **Tại sao k=60 trong RRF là "empirically optimal"?**: Từ bài báo gốc (Cormack, Clarke, Buettcher — SIGIR 2009): k=60 được tìm ra qua thực nghiệm trên nhiều TREC test collections cho kết quả tốt nhất. k=60 balance: nhỏ quá (k=10) → top rank quá dominant, lớn quá (k=100) → tail rank có ảnh hưởng không cần thiết.

- **Nếu BM25 chỉ có 5 results và Dense có 30, RRF xử lý thế nào?**: RRF cộng score từ mỗi list độc lập. Chunk chỉ trong Dense nhưng không trong BM25 chỉ có score từ Dense rank: `1/(60+rank_dense)`. Chunk có trong cả 2 list có score cộng dồn → score cao hơn. Chunks từ Dense-only vẫn xuất hiện nhưng score thấp hơn chunks được cả 2 đồng thuận.

- **Tại sao không dùng weighted sum α·BM25 + β·Dense?**: BM25 score range 0–∞, Dense score range -1–1. Phải normalize trước khi cộng. Và α, β tối ưu khác nhau cho từng query — không có giá trị toàn cục tốt. RRF rank-based không cần normalize và không có hyperparameter cần tune → robust hơn.

- **Khi nào nên tăng BM25_TOP_K/DENSE_TOP_K lên cao hơn 30?**: Khi corpus lớn và recall quan trọng hơn latency. Nhưng reranker vẫn chỉ nhận `RERANK_TOP_N=20` từ RRF merge → phải tăng cả `RERANK_TOP_N` để hưởng lợi từ top_k lớn hơn. Bottleneck vẫn là reranker.

---

## Slide 21 — Cross-Encoder Reranker

- **BAAI/bge-reranker-v2-m3 có hỗ trợ tiếng Việt thực sự không?**: `bge-reranker-v2-m3` là multilingual model support 100+ languages bao gồm tiếng Việt, được BAAI train với data đa ngôn ngữ. "m3" là viết tắt của "multilingual, multi-functionality, multi-granularity". Đây là lý do chọn model này thay vì bge-reranker-base (chỉ tiếng Anh).

- **Reranker được load lên memory khi nào?**: Lazy load trong `_load()`. Model được load lần đầu khi `rerank()` được gọi (request đầu tiên cần retrieval). Sau đó ở trong memory cho đến khi container restart. Lần load đầu có thể mất 5–30 giây.

- **Nếu reranker fail (model không load được), hệ thống có fallback không?**: Có. Từ `reranker.py`: nếu `FlagReranker` init fail → `self._failed = True`. `available` property trả về `False`. Trong store: nếu `USE_RERANKER=True` nhưng reranker unavailable → skip rerank, dùng thẳng RRF score. Graceful degradation.

- **`normalize=True` trong `compute_score()` nghĩa là gì?**: FlagReranker trả về raw logit scores (unbounded). `normalize=True` áp dụng sigmoid → range 0–1. Không ảnh hưởng đến ranking order (sigmoid monotonic), chỉ scale output để dễ interpret và so sánh cross-query.

- **Tại sao cần `_infer_lock` riêng ngoài `_lock`?**: `_lock` chỉ dùng khi init model (1 lần). `_infer_lock` dùng mỗi lần gọi `compute_score()`. Trong complex route, distill+verify chạy parallel với `ThreadPoolExecutor` → nhiều thread có thể gọi reranker đồng thời. HuggingFace model forward pass không thread-safe → `_infer_lock` serialize inference calls, tránh data race.

---

## Slide 22 — Agent Graph: Kiến trúc tổng quan

- **Router có thể route "no_retrieval" nhưng code lại force override sang "simple" — tại sao?**: Từ `graph.py` → `_route()`: nếu Router LLM trả về "no_retrieval", code override thành "simple" với lý do "Buộc tra cứu tài liệu (tránh bỏ sót)". Chỉ 2 trường hợp thực sự bypass retrieval: (1) regex detect greeting/chitchat, (2) không có document nào trong index. Triết lý: thà retrieve thừa còn hơn bỏ sót context.

- **Planner giới hạn ≤4 sub-questions — cơ sở của con số này?**: `subqs[:4]` trong `_plan()`. 4 sub-questions × (retrieve + distill + verify) = ~12 LLM calls per iteration × 3 iterations = ~36 calls worst case. Balance giữa coverage và cost. Config không expose riêng nhưng có thể thay đổi trong code.

- **Tại sao không có node "Clarify" để hỏi lại user khi câu hỏi không rõ?**: [Chưa có] Hiện tại hệ thống luôn cố gắng trả lời, không có interaction loop. Router chỉ classify vào 3 route, không phát hiện ambiguous query. Thêm Clarify node cần giao thức 2-way interaction phức tạp hơn qua SSE.

- **Nếu không có tài liệu nào được upload, Agent xử lý thế nào?**: Từ `_route()`: `if not self.kb.vector.ready and self.kb.bm25.count == 0: return "no_retrieval", "Chưa có tài liệu nào được nạp."` → route thẳng sang answer để trả lời là chưa có tài liệu, không cố retrieve.

---

## Slide 23 — Agent Graph: Flow chi tiết

- **Distill + Verify chạy parallel dùng gì?**: Từ `graph.py`: `ThreadPoolExecutor(max_workers=min(4, len(subqs)))` — tối đa 4 thread, mỗi thread xử lý 1 sub-question (retrieve → distill → verify). Giảm latency khi có nhiều sub-questions.

- **Nếu 1 sub-question throw exception trong distill, flow tiếp tục hay dừng?**: Từ code: trong `_distill_and_verify()`, có `try/except` → trả về `{"error": True, "relevant": False, "grounded": False, "reason": str(e)}`. Flow tiếp tục với các sub-questions khác. Error được emit qua SSE nhưng không dừng toàn bộ agent.

- **Simple route bỏ qua distill/verify — rủi ro gì?**: Simple route retrieve → synthesize trực tiếp, không có kiểm chứng trung gian. LLM có thể hallucinate nếu retrieved chunks có thông tin mâu thuẫn. Tuy nhiên vẫn có `verify_answer` ở cuối pipeline kiểm tra câu trả lời cuối có grounded không — đây là safety net duy nhất cho simple route.

- **Distill node làm gì cụ thể?**: Gọi `DISTILL_SYSTEM` prompt với (sub-question, retrieved chunks) → LLM trích xuất thông tin liên quan, loại bỏ nhiễu, trả về text summary hoặc `"KHÔNG_LIÊN_QUAN"`. Output này được verify ở bước tiếp theo. Mục đích: giảm noise trước khi synthesize.

---

## Slide 24 — Agent Graph: Replanning Loop

- **"Failed count không giảm" — so sánh với vòng nào?**: Từ `graph.py`: `prev_failed_count` được lưu và update mỗi vòng. `if current_failed_count >= prev_failed_count: early_stop`. Nếu vòng này số failed ≥ vòng trước → không cải thiện → dừng sớm. Tránh tiêu tốn thêm LLM calls cho queries mà system không thể resolve.

- **Khi `early_stop`, hệ thống trả lời như thế nào?**: Flow tiếp tục sang Synthesize với thông tin tốt nhất đã thu thập được (grounded notes từ các bước đã thành công). Câu trả lời có thể là partial với disclaimer. Không return error — vẫn cố trả lời với gì có được.

- **Tại sao MAX_REPLAN_ITERS=3 mà không phải 5 hay 10?**: Mỗi iteration thêm nhiều LLM calls. 3 iterations là tradeoff giữa thoroughness và cost/latency. 5 iterations có thể tốt hơn cho câu hỏi cực khó nhưng cost tăng đáng kể và user experience tệ hơn (đợi lâu hơn). Config cho phép tune `MAX_REPLAN_ITERS`.

- **Conversation history > 12 messages được summarize như thế nào?**: Từ `graph.py` → `summarize_conversation()`: dùng `SUMMARIZE_SYSTEM` prompt với toàn bộ conversation → LLM generate summary text. Summary được lưu trong DB, inject vào system message lần chat tiếp. `SUMMARY_MODEL` default là `llm_model_fast` (gpt-4o-mini fast mode).

---

## Slide 25 — System Prompts

- **Chain-of-Thought tốn thêm bao nhiêu token?**: CoT yêu cầu LLM "lập luận từng bước" trước khi kết luận → thường tăng output 2–3× so với direct answer. Đây là lý do chỉ dùng CoT cho Answer và Regenerate — 2 node quan trọng nhất. Các node phân loại (Router, Verify...) không cần CoT, chỉ cần output JSON ngắn.

- **Tất cả prompts bằng tiếng Việt — nếu user hỏi bằng tiếng Anh, câu trả lời ra sao?**: gpt-4o-mini là multilingual — sẽ tự động trả lời bằng ngôn ngữ của câu hỏi ngay cả khi system prompt bằng tiếng Việt. Citation format `[1]`, `[2]` là hard-coded nhưng không phụ thuộc ngôn ngữ.

- **Regenerate prompt khác gì Answer prompt?**: Regenerate nhận thêm `verify_reason` (lý do tại sao Answer bị flag ungrounded) và được yêu cầu "chỉ dùng thông tin từ context, không thêm gì ngoài context". Nghiêm ngặt hơn Answer prompt để đảm bảo câu trả lời lần 2 không hallucinate.

- **JSON mode của OpenAI có đảm bảo 100% valid JSON không?**: OpenAI JSON mode đảm bảo output là valid JSON. Nhưng key names và value types vẫn có thể không đúng schema mong đợi. Trong `chat_json()`: có fallback regex `raw[start:end+1]` để extract JSON nếu có text bao quanh. Nếu JSON parse vẫn fail → raise exception.

---

## Slide 26 — Evaluation Framework

- **LLM-as-judge có vấn đề gì về consistency?**: LLM judge có thể: (1) position bias — ưu tiên candidate xuất hiện trước trong prompt, (2) verbosity bias — ưu tiên câu trả lời dài hơn, (3) inconsistency — cùng input chạy 2 lần cho kết quả khác nhau. Để mitigate: cache kết quả judge (không chạy lại), dùng temperature thấp cho judge.

- **Recall@5, MRR@5, Hit@5 — đâu là metric quan trọng nhất?**: Recall@5 quan trọng nhất — nếu bỏ sót chunk có câu trả lời, agent không thể trả lời đúng dù algorithm có tốt đến đâu. MRR@5 quan trọng thứ 2 — đo xem chunk liên quan có ở đầu không (reranker lấy top để synthesize). Hit@5 là binary — ít granular hơn.

- **Eval script có chạy được offline không?**: Không hoàn toàn. Eval dùng LLM judge gọi OpenAI API để đánh giá relevance. Nếu đã có cached judge results trong `eval_results.json` → có thể tính metrics offline. Query embedding cũng cần API trừ khi đã cache.

- **Eval framework test retrieval hay full agent?**: Từ slide: chạy 4 phương pháp (BM25, Dense, Hybrid, Hybrid+Rerank) → đây là retrieval-only eval. Không test full agent (planner, distill, verify, synthesize). Cung cấp thông tin về chất lượng retrieval, không phải chất lượng câu trả lời cuối.

---

## Slide 27 — Performance & Cost Analysis

- **Reranker chiếm 80–90% thời gian retrieval, có cách giảm không?**: Các hướng: (1) giảm `RERANK_TOP_N` từ 20 xuống 10 → nhanh 2× nhưng có thể miss; (2) dùng reranker nhỏ hơn (bge-reranker-base); (3) cache reranker result cho same query. Tuy nhiên 50–100ms là chấp nhận được khi so với LLM calls (~500ms+).

- **Worst case 26K input tokens — gpt-4o-mini có context limit nào?**: gpt-4o-mini có context window 128K tokens. 26K input (worst case 3 replan iterations) + ~5K output = 31K total — thoải mái trong limit. Cost tuyến tính với token count.

- **Embedding batch_size=128 — nếu document có 1000 chunks cần bao nhiêu API calls?**: `ceil(1000 / 128) = 8` API calls. Với cache: nếu chunk text giống với documents cũ → cache hit → không gọi API. Cache key = SHA256(model::text).

- **Tổng chi phí cho 1 ngày 100 câu hỏi simple + 50 câu phức tạp là bao nhiêu?**: Simple ~$0.005/câu × 100 = $0.50. Phức tạp ~$0.015/câu × 50 = $0.75. Cộng embedding cho upload mới. Tổng ~$1–2/ngày với mức sử dụng trên — rất rẻ cho production RAG.

---

## Slide 28 — Software Architecture: Interrupt & Recovery

- **Nếu server crash giữa chừng khi đang ingest, DB có bị corrupt không?**: SQLite WAL mode cung cấp atomic transactions — nếu crash, WAL file chưa committed → SQLite auto rollback khi restart. Document sẽ stuck ở `status=processing`. [Chưa có] Không có mechanism detect và recover stuck processing documents.

- **"Cooperative check" trong cancel — có thể mất bao lâu mới thực sự dừng?**: Agent check `_cancelled_sessions` trước mỗi LLM call. Mỗi bước có thể mất 0.5–2 giây (LLM latency). Worst case: agent đang ở giữa Synthesize (LLM call dài) → phải đợi call đó complete trước khi check cancel. Có thể mất 1–5 giây từ khi cancel đến khi thực sự dừng.

- **Nếu cả upload và chat chạy cùng lúc, có lock conflict không?**: Upload dùng `self._lock` trong KnowledgeBase khi ghi index. Chat chỉ đọc index (search) — không cần lock. SQLite WAL mode cho phép read không block write. Không conflict trong thực tế.

- **Trace snapshot được lưu vào DB như thế nào khi stream đang chạy?**: Mỗi event (trừ token streaming) được append vào `trace_snapshot` list và `update_message(msg_id, trace=json.dumps(...))` gọi ngay. Đây là SQLite UPDATE per event. WAL mode giúp concurrent reads không bị block bởi những writes này.

---

## Slide 29 & 30 — Upload, Chat & Cancel Flow

- **`AbortController.abort()` trên frontend — server có biết không?**: Server nhận TCP connection close signal → FastAPI raise `CancelledError` trong async generator → SSE stream kết thúc. Nhưng background thread (`_run_agent_sync`) là separate thread — không biết HTTP connection đã close → tiếp tục chạy cho đến khi hoàn tất hoặc bị cancel qua API.

- **Tại sao poll mỗi 2 giây thay vì WebSocket cho upload status?**: Upload là one-shot event (processing → ready/failed), không cần real-time updates như chat. Poll mỗi 2s đủ cho UX. WebSocket thêm complexity không justified cho một status check đơn giản.

- **`_cancelled_sessions` là in-memory set — nếu server restart, session stuck ở status gì?**: `_cancelled_sessions` reset (empty set) sau restart. Background thread đã die. Message trong DB vẫn ở `status=processing` — stuck indefinitely. [Chưa có] Cần cleanup job để detect và mark stale processing messages là `failed`.

- **Khi upload file, backend tạo record "processing" trước khi bắt đầu ingest — tại sao?**: Đảm bảo frontend có thể poll status ngay lập tức sau khi upload. Nếu tạo record sau khi ingest xong → frontend không biết đang có job chạy. "Processing" record là signal "có job đang chạy".

---

## Slide 31 — Limitations

- **"Không lọc input" — prompt injection có thể gây ra vấn đề gì?**: Nếu user upload tài liệu chứa text đặc biệt như "Ignore previous instructions and...", text này sẽ được chunk và embed vào index. Khi retrieve, nó có thể xuất hiện trong context → LLM đọc "instruction" giả → có thể bị manipulation. Hệ thống dùng system prompt "chỉ dùng thông tin từ ĐOẠN TRÍCH" nhưng không có input sanitization.

- **Hệ thống single-user nhưng vẫn có rate limiting — tại sao?**: Rate limiting (`SlowAPI`: 10 upload/phút, 30 chat/phút) bảo vệ OpenAI API key khỏi bị lạm dụng (script tự động spam), và giới hạn chi phí không mong muốn. Ngay cả single-user deployment cũng có thể bị script khai thác nếu endpoint không bảo vệ.

- **Nếu SQLite corrupt (ví dụ disk full khi đang write), recovery path là gì?**: [Chưa có] SQLite WAL mode cung cấp transaction safety nhưng không bảo vệ khỏi disk full hoàn toàn. Nếu DB corrupt: mất metadata, session, message history. Vector index (`.tvim`) và BM25 (`.pkl`) là separate files — có thể không bị ảnh hưởng. Không có backup mechanism hay disaster recovery.

- **"Đánh giá chưa chuẩn" — vấn đề cụ thể là gì?**: (1) LLM judge có thể biased (position, verbosity). (2) Chưa có golden test set tiếng Việt để so sánh. (3) Eval kết quả có thể không reproducible nếu LLM judge chạy với temperature > 0. (4) Chỉ eval retrieval, không eval end-to-end answer quality.

---

## Slide 32 — Future Roadmap

- **"Chunk thông minh theo ngữ nghĩa" (semantic chunking) — cụ thể là gì?**: Thay vì cắt theo số ký tự, semantic chunking embed từng câu và nhóm các câu có embedding gần nhau vào cùng chunk. Câu chuyển topic đột ngột → tạo chunk mới. Mỗi chunk ngữ nghĩa coherent hơn. Tradeoff: cần embed mỗi câu khi ingest → tốn N × API calls.

- **Tại sao cần chuyển sang Elasticsearch khi data lớn?**: Elasticsearch cung cấp: (1) BM25 distributed (không bị giới hạn RAM), (2) inverted index persist không cần load toàn bộ vào RAM, (3) horizontal scaling. SQLite + in-memory BM25 + Turbovec chỉ scale đến vài chục GB RAM trên 1 máy.

- **"Embedding local" — tradeoff cụ thể là gì?**: Model như `BAAI/bge-m3`: Pro — không cần API key, không gửi text ra ngoài, không tốn tiền per-call. Con — cần RAM ~2GB thêm, inference chậm hơn trên CPU-only so với OpenAI API (GPU-accelerated), chất lượng cho tiếng Việt academic cần benchmark để xác nhận.

- **"Feedback từ user để điều chỉnh reranker" — cơ chế?**: [Chưa có] Thu thập thumbs up/down trên câu trả lời → lưu (query, chunk, label) pairs → fine-tune cross-encoder reranker trên domain data. Hiện tại không có feedback table trong SQLite schema.

---

## Slide 33 — Deployment

- **Reranker model ~1GB — ảnh hưởng đến startup time thế nào?**: Lazy load — model chỉ load khi request đầu tiên trigger retrieval. Startup time của container không bị ảnh hưởng. Request đầu tiên cần retrieval sẽ phải chờ model load (~5–30 giây). Sau đó model ở trong memory cho đến khi container restart.

- **Tại sao Python 3.11-slim?**: Python 3.11 có performance improvement đáng kể so với 3.10 (10–15% faster). `-slim` image loại bỏ dev tools, docs → image size nhỏ hơn nhiều. Python 3.12 tại thời điểm build có thể chưa có stable wheel cho tất cả dependencies (PyTorch, underthesea).

- **Để scale lên nhiều máy, cần thay đổi gì?**: Cần: (1) PostgreSQL thay SQLite (shared persistent storage), (2) S3/shared volume cho vector index files (hiện tại local disk), (3) Redis/Celery cho background job queue (hiện tại ThreadPoolExecutor), (4) Shared session store (hiện tại in-process `_cancelled_sessions` set).

- **Vector index được load như thế nào khi startup?**: Từ `store.py` → `_load_indexes()`: đọc `index_meta.json` → check `_INDEX_VERSION` → nếu match, load `vector.tvim` và `bm25.pkl`. Nếu version mismatch → rebuild từ DB chunks. Load từ disk nhanh hơn rebuild vì không cần gọi OpenAI API.

---

## Slide 34 — Tổng kết

- **"Minh bạch hơn là tiện" — debug graph tự viết vs LangGraph dễ hơn hay khó hơn?**: Tự viết: trace log rõ ràng (`RAG_FLOW node_end node=X duration_ms=Y`), có thể đặt breakpoint ngay tại node function, flow là plain Python. LangGraph: visualization tool đẹp, nhưng khi lỗi xảy ra trong middleware layer → stack trace phức tạp hơn, cần hiểu LangGraph internals để debug.

- **Exhaustive search không xấp xỉ — nếu scale lên 1M chunks trong production?**: 1M vectors × 768 bytes (4-bit) = 768MB RAM. Scan time ~30ms theo benchmark — vẫn chấp nhận được nếu latency budget là 100ms+. Với 10M chunks (1.5GB RAM + ~300ms scan) sẽ cần chuyển sang ANN. Conscious tradeoff: correctness over speed cho scale hiện tại.

- **So với ChatPDF, NotebookLM — hệ thống này hơn và kém ở đâu?**: Hơn: (1) privacy (vector index local), (2) math-aware PDF parsing (LaTeX formula), (3) controllable agent với replanning (transparency), (4) Vietnamese-native. Kém: (1) không có Google Drive/Docs integration (NotebookLM), (2) single-user không có collaboration, (3) không có multimodal support (image search), (4) không có web search (Perplexity).

- **Tại sao dùng RRF thay vì learned fusion (train weight cho từng query type)?**: Learned fusion cần training data (labeled query-relevance pairs) và maintenance khi distribution thay đổi. RRF zero-shot, không cần training, empirically competitive với learned methods trên standard benchmarks. Phù hợp khi không có large labeled dataset cho tiếng Việt academic domain.

- **HyDE (Hypothetical Document Embeddings) là gì và có được dùng trong hệ thống không?**: HyDE là kỹ thuật: dùng LLM generate một "câu trả lời giả định" cho query, sau đó embed câu trả lời đó thay vì query gốc để search — vì câu trả lời giả định gần với document hơn về mặt semantic. Config `USE_HYDE=false` tồn tại trong `config.py` và `.env.example`. [Chưa có] Tuy nhiên `use_hyde` không được implement trong code retrieval hiện tại — chỉ là placeholder config chưa được triển khai.
