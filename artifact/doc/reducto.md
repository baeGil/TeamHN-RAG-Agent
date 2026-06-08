## Flow hoàn chỉnh: Upload PDF (Reducto) → Câu trả lời

Giả sử upload file `mô_hình_hóa_ĐATN.pdf` (tài liệu về mô hình hóa di chuyển an toàn trong môi trường bức xạ).

---

### PHASE 1: PARSING — Reducto API

```
POST /api/documents/upload  →  file: mô_hình_hóa_ĐATN.pdf
```

Backend gọi `_load_pdf_reducto()`:

**1a. Upload lên Reducto:**
```
client.upload(file="mô_hình_hóa_ĐATN.pdf")  →  file_id = "abc123"
```

**1b. Parse với config:**
```python
params = {
    enhance: {agentic: [{scope: "table", mode: "auto"}], summarize_figures: True},
    retrieval: {
        chunking: {chunk_mode: "page_sections", chunk_size: 1200},
        embedding_optimized: True,
        filter_blocks: ["Header", "Footer", "Page Number"]
    },
    formatting: {table_output_format: "dynamic", add_page_markers: True}
}
result = client.parse.run(input=file_id, **params)
```

Reducto trả về các **chunks**, mỗi chunk gồm nhiều blocks. Ví dụ chunk từ trang 1-2:

```python
chunk = result.result.chunks[0]
chunk.blocks = [
    Block(type="Section Header", content="Mô hình hóa bài toán di chuyển an toàn"),
    Block(type="Text", content="Robot phải thăm tất cả các mục tiêu trong S..."),
    Block(type="Text", content="Ràng buộc p₁ ≡ p_τP ∈ S..."),
]
chunk.content = "Mô hình hóa bài toán di chuyển an toàn\nRobot phải thăm tất cả các mục tiêu trong S..."
chunk.embed = "Robot navigation in radiation environments, visiting targets in set S with safety constraints S(c)=C1·(24−Nobs(c))/24+..."
```

**Điểm mấu chốt với Reducto:**
- `chunk.content` → thành `Block.text` (nội dung đầy đủ để hiển thị)
- `chunk.embed` → thành `Block.embed_text` (Reducto tự generate optimized summary cho embedding — **đây là lợi thế chính**)
- `chunk.blocks[0].type == "Section Header"` → thành `Block.section`

**1c. Convert sang Block:**
```python
# reducto_result_to_blocks() xử lý:
Block(
    page=1,
    section="Mô hình hóa bài toán di chuyển an toàn",  # ← từ Section Header
    text="Mô hình hóa bài toán di chuyển an toàn\nRobot phải thăm...",  # ← chunk.content
    embed_text="Robot navigation in radiation environments, visiting targets in set S with safety constraints..."  # ← chunk.embed
)
```

Reducto với `embedding_optimized=True` tự động tạo `embed` — một bản tóm tắt keyword-rich bằng tiếng Anh, tối ưu cho semantic search. Đây là bản "embed_text" riêng biệt với "text" hiển thị.

---

### PHASE 2: CHUNKING — Bỏ qua vì Reducto đã chunk

```python
# _ingest() trong store.py:
has_embed_text = any(getattr(b, "embed_text", None) for b in blocks)  # True!

# Vì Reducto đã chunk + có embed_text → KHÔNG gọi chunk_blocks()
chunks = [
    Chunk(text=b.text, page=b.page, section=b.section, embed_text=b.embed_text)
    for b in blocks if b.text.strip()
]
```

Khác với PyMuPDF/MinerU (cần chunk thêm), Reducto trả về sẵn chunks có section + embed_text. Nên bỏ qua bước chunking.

---

### PHASE 3: CCH — 4-tier Contextual Chunk Headers

Đây là bước thêm ngữ cảnh cho mỗi chunk. Chạy **2 loại LLM call** song song:

**3a. Tier 2 — Document Summary (1 LLM call):**

```python
full_text = " ".join(ch.text for ch in chunks)  # Toàn bộ tài liệu
sample = full_text[:24000]  # ~6000 tokens

# Gọi LLM:
prompt = "Tài liệu này là gì và nội dung chính của nó là gì? Trả lời bằng 1 câu duy nhất..."
# → "Tài liệu này trình bày về: mô hình hóa bài toán di chuyển an toàn của robot trong môi trường có bức xạ và vật cản, sử dụng FMF cải tiến."
doc_summary = "Tài liệu này trình bày về: mô hình hóa bài toán di chuyển an toàn của robot trong môi trường có bức xạ và vật cản, sử dụng FMF cải tiến."
```

**3b. Tier 4 — Section Summaries (N LLM calls, song song):**

```python
# Group chunks by section:
section_texts = {
    "Mô hình hóa bài toán di chuyển an toàn": ["Robot phải thăm...", "Ràng buộc p₁..."],
    "Hàm mục tiêu tổng quát": ["Hàm mục tiêu tổng quát có dạng...", "Risk(P) = sum(1-S(p))..."],
    "Tính Dij giữa các mục tiêu": ["Trong pha 1, Dij được tính bằng..."],
    "Thuật toán FMF cải tiến": ["T(x) là chi phí tích lũy tối ưu...", "f(x) = w1 + w2·R(x) + w3·(1-S(x))"],
}

# 1 LLM call per section, chạy song song (ThreadPoolExecutor):
# Section "Mô hình hóa bài toán di chuyển an toàn":
prompt = "Đây là nội dung phần 'Mô hình hóa...' trong tài liệu 'mô_hình_hóa_ĐATN'. Phần này trình bày về điều gì?..."
# → "Phần này trình bày về: định nghĩa bài toán di chuyển an toàn với ràng buộc chu trình và các biến quyết định."

# Section "Hàm mục tiêu tổng quát":
# → "Phần này trình bày về: hàm mục tiêu tổng quát gồm chiều dài, phơi nhiễm bức xạ và rủi ro va chạm với các trọng số w1, w2, w3."

# Section "Thuật toán FMF cải tiến":
# → "Phần này trình bày về: thuật toán Fast Marching Method cải tiến với hàm chi phí cục bộ f(x) và công thức giao thoa Dij."

section_summaries = {
    "Mô hình hóa bài toán di chuyển an toàn": "Phần này trình bày về: định nghĩa bài toán...",
    "Hàm mục tiêu tổng quát": "Phần này trình bày về: hàm mục tiêu tổng quát...",
    "Thuật toán FMF cải tiến": "Phần này trình bày về: thuật toán FMF cải tiến...",
}
```

**3c. Assemble CCH text cho mỗi chunk:**

Ví dụ chunk chứa công thức Risk(P):

```python
# Chunk gốc:
ch.text = "Risk(P) = sum(1 - S(p)) cho các ô p trong đường đi P. Khi P1 đi qua nhiều ô có S(p) thấp..."
ch.section = "Hàm mục tiêu tổng quát"
ch.embed_text = "Risk function for path P, summing 1-S(p) over cells, weighted by w3 in objective..."

# CCH assembly — _index_text():
cch_text = """Document context: the following excerpt is from a document titled 'mô_hình_hóa_ĐATN'. Tài liệu này trình bày về: mô hình hóa bài toán di chuyển an toàn của robot trong môi trường có bức xạ và vật cản, sử dụng FMF cải tiến.

Section context: this excerpt is from the section titled 'Hàm mục tiêu tổng quát'. Phần này trình bày về: hàm mục tiêu tổng quát gồm chiều dài, phơi nhiễm bức xạ và rủi ro va chạm với các trọng số w1, w2, w3.

Risk(P) = sum(1 - S(p)) cho các ô p trong đường đi P. Khi P1 đi qua nhiều ô có S(p) thấp..."""
```

**Và riêng cho embedding**, dùng `embed_text` thay `text`:

```python
dense_text = """Document context: the following excerpt is from a document titled 'mô_hình_hóa_ĐATN'. Tài liệu này trình bày về: mô hình hóa bài toán di chuyển an toàn của robot...

Section context: this excerpt is from the section titled 'Hàm mục tiêu tổng quát'. Phần này trình bày về: hàm mục tiêu tổng quát...

Risk function for path P, summing 1-S(p) over cells, weighted by w3 in objective..."""
#                                    ^^^^^^^^^ Reducto embed_text thay vì text gốc
```

Đây là **lợi thế kép của Reducto**: CCH header (Tier 1-4) + Reducto embed_text kết hợp cho ra dense embedding vừa có ngữ cảnh tài liệu, vừa có keyword-optimized summary từ Reducto.

---

### PHASE 4: INDEXING — BM25 + Vector

```
BM25 ← index_text (CCH text, tiếng Việt)  →  keyword search
Vector ← dense_text (CCH header + Reducto embed_text)  →  semantic search
```

BM25 dùng CCH text tiếng Việt (vì user hỏi bằng tiếng Việt), Vector dùng CCH + Reducto embed_text (vì embed_text keyword-rich hơn cho semantic search).

---

### PHASE 5: RETRIEVAL — Khi user hỏi

User hỏi: *"Vì sao công thức R(P) làm đường dài hơn nhưng ít phơi nhiễm hơn được ưu tiên?"*

```
5a. BM25:     search(query, top_k=30) → [chunk_3, chunk_7, chunk_12, ...]
5b. Dense:    embed_query(query) → vector search → [chunk_7, chunk_5, chunk_3, ...]
5c. RRF:      reciprocal_rank_fusion(bm25, dense, k=60) → [chunk_7(0.033), chunk_3(0.029), ...]
5d. Reranker: Jina API rerank(query, cch_text của 20 candidates) → sắp xếp lại theo relevance
5e. RSE:      gộp các chunk liên tiếp thành segment (ví dụ chunk_3 + chunk_4 + chunk_5 → 1 segment)
```

Reranker dùng `cch_text` (không phải raw text) — nên nó nhìn thấy CCH header và có thể hiểu ngữ cảnh tốt hơn khi xếp hạng.

---

### PHASE 6: GENERATION — Agent Graph

```
6a. Router:       → "complex" (câu hỏi suy luận đa bước)
6b. Planner:      → ["Tại sao R(P) phụ thuộc khoảng cách và suất liều?",
                      "Trọng số w2 ảnh hưởng thế nào đến ưu tiên đường đi?"]
6c. Retrieve:     → 2 sub-queries, mỗi cái lấy 5 chunks
6d. Distill+Verify: chắt lọc thông tin, kiểm chứng grounded
6e. Sufficiency:  kiểm tra đủ thông tin chưa
6f. Synthesize:   tổng hợp câu trả lời từ context + distilled notes
6g. Verify:       kiểm tra hallucination
→  "Công thức R(P) nhân khoảng cách bước đi với suất liều trung bình [3]..."
```

---

### Tổng kết: Reducto vs PyMuPDF/MinerU trong CCH flow

| Khía cạnh | PyMuPDF/MinerU | Reducto |
|---|---|---|
| **Chunking** | Backend tự chunk (markdown_chunker / chunker.py) | Reducto chunk sẵn (`page_sections`) |
| **Section** | Detect từ heading `##` | Detect từ `Section Header` block type |
| **embed_text** | Không có (dùng text gốc để embed) | Reducto generate `chunk.embed` — keyword-optimized summary |
| **CCH Tier 2** | 1 LLM call (gpt-4o-mini, ~24000 chars input) | Giống hệt — 1 LLM call |
| **CCH Tier 4** | N LLM calls song song, 1 per section | Giống hệt — N LLM calls song song |
| **Dense embed input** | CCH header + text gốc | **CCH header + Reducto embed_text** (keyword-rich hơn) |
| **BM25 input** | CCH header + text gốc | CCH header + text gốc (giống nhau) |
| **Reranker input** | `cch_text` | `cch_text` (giống nhau) |

**Lợi thế chính của Reducto trong CCH**: `embed_text` thay thế `text` cho vector embedding. Reducto tự generate bản tóm tắt keyword-rich (thường bằng tiếng Anh) cho mỗi chunk, nên khi kết hợp với CCH header tiếng Việt → vector embedding "hiểu" cả ngữ cảnh tiếng Việt lẫn keyword tiếng Anh, cải thiện semantic recall cho câu hỏi paraphrase.