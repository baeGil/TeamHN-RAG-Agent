### 1. MINERU OCR — TỪ PDF SANG MARKDOWN CẤU TRÚC

#### MinerU là gì?
MinerU là một pipeline OCR lai (hybrid OCR pipeline) chuyển đổi PDF thành markdown có cấu trúc. Nó kết hợp:
- **Layout detection** (PP-DocLayoutV2): xác định vùng tiêu đề, bảng, công thức, hình ảnh
- **VLM text recognition** (Qwen2-VL): đọc text từ hình ảnh/phần quét (scanned regions) bằng mô hình ngôn ngữ thị giác
- **OCR fallback** (PaddleOCR): nhận dạng text cổ điển khi VLM không cần thiết
- **Math formula recognition** (UniMERNet): chuyển công thức hình ảnh sang LaTeX

#### MinerU trả về gì?
MinerU được gọi qua subprocess, tạo ra 2 file chính trong thư mục đầu ra (output directory):

**File 1: `<stem>.md`** — Markdown có cấu trúc. Ví dụ thực tế từ file `mô_hình_hóa_ĐATN.pdf`:

```markdown
PHƯƠNG PHÁP WEIGHTED POTENTIAL FAST MARCHING FIREWORK CHO QUY HOẠCH 
ĐƯỜNG ĐI AN TOÀN ĐA ĐÍCH ĐẾN CỦA ROBOT DI ĐỘNG TRONG MÔI TRƯỜNG 
NHÀ MÁY HẠT NHÂN

## 1 Problem formulation

<table><tr><td>Đầu vào:</td><td>Bản đồ grid map $E$ với kích thước $H \times W$ ...</td></tr>
...nhiều hàng bảng...</table>

Đường đi có dạng:

$$
P = \{p_{i} = (cx_{i}, cy_{i})\}_{i=1}^{\tau_{P}},  \tag{1}
$$

## 1.1 Chỉ số an toàn, độ rủi ro va chạm

• Với mỗi ô $(i,j)$ trong bản đồ...
```

**Đặc điểm:**
- Heading: `## Section` (không có `#` cho H1 vì MinerU dùng `##` cho mục cấp 1)
- Bảng: HTML `<table>...</table>` (không phải markdown table)
- Công thức hiển thị (display math): `$$...$$` (LaTeX)
- Công thức nội tuyến (inline math): `$...$`
- Không có đánh dấu trang (page markers) trong markdown — trang nằm ở file JSON

**File 2: `doc_content_list.json`** — Danh sách block có kiểu với số trang. Định dạng: mảng của các trang, mỗi trang là một mảng các block:

```json
[
  [  // Page 0 (page_idx=0)
    {"type": "paragraph", "content": {"paragraph_content": [
      {"type": "text", "content": "PHƯƠNG PHÁP WEIGHTED POTENTIAL..."}
    ]}}
  ],
  [  // Page 1 (page_idx=1)
    {"type": "title", "content": {"title_content": [
      {"type": "text", "content": "1 Problem formulation"}
    ]}},
    {"type": "table", "page_idx": 1, ...},
    {"type": "equation", "page_idx": 1, ...}
  ],
  ...8 trang tổng cộng
]
```

Mỗi block có thuộc tính `type` (paragraph/title/table/equation) và `page_idx`. Đây là nguồn duy nhất cho **số trang** trong hệ thống.

#### Cách gọi MinerU
```python
# Tự detect binary theo thứ tự:
# 1. MINERU_CMD env var (nếu set)
# 2. <project_root>/.venv_parser/bin/mineru
# 3. "mineru" trên PATH

subprocess.run([cmd, "-p", pdf_path, "-o", output_dir])
# Output: <output_dir>/<stem>/auto/<stem>.md + doc_content_list_v2.json
```

---

### 2. MARKDOWN CHUNKER — TỪ MARKDOWN SANG CÁC KHỐI (BLOCKS)

#### Tại sao cần chunker riêng?
MinerU trả về markdown nguyên khối. RSE (Relevant Segment Extraction) yêu cầu **không có overlap giữa các chunk** — nếu chunk#5 và chunk#6 overlap, khi RSE lắp ghép lại segment [5..6] sẽ bị trùng lặp nội dung. Chunker thông thường (PyMuPDF/Reducto) dùng overlap → không tương thích với RSE.

#### Chiến lược chunking

Chunker hoạt động theo 5 bước:

**Bước 1: Xây dựng bản đồ trang (page map)** từ `doc_content_list_v2.json`:
```
section "1 Problem formulation" → page 2
section "1.1 Chỉ số an toàn..." → page 2
section "1.2 Độ rủi ro phòng xạ" → page 3
...
```

**Bước 2: Phát hiện ranh giới section** từ heading `##`:
```
Khi gặp "## 1.3 Bài toán tối ưu hướng tới":
  → flush prose buffer hiện tại thành block
  → cập nhật current_section = "1.3 Bài toán tối ưu hướng tới"
  → cập nhật current_page từ page_map
```

**Bước 3: Giữ nguyên các đơn vị nguyên tử (atomic units):**
- **Bảng HTML** (`<<table>...</table>`): không bao giờ cắt ngang. Nếu bảng dài 2000 chars → thành 1 chunk 2000 chars (vượt max_chars, nhưng giữ nguyên)
- **Khối công thức** (`$$...$$`): không bao giờ cắt ngang

**Bước 4: Đóng gói câu (Sentence-pack) phần văn xuôi (prose) tối đa `max_chars` (mặc định: 1000 chars), KHÔNG overlap:**
```
1. Gom tất cả prose lines trong section thành 1 đoạn
2. Tách câu bằng _split_sentences() (reuse từ chunker.py)
3. Đóng gói (Pack) từng câu vào buffer:
   - Nếu thêm câu mới vượt max_chars → flush buffer thành Block mới
   - Câu dài hơn max_chars → _smart_split_long() cắt nhỏ
```

**Bước 5: Mỗi Block mang theo (page, section, text):**
```python
Block(page=3, section="1.3 Bài toán tối ưu hướng tới",
      text="## 1.3 Bài toán tối ưu hướng tới\n\n\\[\\begin{array}{l}...")
```

#### Kết quả thực tế — hình dạng (shape) 22 chunks từ `mô_hình_hóa_ĐATN.pdf`:

```
idx=00  pg=1   sec=[-]                           165c   ← Tiêu đề chính (title block)
idx=01  pg=2   sec=[1 Problem formulation]       1427c  ← Chứa bảng HTML 600+ chars (atomic)
idx=02  pg=2   sec=[1.1 Chỉ số an toàn...]        564c
idx=03  pg=3   sec=[-]                            211c  ← Tiếp tục section trước (tràn trang)
idx=04  pg=3   sec=[• Chỉ số độ rủi ro:]          293c  ← Heading đặc biệt (bullet heading)
idx=05  pg=3   sec=[1.2 Độ rủi ro phòng xạ]       802c
idx=06  pg=3   sec=[1.3 Bài toán tối ưu...]        528c  ← Chứa công thức LaTeX (atomic)
idx=07  pg=3   sec=[2 Two phases...]               474c
idx=08  pg=4   sec=[-]                           1309c  ← Tiếp tục section 2 (tràn trang)
idx=09  pg=4   sec=[3 The Fast Marching Firework] 1288c
idx=10  pg=4   sec=[4 Phương pháp FMF...]          267c
idx=11  pg=5   sec=[-]                            306c
idx=12  pg=5   sec=[4.1 Xác định f(x)]            922c
idx=13  pg=5   sec=[4.2 Mô hình T(x)]            1027c
idx=14  pg=5   sec=[4.3 Giải thuật lan truyền]     436c
idx=15  pg=6   sec=[-]                            786c
idx=16  pg=7   sec=[-]                             65c
idx=17  pg=7   sec=[Mã giả thuật toán]              19c
idx=18  pg=7   sec=[Algorithm 1: Giai đoạn 1...]  1758c  ← Bảng mã giả (atomic)
idx=19  pg=7   sec=[// Giai đoạn giao thoa]       1457c
idx=20  pg=8   sec=[-]                            860c
idx=21  pg=8   sec=[6 Thực nghiệm]                731c  ← Chứa bảng HTML kết quả (atomic)
```

**Quan sát:**
- `idx=00` là 165 chars = tiêu đề/trang bìa (title/cover page)
- `idx=01` là 1427 chars (>max_chars 1000) vì chứa bảng HTML nguyên tử không cắt được
- `idx=18` là 1758 chars vì chứa mã giả thuật toán (pseudocode table) không cắt được
- Các chunk có `sec=[-]` là phần tiếp tục tràn trang (continuation) — section=None trong DB
- Tất cả 22 chunk xếp liên tục `idx=0..21`, tạo thành **bản đồ vị trí tuyến tính** mà RSE sử dụng

---

### 3. CCH 4-TIER (NGỮ CẢNH CHUNK ĐƯỢC BỔ SUNG BỐI CẢNH)

#### Tại sao cần CCH?
Khi embed một chunk riêng lẻ, vector chỉ biểu diễn **nội dung cục bộ (local content)**. Ví dụ chunk#06:

```
## 1.3 Bài toán tối ưu hướng tới
\[ \min_{P=...} w_1 \cdot length(P) + w_2 \cdot R(P) + w_3 \cdot Risk(P) \]
```

Vector này không biết:
- Đây là tài liệu nào? (không biết tên tài liệu)
- Tài liệu nói về cái gì? (mất ngữ cảnh tổng thể (lost overall context))
- Đây là phần nào trong tài liệu? (mất cấu trúc phân cấp (lost hierarchical structure))
- Phần này nói về cái gì? (không thể giải quyết các truy vấn diễn đạt lại (can't resolve paraphrase queries))

Kết quả: query "hàm mục tiêu tối ưu hóa đường đi robot" KHÔNG khớp (match) với chunk này vì chunk chỉ có công thức, không có từ "hàm mục tiêu" hay "robot".

#### CCH 4 tầng (4-tier CCH) — tái tạo chính xác dsRAG `auto_context.py`

**Trước khi embed**, mỗi chunk được bọc (wrapped) trong 4 tầng ngữ cảnh (context):

```
Tier 1 — Document title:    "Document context: the following excerpt is from 
                             a document titled 'mô_hình_hóa_ĐATN'."
Tier 2 — Document summary:  "Tài liệu này trình bày về phương pháp Weighted 
                             Potential Fast Marching Firework để quy hoạch 
                             đường đi an toàn đa đích cho robot di động 
                             trong môi trường nhà máy hạt nhân."
Tier 3 — Section title:     "Section context: this excerpt is from the 
                             section titled '1.3 Bài toán tối ưu hướng tới'."
Tier 4 — Section summary:   "Phần này trình bày về: Bài toán tối ưu hóa 
                             với mục tiêu tối thiểu hóa một hàm mục tiêu 
                             bao gồm chiều dài, độ tin cậy và rủi ro."
─────────
Chunk text:                 "## 1.3 Bài toán tối ưu hướng tới\n
                             \[ \min_{P=...} ... \]"
```

**Đoạn nhúng thực tế (Actual embed text)** cho chunk#06 (1050 chars tổng):
```
Document context: the following excerpt is from a document titled 'mô_hình_hóa_ĐATN'. 
Tài liệu này trình bày về phương pháp Weighted Potential Fast Marching Firework 
để quy hoạch đường đi an toàn đa đích cho robot di động trong môi trường nhà máy hạt nhân.

Section context: this excerpt is from the section titled '1.3 Bài toán tối ưu hướng tới'. 
Phần này trình bày về: Bài toán tối ưu hóa với mục tiêu tối thiểu hóa một hàm mục tiêu 
bao gồm chiều dài, độ tin cậy và rủi ro của một tập hợp các điểm trong không gian.

## 1.3 Bài toán tối ưu hướng tới
\[ \begin{array}{l} \min_{P=...} ... \]
```

**Hiệu quả:** Query "hàm mục tiêu tối ưu hóa đường đi robot" giờ sẽ khớp (match) vì:
- Tier 2 chứa "quy hoạch đường đi", "robot di động"
- Tier 4 chứa "tối ưu hóa", "hàm mục tiêu"
- Bản thân chunk chứa công thức min(…)

#### Cách tạo mỗi tầng

**Tier 2 — Document Summary** (1 lời gọi LLM (1 LLM call)):
```python
# Lấy tối đa DOC_SUMMARY_CHARS (24000) ký tự đầu của tài liệu (đầy đủ (full))
full_text = " ".join(ch.text for ch in chunks)  # Tất cả chunk nối lại
sample = full_text[:24000]  # Cắt nếu quá dài

# Gọi LLM:
prompt = "Tài liệu này là gì? Trả lời 1 câu dạng: 'Tài liệu này trình bày về: X.'"
# → "Tài liệu này trình bày về phương pháp Weighted Potential Fast Marching Firework..."
```
Chi phí: ~$0.0005/tài liệu (gpt-4o-mini, 1 lời gọi (1 call)).

**Tier 4 — Section Summaries** (1 lời gọi LLM/section, chạy song song (1 LLM call/section, parallel)):
```python
# Gom text theo section
section_texts = {"1.1 Chỉ số an toàn...": [text_chunk2, text_chunk3], ...}

# Với mỗi section (cắt tại SECTION_SUMMARY_CHARS=2000):
# Gọi LLM: "Phần '1.1 Chỉ số an toàn...' nói về gì? Trả lời 1 câu."
# → "Phần này trình bày về: Chỉ số an toàn và độ rủi ro va chạm..."

# 8 lời gọi (8 calls) chạy song song (parallel) qua ThreadPoolExecutor(max_workers=8)
```

#### Lưu trữ ở đâu?

| Dữ liệu | Bảng SQLite | Cột |
|---------|-----------|-----|
| Document summary (Tier 2) | `documents` | `summary TEXT` |
| Section summaries (Tier 4) | `section_summaries` | `(document_id, section, summary)` |
| Chunk text gốc | `chunks` | `text TEXT` |
| Chunk embed text (4-tier) | `chunks` | `embed_text TEXT` — chỉ dùng khi Reducto cung cấp embed field |
| Chunk vector | `chunks` | `embedding BLOB` — float32 vector 1536 chiều |

**Quan trọng:** Tier 1+2+3+4 không được lưu vào DB. Chúng được **tái tạo (reconstructed)** lúc indexing từ:
```python
def _index_text(title, section, text, summary=None, section_summary=None) -> str:
    parts = []
    # Tier 1+2
    if title:
        doc_ctx = f"Document context: ... '{title}'."
        if summary: doc_ctx += f" {summary}"
        parts.append(doc_ctx)
    # Tier 3+4
    if section:
        sec_ctx = f"Section context: ... '{section}'."
        if section_summary: sec_ctx += f" {section_summary}"
        parts.append(sec_ctx)
    # Tier 5
    parts.append(text)
    return "\n\n".join(parts)
```

Mỗi chunk lưu `document_id` → JOIN với `documents.summary` + `section_summaries.summary` → tái tạo 4-tier text → embed → lưu vector.

---

### 4. DỮ LIỆU LƯU TRONG SQLite — BẢNG NÀO, CỘT NÀO

#### Schema đầy đủ:

```sql
-- Bảng chính: tài liệu
documents (
    id           INTEGER PK,      -- tự tăng
    title        TEXT NOT NULL,    -- "mô_hình_hóa_ĐATN"
    source       TEXT NOT NULL,    -- tên file gốc
    source_type  TEXT,             -- "pdf" | "url" | "text"
    n_chunks     INTEGER,          -- 22
    status       TEXT,             -- "ready" | "processing" | "failed"
    error_message TEXT,
    summary      TEXT,             -- ← Tier 2: LLM document summary
    created_at   TEXT
)

-- Bảng chính: chunks
chunks (
    id           INTEGER PK,      -- == turbovec uint64 id (1,2,3,...,22)
    document_id  INTEGER FK,       -- → documents.id
    chunk_index  INTEGER NOT NULL, -- ← RSE cần cái này! Vị trí thứ tự (0,1,2,...,21)
    text         TEXT NOT NULL,    -- raw chunk text (không có CCH headers)
    page         INTEGER,          -- số trang (1-based)
    section      TEXT,             -- tên section (hoặc NULL nếu tràn trang)
    embedding    BLOB,             -- float32 vector 1536 chiều
    embed_text   TEXT              -- Reducto embed field (nếu có)
)
-- Index: idx_chunks_doc ON chunks(document_id)

-- Bảng: section summaries (Tier 4)
section_summaries (
    id           INTEGER PK,
    document_id  INTEGER FK,
    section      TEXT NOT NULL,    -- "1.3 Bài toán tối ưu hướng tới"
    summary      TEXT NOT NULL     -- "Phần này trình bày về: Bài toán tối ưu..."
)
-- Unique index: (document_id, section) — 1 summary/section/doc

-- Bảng: conversation summaries (tóm tắt hội thoại)
conversation_summaries (
    id                INTEGER PK,
    session_id        TEXT FK,
    summary           TEXT,
    summarized_up_to  INTEGER,    -- message id cuối cùng đã tóm tắt
    created_at, updated_at
)
```

#### Các file persisted khác (ngoài SQLite):

| File | Nội dung | Kích thước |
|------|----------|------------|
| `rag.db` | SQLite database (tất cả bảng trên) | 244 KB |
| `bm25.pkl` | BM25 index đã được pickle (tokenized corpus) | 47.6 KB |
| `vector.tvim` | TurboVec index (binary, uint64 id → vector) | 28.8 KB |
| `emb_cache.db` | Cache embedding (text hash → vector) | 348 KB |
| `index_meta.json` | `{"embed_dim": 1536, "chunk_ids": [1,2,...,22], "index_version": 6}` | 133 B |

**Quan trọng về `chunk_ids` trong meta.json**: `[1, 2, 3, ..., 22]` — đây là DB row id. Còn `chunk_index` trong bảng chunks là **vị trí thứ tự trong tài liệu** (0, 1, 2, ..., 21). Hai cái này khác nhau:
- `id=7, chunk_index=6` → chunk DB id=7, nhưng là chunk thứ 6 trong tài liệu

#### BM25 Index (`bm25.pkl`)
- Không phải key-value store. Là **corpus đã được tokenize**:
  - `self._corpus`: danh sách tokenized text cho mỗi chunk
  - `self._chunk_ids`: danh sách [1, 2, ..., 22] ánh xạ vị trí → chunk DB id
  - Sử dụng `rank_bm25.BM25Okapi` để tính điểm (score)
- Khi search: tokenize query → `bm25.get_scores(tokenized_query)` → trả về [(chunk_id, score)]

#### Vector Index (`vector.tvim`)
- TurboVec: custom ANN index (Approximate Nearest Neighbor)
- Lưu trữ ma trận numpy: vectors[chunk_id] = float32[1536]
- Search: `query_vector @ vectors.T` → cos similarity → trả về [(chunk_id, score)]

---

### 5. RSE — TRÍCH XUẤT ĐOẠN PHÙ HỢP (RELEVANT SEGMENT EXTRACTION)

#### Vấn đề RSE giải quyết
Top-k cố định: lấy 5 chunk điểm cao nhất. Ví dụ:
```
chunk#05  score=0.85  (1.2 Độ rủi ro phòng xạ)
chunk#06  score=0.72  (1.3 Bài toán tối ưu)
chunk#07  score=0.65  (2 Two phases)
chunk#09  score=0.55  (3 Fast Marching Firework)
chunk#18  score=0.78  (Algorithm 1)
```
→ Kết quả: 5 chunk rời rạc, chunk#08 bị bỏ (khoảng hở giữa 7 và 9), chunk#18 cách xa 09-17. LLM nhận context bị đứt đoạn (fragmented context).

#### RSE giải quyết thế nào
RSE lắp ghép **các đoạn liên tục (contiguous segments)** thay vì các chunk rời rạc:

**Thuật toán từng bước (Algorithm step by step):**

**Bước 1: Nhóm (Group) chunks theo document_id:**
```
Doc 1: chunks #5, #6, #7, #9, #18
```

**Bước 2: Xây dựng mảng điểm (Build score array) qua phạm vi (range) [min_idx..max_idx]:**
```
Mỗi vị trí trong [5..18] (14 vị trí):
  - chunk nào ĐƯỢC truy xuất (retrieved): score = rerank_score - irrelevant_penalty
  - chunk nào KHÔNG được truy xuất:        score = -irrelevant_penalty

idx=05: 0.85 - 0.20 = +0.65  ← ANCHOR (được truy xuất (retrieved))
idx=06: 0.72 - 0.20 = +0.52  ← ANCHOR
idx=07: 0.65 - 0.20 = +0.45  ← ANCHOR
idx=08:             -0.20     ← BRIDGE (khoảng hở (gap))
idx=09: 0.55 - 0.20 = +0.35  ← ANCHOR
idx=10:             -0.20     ← GAP
idx=11:             -0.20     ← GAP
idx=12:             -0.20     ← GAP
idx=13:             -0.20     ← GAP
idx=14:             -0.20     ← GAP
idx=15:             -0.20     ← GAP
idx=16:             -0.20     ← GAP
idx=17:             -0.20     ← GAP
idx=18: 0.78 - 0.20 = +0.58  ← ANCHOR
```

**Bước 3: Tìm mảng con có tổng lớn nhất (Max-sum subarray)** (Kadane variant):
```
score_arr = [+0.65, +0.52, +0.45, -0.20, +0.35, -0.20, -0.20, ..., +0.58]

Thử các cửa sổ trượt (sliding windows) độ dài 1..max_segment_chunks(15):
  [5..9]: 0.65 + 0.52 + 0.45 + (-0.20) + 0.35 = 1.77 ← TỐT NHẤT (BEST)
  [5..18]: tổng âm (negative sum) vì quá nhiều GAP
  [18..18]: 0.58 ← TỐT THỨ HAI (SECOND BEST)
```

Kết quả: 2 segments:
1. **[5..9]**: giá trị=1.77 (5 chunks, bao gồm 1 bridge chunk #8)
2. **[18..18]**: giá trị=0.58 (1 chunk)

**Bước 4: Lấy các chunk cầu nối (Fetch bridge chunks) từ DB:**
```python
# fetch_range_fn(doc_id=1, start_idx=5, end_idx=9)
# SQL: SELECT ... FROM chunks WHERE document_id=1 AND chunk_index BETWEEN 5 AND 9
# Trả về: chunk#5, #6, #7, #8 (bridge!), #9
```

**Bước 5: Lắp ráp (Assemble) đoạn (segment):**
```python
text = "\n\n".join(chunk.text for chunk in [5, 6, 7, 8, 9])
# = toàn bộ nội dung từ "Độ rủi ro phòng xạ" → "Fast Marching Firework"
# liên tục, không đứt đoạn!
```

#### Tham số RSE

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `USE_RSE` | false | Bật/tắt RSE |
| `RSE_IRRELEVANT_PENALTY` | 0.2 | Phạt (penalty) cho mỗi chunk không được truy xuất (not retrieved). Cao = RSE thích đoạn (segments) ngắn, tập trung. Thấp = RSE bao dung hơn, lắp đoạn dài hơn |
| `RSE_MAX_SEGMENT_CHUNKS` | 15 | Tối đa chunk/đoạn (chunk/segment). Ngăn 1 đoạn chiếm quá nhiều token |
| `RSE_OVERALL_MAX_CHUNKS` | 30 | Tổng tối đa chunk trên tất cả đoạn (total max chunks across all segments). Soft cap |

#### Bug quan trọng đã sửa
**Trước:** `retrieve()` cắt `top_k` TRƯỚC khi đưa vào RSE → RSE chỉ thấy 5 chunk, không thể tìm đoạn (segments).
**Sau:** `retrieve()` đưa **TẤT CẢ** reranked candidates vào RSE:
```python
all_scored = sorted(candidates, key=_final_score, reverse=True)  # KHÔNG cắt
if s.use_rse:
    out = self._apply_rse(pre_rse)  # Tất cả candidates → RSE
else:
    out = pre_rse[:top_k]           # Không RSE → cắt top_k như cũ
```

#### Kết quả benchmark thực tế

| Parser | RSE | Keyword Hit | Avg Context Chars |
|--------|-----|-------------|-------------------|
| PyMuPDF | ON  | 61% | 1,821 |
| PyMuPDF | OFF | 86% | 9,279 |
| MinerU  | ON  | 59% | 1,231 |
| MinerU  | OFF | 86% | 9,432 |

RSE giảm keyword hit (vì đoạn (segment) đặc hơn (denser), ít token thừa (fewer wasted tokens)) nhưng giảm context xuống 1/5 — LLM nhận **đoạn đặc (dense segments)** thay vì **các chunk phân tán (scattered chunks)**.

---

### 6. TOÀN BỘ PIPELINE TRUY XUẤT — END-TO-END

```
Query: "hàm mục tiêu tối ưu hóa đường đi robot"
  │
  ├── [1] BM25 Search (keyword matching)
  │     bm25.search(query, top_k=30)
  │     → [(chunk_id=7, score=2.1), (chunk_id=5, score=1.8), ...]  30 hits
  │
  ├── [2] Dense Search (semantic matching)
  │     query_vector = embed("hàm mục tiêu tối ưu hóa đường đi robot")
  │     vector.search(query_vector, top_k=30)
  │     → [(chunk_id=6, score=0.89), (chunk_id=5, score=0.85), ...]  30 hits
  │
  ├── [3] RRF Merge (Reciprocal Rank Fusion)
  │     reciprocal_rank_fusion(bm25_hits, dense_hits, k=60)
  │     → [(chunk_id, rrf_score), ...]  merged & deduplicated
  │
  ├── [4] Rerank (cross-encoder re-scoring)
  │     rerank_top_n=20 candidates → BAAI/bge-reranker-v2-m3
  │     → [(chunk_id=5, score=0.85), (chunk_id=6, score=0.72), ...]
  │
  ├── [5] RSE (if USE_RSE=true)
  │     Feed ALL scored candidates (not just top_k!)
  │     → Group by doc → Build score array → Max-sum subarray
  │     → Fetch bridge chunks from DB → Assemble segments
  │     → Return [RseSegment(doc1, idx=[5..9], text="...contiguous..."),
  │               RseSegment(doc1, idx=[18..18], text="...")]
  │
  └── [6] Return to LLM
        Each result is a RetrievedChunk:
          - text = assembled segment text (RSE) or raw chunk text (top_k)
          - is_segment = True (RSE) or False (top_k)
          - segment_chunk_ids = [5,6,7,8,9] (RSE) or [] (top_k)
```

#### Index Version — Tại sao cần?

```python
_INDEX_VERSION = 6  # In index_meta.json
```

Khi format `_index_text()` thay đổi (ví dụ: thêm CCH Tier 4), các vector đã lưu trở nên **lỗi thời (stale)** — chúng được embed từ text format cũ. Nếu `index_version` trong meta.json < `_INDEX_VERSION` trong code → **tự động tái xây dựng chỉ mục (auto-rebuild indexes)**: lấy (re-read) tất cả chunks + section_summaries từ DB, tái tạo (re-generate) `_index_text()`, tái nhúng (re-embed) tất cả, lưu lại vector mới.

Điều này đảm bảo không bao giờ có vector lỗi thời âm thầm làm giảm chất lượng truy xuất (retrieval quality) — một vấn đề phổ biến (common pitfall) trong các hệ thống RAG production.

---

### TÓM TẮT KIẾN TRÚC DỮ LIỆU (DATA ARCHITECTURE)

```
┌─────────────────────────────────────────────────────┐
│                    SQLite (rag.db)                   │
│                                                     │
│  documents          chunks              section_    │
│  ┌────────────┐    ┌─────────────────┐   summaries  │
│  │ id=1       │◄──│ document_id=1   │  ┌─────────┐│
│  │ title      │    │ chunk_index=6   │  │doc_id=1 ││
│  │ summary ───┼──┐ │ text (raw)      │  │section  ││
│  │ n_chunks   │  │ │ page, section   │  │summary ─┼┤
│  └────────────┘  │ │ embedding BLOB  │  └─────────┘│
│                  │ │ embed_text      │              │
│                  │ └─────────────────┘              │
│                  │                                   │
│                  └──→ _index_text(title, section,    │
│                       text, summary, section_summary)│
│                       = 4-tier CCH embed text         │
└─────────────────────────────────────────────────────┘
          │                    │                │
          ▼                    ▼                ▼
   ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐
   │  BM25 Index │  │  Vector Index    │  │  Embed Cache  │
   │  (bm25.pkl) │  │  (vector.tvim)  │  │ (emb_cache.db)│
   │             │  │                  │  │              │
   │ tokenized   │  │ numpy matrix     │  │ text hash →  │
   │ _index_text │  │ float32[1536]    │  │ vector cache │
   │ per chunk   │  │ per chunk_id     │  │              │
   └─────────────┘  └──────────────────┘  └──────────────┘
```

Cả BM25 và Vector đều được lập chỉ mục (indexed) trên **cùng một `_index_text()` (văn bản 4 tầng (4-tier text))** — đảm bảo keyword matching và semantic matching nhìn thấy ngữ cảnh (context) giống nhau.
