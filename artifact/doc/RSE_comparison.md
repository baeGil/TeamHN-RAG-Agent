## RSE: SO SÁNH GIỮA dsRAG GỐC VÀ IMPLEMENT CỦA CHÚNG TA

### 1. ChunkDB — HỆ THỐNG LƯU TR KEY-VALUE MÀ BẠN HỎI

**dsRAG gốc có một thành phần riêng biệt gọi là `ChunkDB`** — một key-value store dùng để RSE lấy nội dung chunk khi cần lắp segment.

Trong dsRAG, kiến trúc chia ra **6 components riêng biệt**:

| Component | Lưu gì | Kiểu storage |
|-----------|--------|-------------|
| **VectorDB** | Embedding vectors + metadata | Faiss/Chroma/Weaviate |
| **ChunkDB** | Chunk text + chunk_header, keyed by `(doc_id, chunk_index)` | **Nested dict → pickle file** hoặc SQLite |
| Embedding | Model config | — |
| Reranker | Model config | — |
| LLM | Model config | — |
| FileSystem | PDF images | Local/S3 |

**`BasicChunkDB`** trong dsRAG lưu dưới dạng:
```python
# Nested dictionary: data[doc_id][chunk_index] = {'chunk_text': ..., 'chunk_header': ...}
# Persist to disk: pickle.dump(data, file) → <storage>/chunk_storage/<kb_id>.pkl
self.data = {
    "doc1": {
        0: {"chunk_text": "phương pháp Weighted Potential...", "chunk_header": "[Document: mô_hình_hóa_ĐATN...]"},
        1: {"chunk_text": "## 1 Problem formulation...", "chunk_header": "[Document: mô_hình_hóa_ĐATN...]"},
        2: {"chunk_text": "## 1.1 Chỉ số an toàn...", "chunk_header": "[Document: mô_hình_hóa_ĐATN...]"},
        ...
    },
    "doc2": {...}
}
```

Khi RSE cần text của segment, dsRAG gọi:
```python
def get_segment_text_from_database(self, doc_id, chunk_start, chunk_end):
    segment = f"[{self.get_chunk_header(doc_id, chunk_start)}]\n"
    for chunk_index in range(chunk_start, chunk_end):  # end non-inclusive
        chunk_text = self.get_chunk_text(doc_id, chunk_index)  # ← KEY-VALUE LOOKUP
        segment += chunk_text
    return segment.strip()
```

Mỗi lần gọi `get_chunk_text(doc_id, chunk_index)` là **1 key-value lookup**: key = `(doc_id, chunk_index)`, value = chunk text.

---

**Trong hệ thống của chúng ta**, thay vì dùng pickle file riêng, chúng ta dùng **SQLite** làm ChunkDB — chức năng tương đương nhưng quan hệ hơn (relational):

```python
# repo.py — tương đương với ChunkDB.get_chunk_text()
def get_chunks_by_doc_range(self, doc_id, start_idx, end_idx):
    rows = self.db.conn.execute(
        """SELECT c.id, c.document_id, c.chunk_index, c.text, c.page, c.section
           FROM chunks c
           WHERE c.document_id = ? AND c.chunk_index >= ? AND c.chunk_index <= ?
           ORDER BY c.chunk_index ASC""",
        (doc_id, start_idx, end_idx),
    ).fetchall()
    return [dict(r) for r in rows]
```

**So sánh trực tiếp:**

| | dsRAG `BasicChunkDB` | Hệ thống của chúng ta |
|---|---|---|
| **Kiểu storage** | Pickle file (`.pkl`) | SQLite (`rag.db`) |
| **Key** | `(doc_id, chunk_index)` | `(document_id, chunk_index)` — cột trong bảng `chunks` |
| **Value** | `{chunk_text, chunk_header}` | `{id, text, page, section, embedding, embed_text}` |
| **Lookup 1 chunk** | `data[doc_id][chunk_index]['chunk_text']` | `SELECT text FROM chunks WHERE document_id=? AND chunk_index=?` |
| **Lookup range (RSE)** | Loop `for i in range(start, end): get_chunk_text(doc_id, i)` | `SELECT ... WHERE chunk_index BETWEEN start AND end ORDER BY chunk_index` |
| **Tốc độ** | O(1) per lookup (dict), nhưng phải load toàn bộ pickle vào RAM | O(log N) per lookup (B-tree index), nhưng chỉ tải những gì cần |
| **Vấn đề RAM** | Toàn bộ chunk text của tất cả documents phải nằm trong RAM | Chỉ kết quả query nằm trong RAM |

**Kết luận: Chúng ta CÓ hệ thống key-value tương đương, nhưng dùng SQLite thay vì pickle.** Key = `(document_id, chunk_index)`, value = row trong bảng `chunks`. Index `idx_chunks_doc` trên cột `document_id` đảm bảo lookup nhanh. Khi RSE cần "bridge chunks", nó gọi `fetch_range_fn` → `get_chunks_by_doc_range()` — chính là key-value range lookup.

**Ưu điểm cách của chúng ta:** Không phải load toàn bộ data vào RAM. Với 1000+ documents, pickle file của dsRAG sẽ nặng hàng GB trong RAM, trong khi SQLite chỉ tải những chunk trong range cần thiết.

**Nhược điểm:** Lookup 1 chunk chậm hơn chút (SQL query vs dict access). Nhưng vì RSE luôn lấy range chứ không lấy 1 chunk riêng lẻ, nên 1 query `BETWEEN` hiệu quả hơn N lần dict lookup.

---

### 2. THUẬT TOÁN RSE — 3 KHÁC BIỆT QUAN TRỌNG

Bây giờ so sánh thuật toán cốt lõi:

#### Khác biệt A: CHUYỂN ĐIỂM SANG GIÁ TRỊ (SCORING FUNCTION)

**dsRAG gốc** dùng **exponential decay từ rank**:
```python
def convert_rank_to_value(rank, irrelevant_chunk_penalty, decay_rate=20):
    return np.exp(-rank / decay_rate) - irrelevant_chunk_penalty
    # rank=0  → exp(0) - 0.2 = 0.8
    # rank=1  → exp(-0.05) - 0.2 = 0.751
    # rank=5  → exp(-0.25) - 0.2 = 0.579
    # rank=20 → exp(-1) - 0.2 = 0.168
    # rank=1000 (not retrieved) → exp(-50) - 0.2 ≈ -0.2
```

**Hệ thống của chúng ta** dùng **reranker score trực tiếp**:
```python
score_arr.append(raw_score - irrelevant_penalty)  # retrieved chunk
score_arr.append(-irrelevant_penalty)             # non-retrieved chunk
```

**Tại sao khác?** dsRAG không có reranker với relevance score liên tục — nó chỉ có **rank** (thứ hạng). Nên phải chuyển rank → value qua hàm decay. Chúng ta có `BAAI/bge-reranker-v2-m3` trả về **score thực** (0→1), dùng trực tiếp chính xác hơn.

| | dsRAG | Chúng ta |
|---|---|---|
| Input | rank (0, 1, 2, ...) | reranker_score (0.85, 0.72, ...) |
| Chuyển đổi | `exp(-rank/20) - penalty` | `score - penalty` |
| Chunk không được truy xuất | rank=1000 → value≈-0.2 | value=-penalty |
| Độ tin cậy | Mất thông tin (rank 1 vs 2 khác biệt nhỏ dù score khác xa) | Giữ nguyên score gốc |

**Đánh giá:** Cách của chúng ta **tốt hơn** vì không mất thông tin qua hàm decay. Tuy nhiên, khi **không bật reranker** (`use_reranker=false`), hệ thống dùng RRF score (không phải 0→1) — cần normalize hoặc dùng fallback. Đây là một điểm cần cải thiện.

#### Khác biệt B: META-DOCUMENT VS PER-DOCUMENT

**dsRAG gốc** nối tất cả documents thành 1 "meta-document" dài**:
```python
# Tài liệu A: chunks 0-21, Tài liệu B: chunks 0-10
# Meta-document: [A0, A1, ..., A21, B0, B1, ..., B10]
# document_splits = [22, 33]  ← điểm chia giữa các tài liệu
# RSE chạy trên meta-document, đảm bảo segment KHÔNG vượt qua document_split
```

**Hệ thống của chúng ta nhóm theo document_id, chạy RSE riêng từng document:**
```python
for doc_id, chunks in doc_groups.items():
    # Chạy RSE cho tài liệu này
    score_arr = build_score_arr(chunks)
    segments = _max_sum_subarrays(score_arr, max_len, min_value)
```

**Đánh giá:** Cách của chúng ta **đơn giản hơn** và **không tạo segment vượt ranh giới tài liệu (cross-document)**. dsRAG cũng đảm bảo điều này qua `document_splits`, nhưng phức tạp hơn vì phải nối meta-document. Kết quả cuối cùng tương đương — RSE không bao giờ ghép chunk từ 2 tài liệu khác nhau.

**Tuy nhiên:** Cách của dsRAG cho phép **nhiều queries** cùng lúc, mỗi query tạo relevance values riêng trên cùng meta-document, rồi `get_best_segments` cycle qua tất cả queries. Chúng ta chỉ hỗ trợ 1 query — đây là **thiếu sót** nếu muốn multi-query retrieval (HyDE tạo nhiều query biến thể).

#### Khác biệt C: THUẬT TOÁN TÌM MAX-SUM SUBARRAY

**dsRAG gốc** — brute force O(N^2 * max_length) với overlap check:
```python
# Duyệt qua tất cả start, end trên meta-document
for start in range(len(rv)):
    for end in range(start+1, start+max_length+1):
        if overlaps_with_existing_segments(start, end): continue
        if overlaps_with_document_splits(start, end): continue
        if total_length + segment_length > overall_max_length: continue
        segment_value = sum(rv[start:end])
        if segment_value > best_value: update_best
```

**Hệ thống của chúng ta** — greedy approach O(iterations * N * max_length):
```python
# Tìm segment tốt nhất, zero-out, tìm segment tiếp theo
for _ in range(max_iterations):
    for start in range(n):
        for end in range(start, start + max_len):
            total += work[end]
            if total > best_val: update
    # Zero-out found segment → next iteration finds next-best non-overlapping
    for i in range(best_start, best_end+1):
        work[i] = 0.0
```

**Đánh giá:** Cả hai đều brute-force (không dùng Kadane O(N) thật sự vì có ràng buộc `max_len`). Cách của chúng ta **greedy (tìm tốt nhất → zero-out → lặp)**, cách dsRAG **global search với overlap constraint**. Kết quả tương đương trong thực tế vì segments hiếm khi overlap.

---

### 3. TÓM TẮT — ĐÚNG HAY KHÔNG ĐÚNG?

| Yêu cầu từ `/improvement` | dsRAG gốc | Chúng ta | Đánh giá |
|---|---|---|---|
| **No-overlap chunking** | chunk_overlap=0 | markdown_chunker: no overlap | ✓ Đúng |
| **Key-value storage (doc_id, chunk_index) → chunk text** | BasicChunkDB (pickle) | SQLite `chunks` table, key=`(document_id, chunk_index)` | ✓ Có, dùng SQLite thay pickle |
| **Bridge chunk fetching** | `get_segment_text_from_database()` | `fetch_range_fn()` → `get_chunks_by_doc_range()` | ✓ Đúng |
| **irrelevant_chunk_penalty** | 0.2 (default), exponential decay | 0.2 (default), trừ trực tiếp từ score | ⚠ Khác cơ chế scoring |
| **Max-sum subarray** | Brute-force global search | Greedy iterative zero-out | ✓ Tương đương |
| **Segment length cap** | max_length=10 | max_segment_chunks=15 | ✓ Có (giá trị khác) |
| **Overall max length** | overall_max_length=20 | overall_max_chunks=30 | ✓ Có |
| **Multiple queries** | Có (cycle qua queries) | Không (1 query duy nhất) | ✗ Thiếu |
| **Reranker score** | Không có (dùng rank → decay) | Có (dùng trực tiếp score) | ✓ Tốt hơn |

### 4. VẤN ĐỀ CẦN CẢI THIỆN

1.  **Không hỗ trợ multi-query RSE:** dsRAG nhận nhiều search queries (ví dụ HyDE tạo 3 variants), chạy RSE trên tất cả queries cùng lúc → tìm segments tổng hợp tốt nhất. Chúng ta chỉ dùng 1 query. Nếu muốn multi-query, cần mở rộng `retrieve()` để tạo `all_ranked_results` từ nhiều queries.

2.  **Scoring function khi không có reranker:** Khi `use_reranker=false`, `_final_score()` trả về `rrf_score` (có thể âm hoặc > 1). RSE trừ `irrelevant_penalty` từ score này → có thể tạo segment sai. Cần normalize RRF score về [0, 1] hoặc dùng fallback `convert_rank_to_value()` như dsRAG.

3.  **`chunk_length_adjustment`:** dsRAG có tham số `chunk_length_adjustment` — scale relevance value theo chiều dài chunk trước khi tính segment value. Chunk ngắn có ít thông tin hơn → giảm score. Chúng ta chưa implement cái này.
