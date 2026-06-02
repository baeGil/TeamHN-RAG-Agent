# PR Docs: Migration từ PyMuPDF sang Reducto

## Báo cáo số liệu cho pipeline hỏi đáp tài liệu toán học tiếng Việt

Mục tiêu của PR này là thay parser PDF cục bộ bằng Reducto API để cải thiện chất lượng trích xuất công thức, giữ nguyên cấu trúc section, và tách rõ 2 đường index:

- `text` cho BM25
- `embed_text` cho dense embedding

---

## 1. Tóm tắt kết quả

| Metric | PyMuPDF | Reducto agentic | Ghi chú |
|---|---:|---:|---|
| Công thức đúng | 2/5 | 5/5 | LaTeX được giữ nguyên tốt hơn |
| Số block từ parser | 18 | 22 | Reducto giữ cấu trúc section tốt hơn |
| Số chunk sau ingest | 20 | 22 | Bỏ `chunk_blocks()` thủ công |
| Trung bình chars/chunk | 541 | 727 / 655 | `text` / `embed_text` |
| Chunks có `embed_text` | 0/20 | 22/22 | 100% coverage |
| Chi phí xử lý | local + VLM tokens | ~2.5 credits/page | Mode `agentic` |
| Tổng chi phí 8 trang | N/A | ~$0.40 | 20 credits ≈ 8 trang |
| Cần VLM fallback | Có | Không | Reducto xử lý sẵn |

---

## 2. Phạm vi thay đổi

PR này thay đổi 4 phần chính:

1. Parser PDF
2. Chunking / ingestion
3. Indexing cho BM25 + dense
4. Schema và config

| Hạng mục | Trước | Sau |
|---|---|---|
| Parser | PyMuPDF + VLM fallback | Reducto API (`agentic`) |
| Chunking | `chunk_blocks()` | `page_sections` từ Reducto |
| Dense embedding | Dùng chung `text` | Dùng `embed_text` |
| BM25 | Dùng `text` | Vẫn dùng `text` |
| DB schema | Không có `embed_text` | Thêm cột `embed_text` |
| Fallback | Scan/empty pages | `REDUCTO_PARSE=off` hoặc API fail |

---

## 3. Pipeline trước và sau

### 3.1 Trước: PyMuPDF

```text
PDF
→ PyMuPDF extract raw text + font metadata
→ Math-aware reconstruction
→ VLM fallback cho scan/empty pages
→ Vietnamese normalization
→ chunk_blocks(max_chars=1000, overlap=200)
→ Dense embedding trên text
→ BM25 trên text
→ Hybrid retrieval
```

### 3.2 Sau: Reducto agentic

```text
PDF
→ Reducto API (agentic)
→ VLM review text + tables
→ page_sections chunking
→ filter Header/Footer/Page Number
→ normalization cho text + embed_text
→ Dense embedding trên embed_text
→ BM25 trên text
→ Hybrid retrieval
```

### 3.3 Điểm khác biệt chính

- Reducto trả về `text` đầy đủ và `embed_text` đã tối ưu cho embedding.
- BM25 tiếp tục dùng `text` để giữ keyword match.
- Dense index chuyển sang `embed_text` để giảm noise và tăng chất lượng semantic search.
- Không còn cần bước chunk thủ công sau khi parse.

---

## 4. Kết quả đánh giá

### 4.1 Độ chính xác công thức

| Công thức | PyMuPDF | Reducto agentic |
|---|---|---|
| `S(c)` | Mất cấu trúc phân số | LaTeX đúng |
| `R(P)` | Sai ký tự, lỗi hiển thị | Đúng `R(P)` |
| `risk(P)` | Thiếu ký hiệu chuẩn | Đúng `\operatorname{risk}(P)` |
| `D_{ij}` | Không tìm thấy | Có LaTeX subscript |
| `f(x)` | Có nhưng thiếu chuẩn hóa | LaTeX rõ ràng hơn |

### 4.2 Ví dụ công thức

**S(c)**

| Trạng thái | Nội dung |
|---|---|
| Trước | `S(c) = C₁ + (1-C₁). (4)` |
| Sau | `S(c)=C₁·\frac{24-N_{obs}(c)}{24}+(1-C₁)·\frac{d_{min}(c,\mathcal{O}_c)}{3}` |

**D_{ij}**

| Trạng thái | Nội dung |
|---|---|
| Trước | Không trích xuất đúng |
| Sau | `D_{i,j}=\min_{P_{ij}} w_1·length(P_{ij}) + w_2·R(P_{ij}) + w_3·Risk(P_{ij})` |

### 4.3 Retrieval benchmark

| Query | Top-1 Section | BM25 Score | Dense Score |
|---|---|---:|---:|
| `chỉ số an toàn S(c)` | `1.1 Chỉ số an toàn` | 9.07 | 0.605 |
| `độ rủi ro R(P)` | `1.1 Chỉ số an toàn` | 13.08 | 0.481 |
| `hàm chi phí f(x)` | `4.1 Xác định f(x)` | 7.30 | 0.556 |
| `ma trận Dij` | `2 Two phases` | 11.50 | 0.489 |
| `2 phases of optimization` | `2 Two phases` | 12.96 | 0.536 |

Kết quả thử nghiệm cho thấy truy vấn keyword và semantic đều trả về section đúng trong bộ tài liệu mục tiêu.

---

## 5. Thống kê ingest

| Metric | Giá trị |
|---|---:|
| Tổng số chunks | 22 |
| Tổng chars (`text`) | 15,996 |
| Tổng chars (`embed_text`) | 14,414 |
| Trung bình chars/chunk (`text`) | 727 |
| Trung bình chars/chunk (`embed_text`) | 655 |
| Embedding dimension | 1,536 |
| Chunks có `embed_text` | 22/22 |
| Chunks per page (avg) | 2.75 |

### Phân bố chunks theo trang

| Trang | Số chunks | Nội dung chính |
|---|---:|---|
| 1 | 1 | Trang bìa |
| 2 | 2 | Problem formulation, `S(c)` |
| 3 | 5 | `S(c)` cont., `risk(P)`, `R(P)`, bài toán tối ưu |
| 4 | 3 | `2 phases` cont., FMF method |
| 5 | 4 | `f(x)`, `T(x)`, giải thuật lan truyền |
| 6 | 1 | FMF cont. |
| 7 | 4 | Algorithm 1, giao thoa |
| 8 | 2 | Thực nghiệm |

---

## 6. Chi phí và hiệu năng

| Metric | PyMuPDF + VLM | Reducto agentic |
|---|---|---|
| Chi phí | `0` local + VLM tokens | ~2.5 credits/page |
| 8 trang | phụ thuộc fallback | ~$0.40 |
| Tốc độ | ~1-2s/page | ~2.5s/page |
| Tổng thời gian 8 trang | ~8-16s | ~20s |
| Công thức toán | Hay lỗi phân số | LaTeX chính xác hơn |
| Văn bản tiếng Việt | Cần normalize nhiều | Ổn định hơn |
| Chunks có `embed_text` | 0/20 | 22/22 |
| Fallback VLM | Có | Không cần ở luồng chính |

**Ghi chú:** mode `default` rẻ hơn nhưng không phù hợp cho tài liệu toán học vì chất lượng công thức không ổn định. Mode `agentic` là lựa chọn dùng để review PR này.

---

## 7. Thay đổi kỹ thuật chính

| File | Thay đổi |
|---|---|
| `app/ingestion/block.py` | Dataclass `Block` có `embed_text` |
| `app/ingestion/reducto_parser.py` | Reducto API client và mapping kết quả |
| `app/ingestion/loaders.py` | Điều hướng Reducto / PyMuPDF |
| `app/ingestion/chunker.py` | Chunk thêm `embed_text` |
| `app/ingestion/pdf_extract.py` | Import `Block` từ module chung |
| `app/ingestion/pdf_vlm.py` | Import `Block` từ module chung |
| `app/indexing/store.py` | Dual-channel indexing |
| `app/db/database.py` | Thêm cột `embed_text` + migration |
| `app/db/repo.py` | Persist và load `embed_text` |
| `app/config.py` | Thêm `REDUCTO_*` vars |
| `.env` | `REDUCTO_PARSE=agentic` và API key |
| `_INDEX_VERSION` | `3 → 4` để rebuild index |

### Config mới

```env
REDUCTO_PARSE=agentic
REDUCTO_API_KEY=sk-...
REDUCTO_CHUNK_MODE=page_sections
REDUCTO_CHUNK_SIZE=1200
REDUCTO_FILTER_BLOCKS=Header,Footer,Page Number
REDUCTO_TABLE_FORMAT=dynamic
```

### Data model / schema

```python
@dataclass
class Chunk:
    text: str
    page: Optional[int]
    section: Optional[str]
    embed_text: Optional[str] = None
```

```sql
ALTER TABLE chunks ADD COLUMN embed_text TEXT;
```

---

## 8. Kết luận

- Reducto agentic cải thiện rõ chất lượng trích xuất công thức: `2/5` lên `5/5`.
- `embed_text` giúp dense embedding sạch hơn trong khi BM25 vẫn giữ được keyword match trên `text`.
- Dữ liệu ingest ổn định hơn với `page_sections`, không cần chunk thủ công sau parse.
- Chi phí tăng so với local parser, nhưng đổi lại là chất lượng và độ ổn định phù hợp cho tài liệu toán.
- Fallback vẫn có thể giữ ở tầng parser nếu `REDUCTO_PARSE=off` hoặc API lỗi.