# [Data] Thêm bộ benchmark đánh giá Context Retrieval cho RAG Agent

## Tóm tắt

PR này thêm bộ dataset benchmark đầu tiên của dự án, dùng để đánh giá khả năng truy xuất ngữ cảnh (context retrieval) của hệ thống RAG Agent trên tài liệu tiếng Việt thực tế.

## Thay đổi

### Tài liệu PDF nguồn (`data/benchmark/pdfs/`)

| File | Mô tả |
|---|---|
| `tt-29-2024-day-them-hoc-them-scanned.pdf` | Thông tư 29/2024/TT-BGDĐT quy định về dạy thêm, học thêm — PDF scan/image-only, 12 trang |
| `mwg-bao-cao-tai-chinh-rieng-q1-2026.pdf` | Báo cáo tài chính riêng giữa niên độ Quý 1/2026 của MWG — PDF scan nặng bảng số liệu, 26 trang |
| `SOURCES.md` | Ghi nguồn gốc và cách xác minh từng file PDF |

### Bộ câu hỏi golden (`data/benchmark/`)

| File | Mô tả |
|---|---|
| `context_retrieval_benchmark.json` | Bộ câu hỏi đầy đủ dạng JSON — 34 câu hỏi, mỗi câu có metadata chi tiết |
| `context_retrieval_benchmark.md` | Tài liệu tóm tắt cấu trúc, phân phối và danh sách câu hỏi |

## Cấu trúc bộ câu hỏi

**Tổng số:** 34 câu hỏi trên 2 PDF

| Nhóm | Số lượng |
|---|---:|
| Answerable | 30 |
| Unanswerable | 4 |
| Easy / Medium / Hard (mỗi PDF) | 5 / 5 / 5 |

### Các chiều đánh giá (dimensions)

| Dimension | Giá trị | Phân phối |
|---|---|---|
| `difficulty` | easy / medium / hard / unanswerable | 10 / 10 / 10 / 4 |
| `explicitness` | explicit / implicit | 15 / 19 |
| `hop_type` | single-hop / multi-hop | 17 / 17 |
| `query_style` | direct / metadata | 19 / 15 |
| `evidence_type` | text / table / mixed / metadata | 19 / 8 / 3 / 4 |

## Lý do chọn 2 tài liệu này

### `tt29` — Thông tư 29/2024/TT-BGDĐT (Dạy thêm/học thêm)
- **Đặc điểm kỹ thuật:** PDF scan/image-only (không có text layer) — buộc pipeline phải dùng OCR/VLM
- **Lý do chọn:** Chủ đề được quan tâm rộng rãi (phụ huynh, học sinh, giáo viên, trung tâm); ngôn ngữ pháp lý tiếng Việt thuần, nhiều điều khoản có thể kiểm tra implicit reasoning

### `mwg` — Báo cáo tài chính MWG Q1/2026
- **Đặc điểm kỹ thuật:** PDF scan 26 trang với nhiều bảng số liệu tài chính — kiểm tra khả năng OCR bảng và table reasoning
- **Lý do chọn:** Tài liệu thực tế từ doanh nghiệp niêm yết; có cấu trúc tài chính phức tạp (BCTC riêng vs. hợp nhất, lưu chuyển tiền tệ, vốn chủ sở hữu)

## Mục đích sử dụng

Bộ benchmark này được thiết kế để:
1. **Đo baseline** của pipeline RAG hiện tại (retrieval recall, precision, answer faithfulness)
2. **So sánh các kỹ thuật cải tiến** đã nghiên cứu (hierarchical indices, contextual headers, query transformation, relevant segment extraction)
3. **Kiểm tra edge case** quan trọng: câu hỏi không có đáp án (unanswerable), câu cần suy luận nhiều bước (multi-hop), truy xuất từ bảng số liệu scan

## Checklist

- [x] 2 file PDF scan tiếng Việt thực tế đã được thêm vào `data/benchmark/pdfs/`
- [x] Nguồn gốc PDF được ghi rõ trong `SOURCES.md`
- [x] 34 câu hỏi golden với đầy đủ metadata trong `context_retrieval_benchmark.json`
- [x] Phân phối đều các chiều difficulty / hop_type / evidence_type
- [x] Bao gồm 4 câu hỏi unanswerable để test hallucination guard
- [x] Tài liệu tóm tắt `context_retrieval_benchmark.md` đã được thêm
