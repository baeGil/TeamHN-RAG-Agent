# Báo cáo thu thập dữ liệu và xây dựng golden dataset

**Branch:** `data_longnv`
**Phạm vi:** `data/50_files/` — 50 tài liệu PDF tiếng Việt, 250 câu hỏi kiểm chứng RAG

---

## 1. Mục tiêu

Xây dựng bộ benchmark tiếng Việt để đánh giá hệ thống RAG theo nhiều chiều: truy hồi chính xác từ văn bản và bảng, tổng hợp nhiều đoạn, tính toán từ số liệu, nhận diện câu hỏi mơ hồ, và từ chối khi tài liệu không có đáp án.

---

## 2. Tiêu chí chọn tài liệu

Mỗi tài liệu được chọn phải đáp ứng đủ các điều kiện sau:

- **PDF tiếng Việt thực tế** — không dịch máy, không tài liệu học thuật trừu tượng.
- **Nguồn rõ ràng** — có URL gốc có thể truy cập lại để xác minh.
- **Trích xuất được text** — ưu tiên PDF text-based; chấp nhận PDF scan nếu nguồn là văn bản pháp luật chính thức.
- **Đủ nội dung để tạo câu hỏi đa dạng** — ít nhất phải sinh được 2 câu easy, 2 câu medium, 1 câu hard theo schema định nghĩa.
- **Không trùng nội dung với tài liệu đã có** trong bộ dữ liệu chung của nhóm.
- **Đa dạng chủ đề** — trải đều 8 chủ đề theo quota định trước.

---

## 3. Nguồn thu thập

| Tổ chức / Nguồn | Số tài liệu | Loại tài liệu |
|---|---:|---|
| UNDP Việt Nam | 24 | Báo cáo chính sách, khảo sát, hướng dẫn, đề xuất dự án |
| UNICEF Việt Nam | 6 | Báo cáo nghiên cứu, tóm lược SDG, tổng kết xã hội |
| WHO IRIS | 3 | Báo cáo nhân lực y tế, tài liệu truyền thông sức khỏe |
| World Bank | 3 | Báo cáo kinh tế vĩ mô, tài liệu chính sách thương mại |
| ADB Legal | 3 | Văn bản pháp luật (Hiến pháp, Luật tố tụng dân sự, Luật phòng chống thiên tai) |
| VinFast | 4 | Brochure sản phẩm xe điện (VF 3, VF 5, VF 9, VF e34) |
| MobiFone / Công ty khác | 2 | Bảng giá dịch vụ, chiến lược quốc gia AI |
| Công báo Chính phủ | 1 | Thông tư quy định về dạy thêm học thêm (PDF scan) |

Tất cả PDF được tải trực tiếp từ URL gốc. URL được ghi lại đầy đủ trong cột `nguon_url` của `master_documents.csv`.

---

## 4. Phân bố chủ đề

Bộ dữ liệu trải đều 8 chủ đề nhằm kiểm tra RAG trên nhiều lĩnh vực ngôn ngữ khác nhau:

| Chủ đề | Số tài liệu | Ví dụ tài liệu tiêu biểu |
|---|---:|---|
| Môi trường - năng lượng - giao thông | 11 | Điện gió ngoài khơi, phục hồi sau bão Yagi, kế hoạch thích ứng khí hậu |
| Pháp luật - chính sách | 9 | Hiến pháp 2013, PAPI 2023-2024, Luật phòng chống thiên tai |
| Lịch sử - văn hóa - xã hội | 7 | Người khuyết tật, già hóa dân số, di cư nội địa, 50 năm vì trẻ em |
| Sản phẩm - dịch vụ - bảng giá | 5 | Brochure VF 3/5/9/e34, bảng giá MobiCloud |
| Khoa học - công nghệ - AI | 5 | AI trong cơ quan nhà nước, đánh giá DVC, chiến lược AI 2030 |
| Y tế - sức khỏe cộng đồng | 5 | Nhân lực y tế, viêm gan vi rút, tác hại rượu bia, sức khỏe trẻ em |
| Tài chính - kinh tế - doanh nghiệp | 5 | FTA thế hệ mới, kinh tế vĩ mô 2025, SIB, kinh doanh có trách nhiệm |
| Giáo dục - đào tạo | 3 | Trẻ em ngoài nhà trường, giáo dục trẻ khuyết tật, SDGCW giáo dục |

---

## 5. Đặc điểm bộ tài liệu

- **Độ dài tài liệu**: từ 1 trang (hướng dẫn sử dụng thiết bị y tế) đến 258 trang (báo cáo phục hồi sau bão Yagi, xã hội già hóa World Bank).
- **Định dạng nội dung đa dạng**: bao gồm báo cáo tự sự, bảng số liệu, văn bản pháp luật có điều khoản, brochure sản phẩm có thông số kỹ thuật, tài liệu khảo sát có biểu đồ và phụ lục.
- **Gần như toàn bộ là text-based PDF** (49/50); duy nhất DOC002 là PDF scan từ Công báo Chính phủ.
- **Thời gian xuất bản**: từ 2013 (văn bản pháp luật ADB) đến 2025–2026 (báo cáo UNDP mới nhất), đảm bảo bộ dữ liệu không lỗi thời.

---

## 6. Xây dựng golden dataset

### 6.1 Schema mỗi câu hỏi

Mỗi câu hỏi gồm 11 trường:

| Trường | Nội dung |
|---|---|
| `doc_id` | Mã tài liệu (DOC001–DOC050) |
| `question_id` | Mã câu hỏi (DOC001_Q01 …) |
| `question` | Câu hỏi tiếng Việt tự nhiên |
| `expected_answer` | Câu trả lời kỳ vọng, chỉ dựa vào tài liệu |
| `evidence` | Trang / mục / bảng làm căn cứ |
| `difficulty` | `easy` / `medium` / `hard` |
| `difficulty_explanation` | Giải thích theo 4 chiều: Retrieval, Reasoning, Semantic, Document |
| `question_types` | Một hoặc nhiều nhãn có giải thích ngắn |
| `human_steps` | Các bước tra cứu mà người thật sẽ làm |
| `answerability` | `answerable` / `unanswerable` / `clarification_needed` |
| `notes` | Ghi chú QC nếu có |

### 6.2 Phân bố độ khó

Mỗi tài liệu có đúng 5 câu, phân bố 2–2–1:

| Difficulty | Câu/tài liệu | Tổng 50 tài liệu |
|---|---:|---:|
| `easy` | 2 | 100 |
| `medium` | 2 | 100 |
| `hard` | 1 | 50 |
| **Tổng** | **5** | **250** |

### 6.3 Phân bố answerability

| Answerability | Số câu |
|---|---:|
| `answerable` | 234 |
| `unanswerable` | 9 |
| `clarification_needed` | 7 |

Câu `unanswerable` kiểm tra khả năng chống hallucination. Câu `clarification_needed` kiểm tra hành vi hỏi lại khi câu hỏi người dùng thiếu thông tin bối cảnh.

### 6.4 Phân bố question_types

Mỗi câu có thể gắn nhiều nhãn. Bảng dưới đây đếm số lần xuất hiện mỗi nhãn trên toàn bộ 250 câu:

| Question type | Số lần xuất hiện | Năng lực RAG được kiểm tra |
|---|---:|---|
| `single_hop` | 174 | Truy hồi một đoạn/trang duy nhất |
| `specific` | 166 | Câu hỏi hẹp, nhắm thông tin cụ thể |
| `factoid` | 76 | Trả lời ngắn: số, tên, ngày, giá trị |
| `list` | 73 | Liệt kê nhiều mục cùng loại |
| `abstract` | 72 | Tổng hợp, diễn giải, rút ý chính |
| `multi_hop` | 52 | Kết hợp nhiều đoạn/trang/bảng |
| `comparison` | 37 | Đối chiếu hai hoặc nhiều đối tượng |
| `definition` | 31 | Hỏi định nghĩa hoặc khái niệm |
| `condition` | 29 | Hiểu điều kiện, ngoại lệ, phạm vi áp dụng |
| `table_lookup` | 18 | Đọc đúng dòng/cột/ô trong bảng |
| `unanswerable` | 10 | Từ chối khi tài liệu không đủ thông tin |
| `calculation` | 10 | Tính toán từ số liệu trong tài liệu |
| `clarification_needed` | 8 | Nhận diện câu hỏi mơ hồ, hỏi lại |

---

## 7. Cấu trúc thư mục

```
data/50_files/
├── raw_pdf/              # 50 file PDF gốc (DOC001–DOC050)
├── golden_dataset/       # 50 file DOC*_qa.md (250 câu hỏi)
├── metadata/
│   └── master_documents.csv   # Metadata đầy đủ 50 tài liệu
└── QUESTION_GUIDELINES.md     # Hướng dẫn tạo câu hỏi và schema
```

---

## 8. Kiểm tra chất lượng

- Toàn bộ 50 PDF mở được và trích xuất text được.
- 50 file `*_qa.md` khớp 1-1 với 50 file PDF theo `doc_id`.
- Không có câu hỏi nào thiếu trường bắt buộc.
- URL nguồn ghi trong metadata có thể truy cập lại để xác minh tài liệu gốc.
- Phân bố 2 easy / 2 medium / 1 hard được giữ đều trên toàn bộ 50 tài liệu.
