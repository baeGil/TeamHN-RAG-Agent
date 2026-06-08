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
| UNDP / UN Việt Nam | 7 | Báo cáo tóm tắt, kế hoạch môi trường - xã hội, khí hậu và phát triển con người |
| UNICEF / UNFPA / WHO | 10 | Tờ tin, báo cáo tóm tắt, tài liệu sức khỏe và xã hội |
| World Bank / JICA / ADB | 8 | Policy note, hồ sơ đầu tư, giao thông, tăng trưởng xanh và trái phiếu bền vững |
| VinFast / MobiFone / Manulife | 6 | Brochure, bảng giá, tài liệu sản phẩm - dịch vụ |
| PwC / FPT / CSIRO-Aus4Innovation | 6 | Legal brief, công nghệ, blockchain, an ninh mạng và AI |
| Cổng Công báo / ADB Legal / Bộ GDĐT / ILO và nguồn khác | 13 | Văn bản chính sách, giáo dục, lao động và an sinh |

Tất cả PDF được tải trực tiếp từ URL gốc. URL được ghi lại đầy đủ trong cột `nguon_url` của `master_documents.csv`.

---

## 4. Phân bố chủ đề

Bộ dữ liệu trải đều 8 chủ đề nhằm kiểm tra RAG trên nhiều lĩnh vực ngôn ngữ khác nhau:

| Chủ đề | Số tài liệu | Ví dụ tài liệu tiêu biểu |
|---|---:|---|
| Môi trường - năng lượng - giao thông | 9 | NDC, VITRANSS 3, giảm nhựa dùng một lần, SREX, trái phiếu xanh |
| Pháp luật - chính sách | 8 | Thông tư 29/2024, Hiến pháp 2013, Luật phòng chống thiên tai, Bộ quy tắc bảo vệ trẻ em trên mạng |
| Lịch sử - văn hóa - xã hội | 7 | Lao động trẻ em, mức sinh, di cư nội địa, HDI, bảo hiểm xã hội |
| Sản phẩm - dịch vụ - bảng giá | 6 | Brochure VF 3/5/9/e34, MobiCloud, Manulife Sống Khỏe Mỗi Ngày |
| Khoa học - công nghệ - AI | 6 | Chuyển đổi số quốc gia, Decree 53, dữ liệu cá nhân, blockchain, chiến lược AI 2030 |
| Y tế - sức khỏe cộng đồng | 6 | Sức khỏe bà mẹ COVID-19, chăm sóc thai sản, tác hại rượu bia, SDGCW sức khỏe |
| Tài chính - kinh tế - doanh nghiệp | 5 | Enterprise Survey, chính sách đầu tư, kinh doanh có trách nhiệm, kinh tế vĩ mô 2025 |
| Giáo dục - đào tạo | 3 | SDGCW giáo dục, chương trình giáo dục phổ thông tổng thể, tài chính giáo dục đại học |

---

## 5. Đặc điểm bộ tài liệu

- **Độ dài tài liệu**: từ 1 trang đến 48 trang; toàn bộ 50 PDF đều dưới 50 trang theo yêu cầu lọc lại.
- **Định dạng nội dung đa dạng**: bao gồm báo cáo tự sự, bảng số liệu, văn bản pháp luật có điều khoản, brochure sản phẩm có thông số kỹ thuật, tài liệu khảo sát có biểu đồ và phụ lục.
- **Toàn bộ metadata hiện đánh dấu text-based**; DOC002 là văn bản Công báo dạng scan nhưng vẫn giữ được nguồn chính thức.
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
| `answerable` | 237 |
| `unanswerable` | 5 |
| `clarification_needed` | 8 |

Câu `unanswerable` kiểm tra khả năng chống hallucination. Câu `clarification_needed` kiểm tra hành vi hỏi lại khi câu hỏi người dùng thiếu thông tin bối cảnh.

### 6.4 Phân bố question_types

Mỗi câu có thể gắn nhiều nhãn. Bảng dưới đây đếm số lần xuất hiện mỗi nhãn trên toàn bộ 250 câu:

| Question type | Số lần xuất hiện | Năng lực RAG được kiểm tra |
|---|---:|---|
| `single_hop` | 150 | Truy hồi một đoạn/trang duy nhất |
| `specific` | 127 | Câu hỏi hẹp, nhắm thông tin cụ thể |
| `factoid` | 76 | Trả lời ngắn: số, tên, ngày, giá trị |
| `definition` | 75 | Hỏi định nghĩa hoặc khái niệm |
| `comparison` | 67 | Đối chiếu hai hoặc nhiều đối tượng |
| `abstract` | 58 | Tổng hợp, diễn giải, rút ý chính |
| `condition` | 55 | Hiểu điều kiện, ngoại lệ, phạm vi áp dụng |
| `multi_hop` | 50 | Kết hợp nhiều đoạn/trang/bảng |
| `list` | 26 | Liệt kê nhiều mục cùng loại |
| `table_lookup` | 26 | Đọc đúng dòng/cột/ô trong bảng |
| `calculation` | 19 | Tính toán từ số liệu trong tài liệu |
| `clarification_needed` | 16 | Nhận diện câu hỏi mơ hồ, hỏi lại |
| `unanswerable` | 5 | Từ chối khi tài liệu không đủ thông tin |

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

- Toàn bộ 50 PDF mở được bằng PyMuPDF.
- Toàn bộ 50 PDF có số trang dưới 50; tài liệu dài nhất hiện là 48 trang.
- 50 file `*_qa.md` khớp 1-1 với 50 file PDF theo `doc_id`.
- Không có câu hỏi nào thiếu trường bắt buộc.
- URL nguồn ghi trong metadata có thể truy cập lại để xác minh tài liệu gốc.
- Phân bố 2 easy / 2 medium / 1 hard được giữ đều trên toàn bộ 50 tài liệu.
