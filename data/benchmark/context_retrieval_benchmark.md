# Context Retrieval Benchmark

Benchmark này dùng 2 tài liệu tiếng Việt có nhu cầu hỏi cao: dạy thêm/học thêm và báo cáo tài chính doanh nghiệp. Bộ câu hỏi kiểm tra PDF scan, bảng số liệu, câu hỏi implicit/explicit, single-hop/multi-hop và câu kiểu metadata/tóm tắt/ý nghĩa.

## Cấu trúc

| Thành phần | Số lượng |
|---|---:|
| PDF | 2 |
| Answerable questions | 30 |
| Unanswerable questions | 4 |
| Easy / Medium / Hard mỗi PDF | 5 / 5 / 5 |

## Dimensions

| Dimension | Giá trị |
|---|---|
| `difficulty` | `easy`, `medium`, `hard`, `unanswerable` |
| `explicitness` | `explicit`, `implicit` |
| `hop_type` | `single-hop`, `multi-hop` |
| `query_style` | `direct`, `metadata` |
| `evidence_type` | `text`, `table`, `mixed`, `metadata` |

## PDF

| Doc ID | File | Ghi chú |
|---|---|---|
| `tt29_day_them_2024` | `data/benchmark/pdfs/tt-29-2024-day-them-hoc-them-scanned.pdf` | Scanned/image-only PDF converted from the official Công báo PDF; high public-interest topic for parents, students, teachers, and tutoring centers. |
| `mwg_fs_q1_2026` | `data/benchmark/pdfs/mwg-bao-cao-tai-chinh-rieng-q1-2026.pdf` | 26-page scanned/image-only table-heavy finance document; suitable for OCR/VLM and table reasoning questions. |

## Distribution

- `difficulty`: `easy`=10, `hard`=10, `medium`=10, `unanswerable`=4
- `explicitness`: `explicit`=15, `implicit`=19
- `hop_type`: `multi-hop`=17, `single-hop`=17
- `query_style`: `direct`=19, `metadata`=15
- `evidence_type`: `metadata`=4, `mixed`=3, `table`=8, `text`=19
- `answerability`: `answerable`=30, `unanswerable`=4

## Questions

### Thông tư 29/2024/TT-BGDĐT quy định về dạy thêm, học thêm

| ID | Mức | Explicit | Hop | Style | Evidence | Câu hỏi |
|---|---|---|---|---|---|---|
| `tt29_easy_01` | easy | explicit | single-hop | direct | text | Thông tư 29/2024/TT-BGDĐT quy định về hoạt động nào và áp dụng cho những ai? |
| `tt29_easy_02` | easy | explicit | single-hop | direct | text | Theo Điều 2, dạy thêm, học thêm được hiểu là gì? |
| `tt29_easy_03` | easy | explicit | single-hop | direct | text | Thông tư có hiệu lực từ ngày nào và thay thế văn bản nào? |
| `tt29_easy_04` | easy | explicit | single-hop | metadata | text | Phụ lục của Thông tư gồm những mẫu biểu chính nào? |
| `tt29_easy_05` | easy | explicit | single-hop | direct | text | Một lớp dạy thêm trong nhà trường được xếp tối đa bao nhiêu học sinh và mỗi môn học được tổ chức không quá bao nhiêu tiết mỗi tuần? |
| `tt29_medium_01` | medium | implicit | single-hop | direct | text | Con học tiểu học muốn học thêm Toán theo lớp học thêm thông thường thì Thông tư nhìn chung có cho tổ chức không? |
| `tt29_medium_02` | medium | explicit | multi-hop | direct | text | Dạy thêm trong nhà trường có được thu tiền không và chỉ dành cho những nhóm học sinh nào? |
| `tt29_medium_03` | medium | implicit | multi-hop | direct | text | Một giáo viên đang dạy chính khóa có thể mở lớp thu tiền ngoài nhà trường cho chính học sinh mình đang được phân công dạy không; nếu tham gia dạy thêm ngoài trường nói chung thì cần báo cáo ai? |
| `tt29_medium_04` | medium | implicit | multi-hop | metadata | text | Nhìn từ các nguyên tắc ở Điều 3, Thông tư muốn giảm những rủi ro nào của việc dạy thêm? |
| `tt29_medium_05` | medium | explicit | multi-hop | metadata | text | Mẫu số 02 yêu cầu cơ sở dạy thêm ngoài nhà trường công khai những nhóm thông tin nào trước khi tuyển sinh? |
| `tt29_hard_01` | hard | implicit | multi-hop | metadata | text | Nếu tóm tắt thành một thông điệp quản lý, Thông tư phân biệt dạy thêm trong trường và ngoài trường như thế nào? |
| `tt29_hard_02` | hard | implicit | multi-hop | metadata | text | Vì sao văn bản này không chỉ là danh sách điều cấm, mà còn là cơ chế quản trị nhiều tầng đối với dạy thêm? |
| `tt29_hard_03` | hard | implicit | multi-hop | direct | text | Nếu một phụ huynh phản ánh trung tâm dạy thêm không công khai học phí, danh sách giáo viên và lớp học có vấn đề an toàn, tài liệu gợi ý những lớp trách nhiệm nào có thể liên quan? |
| `tt29_hard_04` | hard | implicit | multi-hop | metadata | text | Ba mẫu trong phụ lục cho thấy luồng thông tin quản lý dạy thêm đi theo những hướng nào? |
| `tt29_hard_05` | hard | implicit | multi-hop | direct | text | Một trường muốn mở lớp ôn thi tốt nghiệp cho học sinh cuối cấp tự nguyện. Để đúng tinh thần Thông tư, trường cần đáp ứng các điều kiện vận hành nào? |
| `tt29_unanswerable_01` | unanswerable | explicit | single-hop | direct | text | Thông tư quy định mức học phí cụ thể mỗi giờ cho từng môn dạy thêm ngoài nhà trường là bao nhiêu? |
| `tt29_unanswerable_02` | unanswerable | implicit | single-hop | metadata | metadata | Dựa vào Thông tư này, có thể kết luận trung tâm dạy thêm cụ thể nào ở Hà Nội đang được cấp phép hay không? |

### MWG - Báo cáo tài chính riêng giữa niên độ Quý 1/2026

| ID | Mức | Explicit | Hop | Style | Evidence | Câu hỏi |
|---|---|---|---|---|---|---|
| `mwg_easy_01` | easy | explicit | single-hop | metadata | metadata | Tài liệu MWG này là loại báo cáo nào, của công ty nào và cho kỳ nào? |
| `mwg_easy_02` | easy | explicit | single-hop | metadata | metadata | Mục lục cho thấy báo cáo có những phần chính nào? |
| `mwg_easy_03` | easy | explicit | single-hop | direct | table | Tổng tài sản của MWG tại ngày 31/03/2026 là bao nhiêu? |
| `mwg_easy_04` | easy | explicit | single-hop | direct | table | Khoản tiền gửi ngân hàng của MWG tại ngày 31/03/2026 là bao nhiêu? |
| `mwg_easy_05` | easy | explicit | single-hop | metadata | text | Hoạt động chính của MWG trong báo cáo này được mô tả là gì? |
| `mwg_medium_01` | medium | implicit | multi-hop | direct | table | Tổng tài sản cuối quý 1/2026 tăng hay giảm so với cuối năm 2025, và biến động đó chủ yếu đi cùng chiều với tài sản ngắn hạn hay tài sản dài hạn? |
| `mwg_medium_02` | medium | implicit | multi-hop | direct | table | Lợi nhuận sau thuế quý 1/2026 giảm mạnh so với cùng kỳ 2025; nhìn vào bảng kết quả kinh doanh, nguyên nhân số liệu nổi bật nhất là gì? |
| `mwg_medium_03` | medium | explicit | single-hop | direct | table | Trong báo cáo lưu chuyển tiền tệ quý 1/2026, dòng tiền thuần từ hoạt động kinh doanh, đầu tư và tài chính lần lượt là bao nhiêu? |
| `mwg_medium_04` | medium | implicit | single-hop | metadata | text | Có nên dùng riêng báo cáo tài chính riêng này để hiểu đầy đủ tình hình của cả tập đoàn MWG không? |
| `mwg_medium_05` | medium | implicit | multi-hop | direct | table | Vì sao tiền cuối kỳ giảm mạnh dù dòng tiền kinh doanh quý 1/2026 là dương? |
| `mwg_hard_01` | hard | implicit | multi-hop | metadata | mixed | Nhìn ở mức ý nghĩa kinh doanh, báo cáo này cho thấy MWG trong vai trò công ty mẹ tạo doanh thu/lợi nhuận chủ yếu qua hoạt động nào? |
| `mwg_hard_02` | hard | implicit | multi-hop | direct | table | Vốn chủ sở hữu cuối quý 1/2026 tăng ít so với đầu năm. Bảng biến động vốn cho thấy hai tác động chính nào tạo ra mức tăng đó? |
| `mwg_hard_03` | hard | implicit | multi-hop | metadata | mixed | Nếu tóm tắt cấu trúc tài chính của công ty mẹ, vì sao có thể nói khoản đầu tư vào công ty con quan trọng hơn nhiều so với tiền mặt hoặc vay ngắn hạn? |
| `mwg_hard_04` | hard | implicit | multi-hop | direct | table | Vì sao báo cáo có lợi nhuận kế toán trước thuế 79,28 tỷ đồng nhưng chi phí thuế TNDN hiện hành lại bằng 0? |
| `mwg_hard_05` | hard | implicit | multi-hop | metadata | mixed | Phần sự kiện sau ngày kết thúc kỳ kế toán và chữ ký cuối báo cáo giúp hiểu gì về trạng thái hoàn tất của báo cáo? |
| `mwg_unanswerable_01` | unanswerable | implicit | single-hop | metadata | metadata | Báo cáo này có đủ số liệu để kết luận doanh thu bán lẻ hợp nhất của toàn tập đoàn MWG quý 1/2026 là bao nhiêu không? |
| `mwg_unanswerable_02` | unanswerable | explicit | single-hop | direct | text | Kiểm toán viên đã đưa ý kiến kiểm toán nào đối với báo cáo tài chính riêng quý 1/2026 này? |

## Files

- Full JSON: `data/benchmark/context_retrieval_benchmark.json`
- Summary: `data/benchmark/context_retrieval_benchmark.md`
