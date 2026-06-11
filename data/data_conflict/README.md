# Data Conflict QA Benchmark

Thư mục này chứa bộ câu hỏi kiểm thử conflict được sinh từ các PDF đang có trong `data/data_conflict/`.

## Files

- `conflict_test_qa.jsonl`: 100 câu hỏi, mỗi dòng là một JSON object.
- `conflict_test_qa.csv`: cùng dữ liệu ở dạng bảng CSV UTF-8 BOM để mở bằng Excel.
- `pdf/`: 3 PDF scan/trích đoạn nghị định giảm thuế GTGT.
- `pdfs/`: 5 PDF text-based của World Bank về cập nhật kinh tế vĩ mô Việt Nam.

## Schema

- `qid`: mã câu hỏi.
- `question_type`: một trong 8 loại câu hỏi: Simple, Simple w. condition, Set, Comparison, Aggregation, Multi-hop, Post-processing heavy, False premise.
- `difficulty`: mức đọc hiểu dự kiến.
- `conflict_type`: kiểu conflict/no-conflict chính.
- `question`: câu hỏi tiếng Việt.
- `expected_answer`: câu trả lời kỳ vọng, kể cả khi có conflict.
- `expected_behavior`: hành vi mong đợi của hệ thống khi trả lời.
- `evidence_pages`: trang/tài liệu làm căn cứ.
- `source_documents`: tài liệu nguồn chính.
- `conflicting_sources`: nguồn có khả năng gây nhiễu hoặc mốc cũ hơn, nếu có.

## Distribution

- Simple: 13 câu
- Simple w. condition: 13 câu
- Set: 12 câu
- Comparison: 12 câu
- Aggregation: 12 câu
- Multi-hop: 13 câu
- Post-processing heavy: 12 câu
- False premise: 13 câu

## Notes

- Các PDF nghị định trong `pdf/` là scan nên script không OCR tự động; các câu hỏi luật chỉ dùng những trang đã kiểm tra trực quan được trong trích đoạn.
- Các câu hỏi World Bank ưu tiên mốc thời gian cụ thể. Khi nhiều PDF đưa các số khác nhau, expected answer giải thích số nào thuộc tháng nào thay vì coi mọi số là mâu thuẫn ngang hàng.
- Bộ test được thiết kế để kiểm RAG/DRAG về temporal conflict, false premise, multi-hop evidence, và xử lý câu hỏi thiếu điều kiện.
