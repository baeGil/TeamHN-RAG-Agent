# Dữ liệu mock đánh giá DRAG

Thư mục này chứa bộ benchmark mock cho `data/vietanh-data/DRAG.pdf`.

- `drag_mock_qa.jsonl`: 24 câu hỏi tiếng Việt mức medium/hard, đã trộn thứ tự, cân bằng 12 câu `no_conflict` và 12 câu có conflict.
- Các câu có conflict bao phủ `complementary_information`, `conflicting_opinions`, `freshness`, và `misinformation`.
- Sau khi chạy, script benchmark sẽ ghi `drag_benchmark_results.json`, `drag_benchmark_report.md`, `drag_answer_comparison.md`, và các file cache judge vào thư mục này.

Chạy từ thư mục `backend/`:

```bash
python -m eval.drag_benchmark --reset
```

Chạy thử một câu trước:

```bash
python -m eval.drag_benchmark --qid D001
```

Hoặc chạy câu đầu tiên trong dataset:

```bash
python -m eval.drag_benchmark --limit 1
```

Nếu đã có `drag_benchmark_results.json` và chỉ muốn tạo lại file so sánh câu trả lời:

```bash
python -m eval.drag_benchmark --write-comparison-only
```

Benchmark so sánh baseline RAG với DRAG trên cùng PDF và cùng bộ câu hỏi, rồi báo cáo:

- Retrieval: Recall@5, MRR@5, Hit@5.
- RAGAS-style: faithfulness, answer relevancy, context precision, context recall, answer correctness, và `RAGAS avg`.
- DRAG riêng: behavior alignment, tức câu trả lời có xử lý đúng kiểu conflict hay không.
- Cost/performance: độ trễ trung bình/tổng và token trung bình/tổng.
