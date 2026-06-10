# 50-Files PDF Evaluation Pipeline

Pipeline riêng cho bộ dữ liệu `data/50_files`.

## Điểm bắt buộc

- Ingest PDF bằng Reducto trực tiếp qua `parse_pdf_reducto`.
- Không dùng `load_pdf()` vì hàm đó có fallback local.
- Nếu thiếu `REDUCTO_API_KEY`, thiếu `OPENAI_API_KEY`, hoặc Reducto lỗi ở bất kỳ PDF nào, pipeline dừng.
- Storage mặc định: `backend/storage_50_files_reducto`, tách khỏi storage app hiện tại.

## Chạy

Từ repo root:

```powershell
python evaluation/benchmark_50_files_pipeline.py --reset
```

Chạy thử ít câu sau khi ingest đủ 50 PDF:

```powershell
python evaluation/benchmark_50_files_pipeline.py --limit 5
```

Chỉ chạy riêng phần Reducto ingest/index, chưa đánh giá:

```powershell
python evaluation/benchmark_50_files_pipeline.py --reset --ingest-only
```

Chỉ ingest một hoặc vài tài liệu:

```powershell
python evaluation/benchmark_50_files_pipeline.py --reset --ingest-only --doc-id DOC001
python evaluation/benchmark_50_files_pipeline.py --reset --ingest-only --doc-id DOC001,DOC048
python evaluation/benchmark_50_files_pipeline.py --reset --ingest-only --pdf-name DOC001_vinfast_vf3_brochure.pdf
```

Ingest một thư mục PDF khác và lưu vào storage riêng:

```powershell
python evaluation/benchmark_50_files_pipeline.py `
  --reset `
  --ingest-only `
  --pdf-dir data/data_hung/dataset `
  --storage backend/storage_data_hung_reducto `
  --out-dir evaluation/data_hung_runs
```

Sau khi ingest xong, chạy đánh giá trên cùng bộ `data_hung` bằng storage đã có:

```powershell
python evaluation/benchmark_50_files_pipeline.py `
  --pdf-dir data/data_hung/dataset `
  --storage backend/storage_data_hung_reducto `
  --out-dir evaluation/data_hung_runs
```

Đổi mode Reducto:

```powershell
python evaluation/benchmark_50_files_pipeline.py --reducto-mode default --reset
```

## Output

Mặc định ghi vào `evaluation/benchmark_50_files_runs/`:

- `ingest_results.json`: metadata ingest/Reducto.
- `ingest_summary.json`: chỉ có khi chạy `--ingest-only`.
- `relevance_judgments.json`: cache LLM judge cho retrieval relevance.
- `ragas_proxy_judgments.json`: cache LLM judge cho nhóm metric `ragas_*`.
- `per_question.jsonl`: kết quả từng câu, gồm `retrieved_chunks` top-5 và `judge_pool_chunks`.
- `benchmark_50_files_results.json`: aggregate dạng JSON.
- `benchmark_50_files_report.md`: report cùng cấu trúc với benchmark mẫu.

## Metrics

Retrieval:

- `hit@5`
- `precision@5`
- `recall@5`
- `mrr@5`
- `map@5`
- `ndcg@5`

Answer/RAGAS-compatible:

- `ragas_faithfulness`
- `ragas_answer_relevancy`
- `ragas_answer_correctness`
- `ragas_semantic_similarity`
- `ragas_context_precision`
- `ragas_context_recall`
- `ragas_context_entity_recall`
- `ragas_noise_sensitivity`

Nhóm `ragas_*` trong pipeline này được chấm bằng LLM judge nội bộ với thang 0-1 để xuất đủ metric theo report mẫu, không phụ thuộc package `ragas`.
