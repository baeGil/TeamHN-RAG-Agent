# Final Benchmark EDA

Notebook chính:

```powershell
evaluation/eval_final/benchmark_final_eda.ipynb
```

## Cài dependency

Từ repo root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy matplotlib seaborn openpyxl ipykernel
```

Nếu muốn chạy bằng Jupyter Lab thay vì VS Code:

```powershell
python -m pip install jupyterlab
jupyter lab evaluation/eval_final/benchmark_final_eda.ipynb
```

## Chạy trong VS Code

1. Mở `evaluation/eval_final/benchmark_final_eda.ipynb`.
2. Chọn kernel là `.venv`.
3. Bấm `Run All`.

Notebook sẽ tự tìm dữ liệu trong `evaluation/eval_final/` nếu bạn mở từ repo root.
Các nguồn dữ liệu vẫn được nạp từ nhiều file/run, nhưng EDA hiện xem toàn bộ 1005 câu như một benchmark chung đại diện cho cả nhóm; cột `run` chỉ dùng để truy vết khi cần audit.

## Dữ liệu đầu vào

- `per_question_Hung.jsonl`
- `per_question_Long.jsonl`
- `vinfast_Lam.csv`
- `hcmue_benchmark_review.xlsx`
- `vinfast_benchmark_results_vsf_rag_60pdf.json`

## Phân loại câu hỏi

EDA hiện dùng 8 loại CRAG:

- Simple
- Simple w. condition
- Set
- Comparison
- Aggregation
- Multi-hop
- Post-processing heavy
- False premise

Với `Hung`, dữ liệu gốc chưa có cột `question_types`, nên notebook phân loại lại bằng heuristic từ nội dung câu hỏi. Các nguồn còn lại dùng nhãn có sẵn hoặc cột `do_kho` rồi chuẩn hóa về 8 loại trên.
