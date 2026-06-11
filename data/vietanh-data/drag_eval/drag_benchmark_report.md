# DRAG vs Baseline Benchmark

- PDF: `D:\TeamHN-RAG-Agent\data\vietanh-data\DRAG.pdf`
- Dataset: `D:\TeamHN-RAG-Agent\data\vietanh-data\drag_eval\drag_mock_qa.jsonl`
- Questions: **1**
- Retrieval k: **5**

## Overall

| Method | Recall@5 | MRR@5 | Hit@5 | RAGAS avg | Faithfulness | Answer rel | Ctx precision | Ctx recall | Correctness | Behavior | Avg latency | Total latency | Avg tokens | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 31.55s | 31.55s | 1709.0 | 1709 |
| drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 207.62s | 207.62s | 3846.0 | 3846 |

## Delta (DRAG - Baseline)

| Metric | Delta |
|---|---:|
| Recall@5 | +0.000 |
| MRR@5 | +0.500 |
| Hit@5 | +0.000 |
| RAGAS avg | +0.000 |
| Faithfulness | +0.000 |
| Answer relevancy | +0.000 |
| Context precision | +0.000 |
| Context recall | +0.000 |
| Answer correctness | +0.000 |
| Behavior alignment | +0.000 |
| Avg latency | +176.070s |
| Avg tokens | +2137.000 |

## By Conflict Type

### conflicting_opinions

| Method | Recall@5 | MRR@5 | Hit@5 | RAGAS avg | Faithfulness | Answer rel | Ctx precision | Ctx recall | Correctness | Behavior | Avg latency | Total latency | Avg tokens | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 31.55s | 31.55s | 1709.0 | 1709 |
| drag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 207.62s | 207.62s | 3846.0 | 3846 |

## Per Question

| QID | Type | Baseline R@5 | DRAG R@5 | Baseline RAGAS avg | DRAG RAGAS avg | DRAG behavior | DRAG predicted |
|---|---|---:|---:|---:|---:|---:|---|
| D020 | conflicting_opinions | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | conflicting_opinions |