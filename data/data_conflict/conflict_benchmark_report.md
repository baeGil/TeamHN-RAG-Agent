# Data Conflict: DRAG vs Baseline

- Dataset: `D:\TeamHN-RAG-Agent\data\data_conflict\conflict_test_qa.csv`
- PDF root: `D:\TeamHN-RAG-Agent\data\data_conflict`
- Questions: **2**

## Overall

| Method | Recall@5 | MRR@5 | Hit@5 | RAGAS avg | Behavior | Avg latency | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.000 | 0.667 | 1.000 | 1.000 | 1.000 | 6.6s | 4044 |
| drag | 1.000 | 0.667 | 1.000 | 0.920 | 0.900 | 5.9s | 11306 |

## Per Question

| QID | Q type | Conflict | Baseline RAGAS | DRAG RAGAS | Baseline behavior | DRAG behavior | DRAG predicted |
|---|---|---|---:|---:|---:|---:|---|
| DC014 | Simple w. condition | temporal_scope | 1.000 | 1.000 | 1.000 | 1.000 | temporal_scope |
| DC089 | False premise | false_premise | 1.000 | 0.840 | 1.000 | 0.800 | freshness |

## Delta

| Metric | Delta |
|---|---:|
| Recall@5 | 0.000 |
| MRR@5 | 0.000 |
| RAGAS avg | -0.080 |
| Behavior | -0.100 |
| Avg latency | -0.7s |
| Avg tokens | 3631.000 |