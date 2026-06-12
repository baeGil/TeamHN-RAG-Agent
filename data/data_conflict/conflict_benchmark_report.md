# Data Conflict: DRAG vs Baseline

- Dataset: `D:\TeamHN-RAG-Agent\data\data_conflict\conflict_test_qa.csv`
- PDF root: `D:\TeamHN-RAG-Agent\data\data_conflict`
- Questions: **20**

## Overall

| Method | Recall@5 | MRR@5 | Hit@5 | RAGAS avg | Behavior | Avg latency | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.800 | 0.481 | 0.800 | 0.511 | 0.475 | 3.7s | 42465 |
| drag | 0.800 | 0.481 | 0.800 | 0.628 | 0.675 | 8.2s | 89067 |

## Per Question

| QID | Q type | Conflict | Baseline RAGAS | DRAG RAGAS | Baseline behavior | DRAG behavior | DRAG predicted |
|---|---|---|---:|---:|---:|---:|---|
| DC001 | Simple | no_conflict | 0.200 | 0.400 | 0.000 | 0.000 | freshness |
| DC002 | Simple | no_conflict | 0.500 | 0.800 | 0.000 | 1.000 | freshness |
| DC003 | Simple | no_conflict | 0.000 | 0.600 | 0.000 | 1.000 | freshness |
| DC004 | Simple | no_conflict | 1.000 | 0.680 | 1.000 | 1.000 | freshness |
| DC005 | Simple | no_conflict | 0.000 | 0.100 | 0.000 | 0.000 | freshness |
| DC006 | Simple | no_conflict | 0.000 | 0.000 | 0.000 | 0.000 | freshness |
| DC007 | Simple | no_conflict | 1.000 | 1.000 | 1.000 | 1.000 | no_conflict |
| DC008 | Simple | no_conflict | 1.000 | 1.000 | 1.000 | 1.000 | freshness |
| DC009 | Simple | no_conflict | 1.000 | 1.000 | 1.000 | 1.000 | freshness |
| DC010 | Simple | no_conflict | 1.000 | 0.200 | 1.000 | 0.000 | freshness |
| DC014 | Simple w. condition | temporal_scope | 0.400 | 0.760 | 0.500 | 1.000 | freshness |
| DC015 | Simple w. condition | temporal_scope | 0.840 | 0.800 | 1.000 | 1.000 | freshness |
| DC020 | Simple w. condition | freshness | 0.200 | 0.200 | 0.000 | 0.000 | freshness |
| DC021 | Simple w. condition | freshness | 1.000 | 1.000 | 1.000 | 1.000 | freshness |
| DC022 | Simple w. condition | freshness | 1.000 | 1.000 | 1.000 | 1.000 | freshness |
| DC037 | Set | complementary_information | 0.000 | 0.420 | 0.000 | 0.500 | freshness |
| DC063 | Multi-hop | temporal_scope | 0.000 | 0.680 | 0.000 | 1.000 | freshness |
| DC070 | Multi-hop | freshness | 0.880 | 0.920 | 1.000 | 1.000 | freshness |
| DC088 | False premise | false_premise | 0.000 | 0.800 | 0.000 | 1.000 | freshness |
| DC089 | False premise | false_premise | 0.200 | 0.200 | 0.000 | 0.000 | freshness |

## Delta

| Metric | Delta |
|---|---:|
| Recall@5 | 0.000 |
| MRR@5 | 0.000 |
| RAGAS avg | 0.117 |
| Behavior | 0.200 |
| Avg latency | 4.5s |
| Avg tokens | 2330.100 |