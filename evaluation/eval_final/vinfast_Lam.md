# VinFast RAG Benchmark

- Questions: **200**
- Total latency: **7002.01s**
- Avg latency: **35.01s**
- Total tokens: **92153861**
- Avg tokens: **460769.3**
- Relevance mode: **llm**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 0.990 |
| precision@5 | 0.198 |
| recall@5 | 0.495 |
| mrr@5 | 0.980 |
| map@5 | 0.784 |
| ndcg@5 | 0.882 |

## RAGAS

- ragas_faithfulness: **0.682**
- ragas_answer_relevancy: **0.881**
- ragas_answer_correctness: **0.721**
- ragas_semantic_similarity: **0.794**
- ragas_context_precision: **0.799**
- ragas_context_recall: **0.738**
- ragas_context_entity_recall: **0.704**
- ragas_noise_sensitivity: **0.809**

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 26 | 1.000 | 0.200 | 0.500 | 0.974 | 0.780 | 0.877 | 33.38s | 492736.6 |
| comparison | 25 | 1.000 | 0.200 | 0.500 | 1.000 | 0.800 | 0.900 | 29.18s | 424895.7 |
| false premise | 22 | 1.000 | 0.200 | 0.500 | 0.939 | 0.752 | 0.845 | 38.85s | 691759.4 |
| multi-hop | 25 | 1.000 | 0.200 | 0.500 | 1.000 | 0.800 | 0.900 | 45.42s | 543844.4 |
| post-processing | 12 | 0.917 | 0.183 | 0.458 | 0.917 | 0.733 | 0.825 | 33.23s | 387159.1 |
| post-processing heavy | 13 | 1.000 | 0.200 | 0.500 | 1.000 | 0.800 | 0.900 | 50.28s | 840949.6 |
| set | 26 | 0.962 | 0.192 | 0.481 | 0.962 | 0.769 | 0.865 | 30.47s | 375617.2 |
| simple | 26 | 1.000 | 0.200 | 0.500 | 1.000 | 0.800 | 0.900 | 31.44s | 262568.5 |
| simple w. condition | 25 | 1.000 | 0.200 | 0.500 | 1.000 | 0.800 | 0.900 | 30.10s | 309376.7 |
