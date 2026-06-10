# VinFast RAG Benchmark

- Questions: **60**
- Total latency: **187.26s**
- Avg latency: **3.12s**
- Total tokens: **61446**
- Avg tokens: **1024.1**
- Relevance mode: **expected_doc_match**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 0.883 |
| precision@5 | 0.560 |
| recall@5 | 0.883 |
| mrr@5 | 0.807 |
| map@5 | 0.807 |
| ndcg@5 | 0.827 |

## RAGAS

- ragas_faithfulness: **0.690**
- ragas_answer_relevancy: **0.747**
- ragas_answer_correctness: **0.624**
- ragas_semantic_similarity: **0.421**
- ragas_context_precision: **0.560**
- ragas_context_recall: **0.883**
- ragas_context_entity_recall: **0.421**
- ragas_noise_sensitivity: **0.703**

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.76s | 679.0 |
| comparison | 1 | 1.000 | 0.600 | 1.000 | 1.000 | 1.000 | 1.000 | 2.27s | 1436.0 |
| simple | 55 | 0.873 | 0.556 | 0.873 | 0.799 | 0.799 | 0.818 | 3.15s | 1015.0 |
| simple w. condition | 3 | 1.000 | 0.467 | 1.000 | 0.833 | 0.833 | 0.877 | 3.33s | 1169.0 |

> Note: RAGAS values in this run are lightweight heuristic proxies (`ragas_backend=heuristic_proxy`), not scores from the external RAGAS library.