# HCMUE Education RAG Benchmark

- Questions: **170**
- Total latency: **1932.89s**
- Avg latency: **11.37s**
- Total tokens: **2433638**
- Avg tokens: **14315.5**
- Relevance mode: **llm**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 0.835 |
| precision@5 | 0.299 |
| recall@5 | 0.609 |
| mrr@5 | 0.705 |
| map@5 | 0.499 |
| ndcg@5 | 0.583 |

## RAGAS

- ragas_faithfulness: **0.922**
- ragas_answer_relevancy: **0.895**
- ragas_answer_correctness: **0.835**
- ragas_semantic_similarity: **0.839**
- ragas_context_precision: **0.748**
- ragas_context_recall: **0.866**
- ragas_context_entity_recall: **0.836**
- ragas_noise_sensitivity: **0.183**

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregation | 8 | 0.875 | 0.250 | 0.833 | 0.812 | 0.704 | 0.765 | 7.41s | 9610.5 |
| Comparison | 11 | 0.727 | 0.255 | 0.568 | 0.621 | 0.456 | 0.528 | 13.99s | 19305.5 |
| Post-processing heavy | 74 | 0.797 | 0.259 | 0.545 | 0.634 | 0.430 | 0.514 | 15.07s | 20396.9 |
| Set | 29 | 0.862 | 0.297 | 0.617 | 0.733 | 0.502 | 0.592 | 9.02s | 9007.3 |
| Simple | 25 | 0.920 | 0.408 | 0.745 | 0.893 | 0.690 | 0.761 | 6.87s | 7669.4 |
| Simple w. condition | 23 | 0.870 | 0.348 | 0.601 | 0.693 | 0.461 | 0.562 | 7.45s | 7916.3 |
