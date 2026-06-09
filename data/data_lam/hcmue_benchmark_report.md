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
| easy | 68 | 0.912 | 0.362 | 0.673 | 0.802 | 0.570 | 0.660 | 7.55s | 8111.5 |
| hard | 34 | 0.735 | 0.259 | 0.381 | 0.499 | 0.258 | 0.357 | 14.64s | 17979.9 |
| medium | 68 | 0.809 | 0.256 | 0.659 | 0.711 | 0.549 | 0.618 | 13.56s | 18687.3 |