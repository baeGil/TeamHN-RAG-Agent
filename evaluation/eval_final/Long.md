# 50-Files PDF RAG Benchmark

- Questions: **250**
- Total latency: **9700.20s**
- Avg latency: **38.80s**
- Total tokens: **3787616**
- Avg tokens: **15150.5**
- Relevance mode: **llm**
- PDF parser: **Reducto only**
- RAGAS mode: **llm_proxy**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 0.460 |
| precision@5 | 0.205 |
| recall@5 | 0.330 |
| mrr@5 | 0.312 |
| map@5 | 0.225 |
| ndcg@5 | 0.290 |

## RAGAS

- ragas_faithfulness: **0.575**
- ragas_answer_relevancy: **0.575**
- ragas_answer_correctness: **0.362**
- ragas_semantic_similarity: **0.481**
- ragas_context_precision: **0.572**
- ragas_context_recall: **0.566**
- ragas_context_entity_recall: **0.564**
- ragas_noise_sensitivity: **0.387**

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 100 | 0.510 | 0.210 | 0.354 | 0.340 | 0.231 | 0.307 | 14.61s | 4838.5 |
| medium | 100 | 0.180 | 0.052 | 0.130 | 0.101 | 0.071 | 0.095 | 27.13s | 11181.5 |
| hard | 50 | 0.920 | 0.500 | 0.682 | 0.682 | 0.523 | 0.648 | 110.53s | 43712.2 |