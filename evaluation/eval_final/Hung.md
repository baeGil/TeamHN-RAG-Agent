# 50-Files PDF RAG Benchmark

- Questions: **325**
- Total latency: **8174.55s**
- Avg latency: **25.15s**
- Total tokens: **5712499**
- Avg tokens: **17576.9**
- Relevance mode: **llm**
- PDF parser: **Reducto only**
- RAGAS mode: **llm_proxy**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 0.723 |
| precision@5 | 0.245 |
| recall@5 | 0.618 |
| mrr@5 | 0.600 |
| map@5 | 0.511 |
| ndcg@5 | 0.567 |

## RAGAS

- ragas_faithfulness: **0.772**
- ragas_answer_relevancy: **0.725**
- ragas_answer_correctness: **0.656**
- ragas_semantic_similarity: **0.675**
- ragas_context_precision: **0.713**
- ragas_context_recall: **0.723**
- ragas_context_entity_recall: **0.723**
- ragas_noise_sensitivity: **0.246**

## By Question Type

| Question type | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple | 209 | 0.646 | 0.181 | 0.584 | 0.525 | 0.469 | 0.514 | 16.21s | 8123.0 |
| Simple w. condition | 17 | 0.471 | 0.176 | 0.451 | 0.412 | 0.371 | 0.404 | 23.77s | 11218.3 |
| Set | 17 | 0.941 | 0.341 | 0.799 | 0.838 | 0.705 | 0.767 | 53.00s | 21674.9 |
| Comparison | 6 | 0.500 | 0.233 | 0.444 | 0.389 | 0.374 | 0.403 | 76.91s | 30487.3 |
| Aggregation | 5 | 1.000 | 0.360 | 0.640 | 0.800 | 0.562 | 0.647 | 14.64s | 20379.8 |
| Multi-hop | 71 | 0.958 | 0.420 | 0.728 | 0.813 | 0.632 | 0.725 | 41.51s | 44659.0 |
| Post-processing heavy | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00s | 0.0 |
| False premise | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00s | 0.0 |
