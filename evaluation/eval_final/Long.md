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

## By Question Type

| Question type | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple | 101 | 0.515 | 0.210 | 0.355 | 0.338 | 0.229 | 0.307 | 15.37s | 4831.8 |
| simple_w_condition | 32 | 0.094 | 0.019 | 0.062 | 0.049 | 0.030 | 0.043 | 23.16s | 7844.7 |
| set | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 21.14s | 7750.4 |
| comparison | 32 | 0.344 | 0.119 | 0.281 | 0.240 | 0.180 | 0.223 | 34.04s | 12139.5 |
| aggregation | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00s | 0.0 |
| multi_hop | 50 | 0.920 | 0.500 | 0.682 | 0.682 | 0.523 | 0.648 | 110.53s | 43712.2 |
| post_processing_heavy | 14 | 0.214 | 0.043 | 0.107 | 0.043 | 0.021 | 0.051 | 26.82s | 22500.3 |
| false_premise | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 15.46s | 7097.0 |