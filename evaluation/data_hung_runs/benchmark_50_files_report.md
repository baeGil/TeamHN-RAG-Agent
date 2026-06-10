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

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 119 | 0.664 | 0.180 | 0.613 | 0.556 | 0.499 | 0.542 | 16.71s | 6910.3 |
| medium | 121 | 0.636 | 0.188 | 0.564 | 0.511 | 0.455 | 0.500 | 16.71s | 10494.3 |
| hard | 85 | 0.929 | 0.416 | 0.704 | 0.789 | 0.609 | 0.699 | 48.99s | 42592.4 |