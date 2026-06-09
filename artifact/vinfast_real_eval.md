# HCMUE Education RAG Benchmark

- Questions: **10**
- Total latency: **25.60s**
- Avg latency: **2.56s**
- Total tokens: **14842**
- Avg tokens: **1484.2**
- Relevance mode: **llm**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 1.000 |
| precision@5 | 0.200 |
| recall@5 | 0.500 |
| mrr@5 | 0.933 |
| map@5 | 0.747 |
| ndcg@5 | 0.840 |

## RAGAS

- ragas_faithfulness: **0.755**
- ragas_answer_relevancy: **0.750**
- ragas_answer_correctness: **0.760**
- ragas_context_precision: **0.695**
- ragas_context_recall: **0.700**
- ragas_context_entity_recall: **0.695**
- ragas_noise_sensitivity: **0.110**
- ragas_semantic_similarity: **0.725**

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple | 10 | 1.000 | 0.200 | 0.500 | 0.933 | 0.747 | 0.840 | 2.56s | 1484.2 |
