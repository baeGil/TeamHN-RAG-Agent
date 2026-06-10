# Final RAG Benchmark

- Questions: **1005**
- Total latency: **26996.91s**
- Avg latency: **26.86s**
- Total tokens: **104149060**
- Avg tokens: **103630.9**
- Relevance mode: **mixed (llm, expected_doc_match)**
- PDF parser: **mixed**
- RAGAS mode: **mixed (llm_proxy, heuristic_proxy)**

## Retrieval

| Metric | Score |
|---|---:|
| hit@5 | 0.739 |
| precision@5 | 0.254 |
| recall@5 | 0.536 |
| mrr@5 | 0.634 |
| map@5 | 0.510 |
| ndcg@5 | 0.579 |

## RAGAS

- ragas_faithfulness: **0.726**
- ragas_answer_relevancy: **0.749**
- ragas_answer_correctness: **0.624**
- ragas_semantic_similarity: **0.663**
- ragas_context_precision: **0.692**
- ragas_context_recall: **0.721**
- ragas_context_entity_recall: **0.681**
- ragas_noise_sensitivity: **0.410**

## By Question Type

| Question type | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple | 416 | 0.683 | 0.252 | 0.571 | 0.568 | 0.488 | 0.543 | 14.67s | 22259.8 |
| Simple w. condition | 100 | 0.590 | 0.180 | 0.390 | 0.520 | 0.404 | 0.463 | 20.79s | 83617.4 |
| Set | 88 | 0.750 | 0.221 | 0.500 | 0.688 | 0.529 | 0.599 | 26.06s | 119542.5 |
| Comparison | 75 | 0.640 | 0.181 | 0.419 | 0.571 | 0.454 | 0.518 | 32.49s | 152101.0 |
| Aggregation | 40 | 0.975 | 0.250 | 0.597 | 0.921 | 0.743 | 0.829 | 25.05s | 324765.3 |
| Multi-hop | 146 | 0.952 | 0.410 | 0.674 | 0.800 | 0.623 | 0.729 | 65.81s | 129811.7 |
| Post-processing heavy | 113 | 0.761 | 0.217 | 0.476 | 0.633 | 0.454 | 0.534 | 22.50s | 154005.6 |
| False premise | 27 | 0.815 | 0.163 | 0.407 | 0.765 | 0.612 | 0.689 | 34.52s | 564970.1 |

## Included Runs

| Run | Questions |
|---|---:|
| Hung | 325 |
| Long | 250 |
| VinFast Lam | 200 |
| HCMUE | 170 |
| VinFast VSF RAG 60 PDF | 60 |
| **Total** | **1005** |

> The VinFast VSF RAG 40 PDF report is a subset of the 60 PDF run and is excluded from the total to avoid double-counting.
