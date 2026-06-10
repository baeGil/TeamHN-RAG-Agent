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

## By Difficulty

| Difficulty | n | Hit@5 | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 | Avg latency | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggregation | 35 | 0.971 | 0.234 | 0.590 | 0.938 | 0.769 | 0.855 | 26.54s | 368249.0 |
| comparison | 37 | 0.919 | 0.227 | 0.534 | 0.887 | 0.703 | 0.792 | 23.94s | 292870.0 |
| easy | 219 | 0.594 | 0.194 | 0.495 | 0.457 | 0.376 | 0.435 | 15.75s | 5964.3 |
| false premise | 22 | 1.000 | 0.200 | 0.500 | 0.939 | 0.752 | 0.845 | 38.85s | 691759.4 |
| hard | 135 | 0.926 | 0.447 | 0.696 | 0.749 | 0.577 | 0.680 | 71.78s | 43007.2 |
| medium | 221 | 0.430 | 0.127 | 0.367 | 0.325 | 0.281 | 0.317 | 21.42s | 10805.3 |
| multi-hop | 25 | 1.000 | 0.200 | 0.500 | 1.000 | 0.800 | 0.900 | 45.42s | 543844.4 |
| post-processing | 12 | 0.917 | 0.183 | 0.458 | 0.917 | 0.733 | 0.825 | 33.23s | 387159.1 |
| post-processing heavy | 87 | 0.827 | 0.250 | 0.538 | 0.689 | 0.485 | 0.572 | 20.33s | 143008.2 |
| set | 55 | 0.909 | 0.248 | 0.553 | 0.841 | 0.628 | 0.721 | 19.16s | 182313.8 |
| simple | 106 | 0.915 | 0.434 | 0.751 | 0.871 | 0.774 | 0.825 | 10.97s | 66739.1 |
| simple w. condition | 51 | 0.941 | 0.282 | 0.575 | 0.852 | 0.649 | 0.746 | 18.31s | 155294.1 |

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
