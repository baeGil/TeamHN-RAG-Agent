# Pipeline Benchmark Report

**Parser:** PYMUPDF  **Doc summary (CCH):** True  **RSE:** True

## 1. Indexing

| Phase | Time (ms) |
|---|---|
| PDF Parsing (pymupdf) | 16395 |
| Chunking | 0 |
| Doc Summary LLM (CCH) | 2870 |
| Embedding (22 chunks) | 2931 |
| Index build (BM25 + TurboVec) | 3542 |
| **Total ingest** | **29064** |

| Metric | Value |
|---|---|
| Pages | 8 |
| Blocks (parser output) | 22 |
| Chunks (after splitting) | 22 |
| Sections detected | 15 |
| Avg chunk size (chars) | 713 |
| Min / Max chunk size | 19 / 1758 |

**Cost:**
- Embedding: ~6,171 tokens → **$0.0001**
- Doc summary: 6128 in + 40 out → **$0.00094**
- **Total indexing cost: $0.0022**

**Generated doc summary (CCH header):**
> Tài liệu này trình bày về phương pháp Weighted Potential Fast Marching Firework để quy hoạch đường đi an toàn đa đích cho robot di động trong môi trường nhà máy hạt nhân.

## 2. Retrieval – RSE ON

**Keyword hit rate:** 100%  avg overlap: 61%

### Latency

| Phase | Mean ms | P50 ms | P95 ms | Max ms |
|---|---|---|---|---|
| bm25 | 3.9 | 3.0 | 11.7 | 11.7 |
| embed_query | 207.4 | 166.1 | 768.6 | 768.6 |
| dense_search | 11.6 | 0.7 | 212.4 | 212.4 |
| rrf | 0.1 | 0.0 | 0.2 | 0.2 |
| rerank | 2973.2 | 2348.6 | 14815.9 | 14815.9 |
| rse | 0.6 | 0.2 | 7.4 | 7.4 |
| total | 3197.1 | 2516.0 | 15304.3 | 15304.3 |

### Context quality

| Metric | Value |
|---|---|
| avg_context_chars | 1820.95 |
| avg_n_returned | 1.70 |
| avg_sections | 1.20 |
| avg_segment_chars | 1205.00 |
| avg_keyword_overlap_pct | 0.61 |

### Per-query breakdown

| Q | Diff | BM25 | Dense | Rerank | RSE | Total | Ctx chars | kw% | Hit |
|---|---|---|---|---|---|---|---|---|---|
| e1 | easy | 2 | 478 | 14816 | 7 | 15304 | 1427 | 84% | ✓ |
| e2 | easy | 12 | 312 | 2349 | 0 | 2674 | 1427 | 89% | ✓ |
| e3 | easy | 2 | 168 | 2353 | 0 | 2523 | 1427 | 98% | ✓ |
| e4 | easy | 3 | 769 | 2368 | 0 | 3141 | 1309 | 53% | ✓ |
| e5 | easy | 3 | 167 | 2345 | 0 | 2516 | 777 | 65% | ✓ |
| e6 | easy | 2 | 161 | 2350 | 0 | 2514 | 1097 | 33% | ✓ |
| e7 | easy | 2 | 167 | 2348 | 0 | 2518 | 1097 | 66% | ✓ |
| e8 | easy | 2 | 161 | 2352 | 0 | 2516 | 1785 | 84% | ✓ |
| e9 | easy | 2 | 164 | 2348 | 0 | 2516 | 2259 | 90% | ✓ |
| e10 | easy | 3 | 161 | 2348 | 0 | 2513 | 786 | 75% | ✓ |
| h1 | hard | 5 | 157 | 2353 | 0 | 2515 | 922 | 50% | ✓ |
| h2 | hard | 2 | 166 | 2347 | 0 | 2516 | 564 | 60% | ✓ |
| h3 | hard | 3 | 163 | 2347 | 0 | 2513 | 777 | 8% | ✓ |
| h4 | hard | 5 | 181 | 2349 | 0 | 2536 | 802 | 39% | ✓ |
| h5 | hard | 3 | 180 | 2349 | 0 | 2532 | 1309 | 44% | ✓ |
| h6 | hard | 5 | 149 | 2351 | 0 | 2506 | 1309 | 30% | ✓ |
| h7 | hard | 7 | 151 | 2348 | 0 | 2507 | 13199 | 81% | ✓ |
| h8 | hard | 3 | 164 | 2348 | 0 | 2516 | 1027 | 58% | ✓ |
| h9 | hard | 6 | 159 | 2346 | 0 | 2511 | 860 | 46% | ✓ |
| h10 | hard | 7 | 199 | 2348 | 0 | 2555 | 2259 | 57% | ✓ |

## 2. Retrieval – RSE OFF (top-k baseline)

**Keyword hit rate:** 100%  avg overlap: 86%

### Latency

| Phase | Mean ms | P50 ms | P95 ms | Max ms |
|---|---|---|---|---|
| bm25 | 0.5 | 0.4 | 0.9 | 0.9 |
| embed_query | 0.2 | 0.2 | 0.3 | 0.3 |
| dense_search | 0.8 | 0.7 | 1.9 | 1.9 |
| rrf | 0.0 | 0.0 | 0.1 | 0.1 |
| rerank | 2350.2 | 2350.5 | 2353.8 | 2353.8 |
| rse | 0.0 | 0.0 | 0.0 | 0.0 |
| total | 2352.2 | 2352.4 | 2355.2 | 2355.2 |

### Context quality

| Metric | Value |
|---|---|
| avg_context_chars | 9278.80 |
| avg_n_returned | 10.00 |
| avg_sections | 7.75 |
| avg_segment_chars | 927.88 |
| avg_keyword_overlap_pct | 0.86 |

### Per-query breakdown

| Q | Diff | BM25 | Dense | Rerank | RSE | Total | Ctx chars | kw% | Hit |
|---|---|---|---|---|---|---|---|---|---|
| e1 | easy | 0 | 2 | 2349 | 0 | 2352 | 7725 | 89% | ✓ |
| e2 | easy | 0 | 1 | 2350 | 0 | 2352 | 10376 | 100% | ✓ |
| e3 | easy | 0 | 1 | 2354 | 0 | 2355 | 9947 | 98% | ✓ |
| e4 | easy | 1 | 1 | 2350 | 0 | 2353 | 9415 | 88% | ✓ |
| e5 | easy | 0 | 1 | 2349 | 0 | 2351 | 8285 | 85% | ✓ |
| e6 | easy | 0 | 2 | 2350 | 0 | 2352 | 8859 | 95% | ✓ |
| e7 | easy | 0 | 1 | 2350 | 0 | 2352 | 8859 | 80% | ✓ |
| e8 | easy | 0 | 1 | 2350 | 0 | 2352 | 10003 | 94% | ✓ |
| e9 | easy | 0 | 1 | 2353 | 0 | 2355 | 9420 | 94% | ✓ |
| e10 | easy | 1 | 1 | 2351 | 0 | 2353 | 9184 | 96% | ✓ |
| h1 | hard | 0 | 0 | 2344 | 0 | 2345 | 8792 | 78% | ✓ |
| h2 | hard | 1 | 1 | 2351 | 0 | 2353 | 8784 | 78% | ✓ |
| h3 | hard | 0 | 1 | 2352 | 0 | 2353 | 9956 | 58% | ✓ |
| h4 | hard | 1 | 1 | 2350 | 0 | 2352 | 8954 | 78% | ✓ |
| h5 | hard | 0 | 1 | 2351 | 0 | 2353 | 9311 | 84% | ✓ |
| h6 | hard | 0 | 1 | 2349 | 0 | 2351 | 8881 | 91% | ✓ |
| h7 | hard | 1 | 1 | 2351 | 0 | 2353 | 9983 | 78% | ✓ |
| h8 | hard | 1 | 2 | 2351 | 0 | 2354 | 9987 | 83% | ✓ |
| h9 | hard | 1 | 1 | 2350 | 0 | 2352 | 10269 | 98% | ✓ |
| h10 | hard | 1 | 1 | 2350 | 0 | 2352 | 8586 | 67% | ✓ |

## 3. RSE Impact (delta)

| Metric | No RSE | With RSE | Delta |
|---|---|---|---|
| Keyword hit rate | 100% | 100% | +0% |
| Avg context chars | 9279 | 1821 | -7458 |
| Avg kw overlap | 86% | 61% | -25% |
| Avg total latency ms | 2352.2 | 3197.1 | +844.9 |
