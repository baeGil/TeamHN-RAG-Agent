# Pipeline Benchmark Report

**Parser:** MINERU  **Doc summary (CCH):** True  **RSE:** True

## 1. Indexing

| Phase | Time (ms) |
|---|---|
| PDF Parsing (mineru) | 16634 |
| Chunking | 0 |
| Doc Summary LLM (CCH) | 1659 |
| Embedding (22 chunks) | 1024 |
| Index build (BM25 + TurboVec) | 3571 |
| **Total ingest** | **25608** |

| Metric | Value |
|---|---|
| Pages | 8 |
| Blocks (parser output) | 22 |
| Chunks (after splitting) | 22 |
| Sections detected | 15 |
| Avg chunk size (chars) | 725 |
| Min / Max chunk size | 19 / 1775 |

**Cost:**
- Embedding: ~6,157 tokens → **$0.0001**
- Doc summary: 6233 in + 40 out → **$0.00096**
- **Total indexing cost: $0.0023**

**Generated doc summary (CCH header):**
> Tài liệu này trình bày về phương pháp Weighted Potential Fast Marching Firework để quy hoạch đường đi an toàn đa đích cho robot di động trong môi trường nhà máy hạt nhân.

## 2. Retrieval – RSE ON

**Keyword hit rate:** 100%  avg overlap: 59%

### Latency

| Phase | Mean ms | P50 ms | P95 ms | Max ms |
|---|---|---|---|---|
| bm25 | 4.1 | 4.0 | 7.8 | 7.8 |
| embed_query | 213.3 | 169.5 | 797.5 | 797.5 |
| dense_search | 11.6 | 0.6 | 216.5 | 216.5 |
| rrf | 0.1 | 0.0 | 0.3 | 0.3 |
| rerank | 2956.7 | 2345.4 | 14578.1 | 14578.1 |
| rse | 0.5 | 0.2 | 5.1 | 5.1 |
| total | 3186.8 | 2520.2 | 15599.5 | 15599.5 |

### Context quality

| Metric | Value |
|---|---|
| avg_context_chars | 1231.30 |
| avg_n_returned | 1.00 |
| avg_sections | 0.60 |
| avg_segment_chars | 1231.30 |
| avg_keyword_overlap_pct | 0.59 |

### Per-query breakdown

| Q | Diff | BM25 | Dense | Rerank | RSE | Total | Ctx chars | kw% | Hit |
|---|---|---|---|---|---|---|---|---|---|
| e1 | easy | 2 | 1014 | 14578 | 5 | 15600 | 1432 | 84% | ✓ |
| e2 | easy | 4 | 265 | 2347 | 0 | 2618 | 1432 | 89% | ✓ |
| e3 | easy | 3 | 164 | 2339 | 0 | 2507 | 1432 | 98% | ✓ |
| e4 | easy | 3 | 162 | 2341 | 0 | 2506 | 1307 | 53% | ✓ |
| e5 | easy | 3 | 176 | 2342 | 0 | 2522 | 777 | 65% | ✓ |
| e6 | easy | 2 | 159 | 2347 | 0 | 2509 | 1097 | 33% | ✓ |
| e7 | easy | 3 | 163 | 2344 | 0 | 2511 | 1097 | 66% | ✓ |
| e8 | easy | 2 | 180 | 2345 | 0 | 2528 | 1783 | 84% | ✓ |
| e9 | easy | 3 | 165 | 2348 | 0 | 2517 | 2259 | 90% | ✓ |
| e10 | easy | 4 | 170 | 2345 | 0 | 2520 | 786 | 75% | ✓ |
| h1 | hard | 6 | 166 | 2343 | 0 | 2516 | 922 | 50% | ✓ |
| h2 | hard | 7 | 230 | 2345 | 0 | 2583 | 564 | 60% | ✓ |
| h3 | hard | 4 | 166 | 2348 | 0 | 2519 | 777 | 8% | ✓ |
| h4 | hard | 5 | 174 | 2345 | 1 | 2526 | 802 | 39% | ✓ |
| h5 | hard | 5 | 192 | 2349 | 0 | 2547 | 1307 | 44% | ✓ |
| h6 | hard | 5 | 167 | 2343 | 0 | 2516 | 1307 | 30% | ✓ |
| h7 | hard | 6 | 238 | 2346 | 0 | 2590 | 1307 | 43% | ✓ |
| h8 | hard | 2 | 156 | 2349 | 0 | 2508 | 1027 | 58% | ✓ |
| h9 | hard | 8 | 156 | 2346 | 0 | 2510 | 952 | 46% | ✓ |
| h10 | hard | 6 | 236 | 2342 | 0 | 2585 | 2259 | 57% | ✓ |

## 2. Retrieval – RSE OFF (top-k baseline)

**Keyword hit rate:** 100%  avg overlap: 86%

### Latency

| Phase | Mean ms | P50 ms | P95 ms | Max ms |
|---|---|---|---|---|
| bm25 | 0.5 | 0.4 | 0.9 | 0.9 |
| embed_query | 0.2 | 0.2 | 0.4 | 0.4 |
| dense_search | 0.9 | 0.8 | 1.5 | 1.5 |
| rrf | 0.1 | 0.1 | 0.1 | 0.1 |
| rerank | 2346.8 | 2347.2 | 2352.6 | 2352.6 |
| rse | 0.0 | 0.0 | 0.0 | 0.0 |
| total | 2348.9 | 2349.7 | 2354.5 | 2354.5 |

### Context quality

| Metric | Value |
|---|---|
| avg_context_chars | 9431.90 |
| avg_n_returned | 10.00 |
| avg_sections | 7.70 |
| avg_segment_chars | 943.19 |
| avg_keyword_overlap_pct | 0.86 |

### Per-query breakdown

| Q | Diff | BM25 | Dense | Rerank | RSE | Total | Ctx chars | kw% | Hit |
|---|---|---|---|---|---|---|---|---|---|
| e1 | easy | 0 | 1 | 2353 | 0 | 2355 | 7791 | 89% | ✓ |
| e2 | easy | 0 | 1 | 2347 | 0 | 2349 | 10639 | 100% | ✓ |
| e3 | easy | 0 | 1 | 2346 | 0 | 2348 | 10059 | 98% | ✓ |
| e4 | easy | 0 | 1 | 2345 | 0 | 2347 | 10571 | 88% | ✓ |
| e5 | easy | 0 | 1 | 2348 | 0 | 2350 | 8966 | 85% | ✓ |
| e6 | easy | 0 | 2 | 2347 | 0 | 2350 | 8879 | 95% | ✓ |
| e7 | easy | 0 | 1 | 2345 | 0 | 2347 | 8879 | 80% | ✓ |
| e8 | easy | 0 | 2 | 2348 | 0 | 2351 | 9911 | 94% | ✓ |
| e9 | easy | 0 | 1 | 2347 | 0 | 2349 | 9680 | 94% | ✓ |
| e10 | easy | 0 | 1 | 2344 | 0 | 2346 | 9291 | 96% | ✓ |
| h1 | hard | 1 | 1 | 2349 | 0 | 2351 | 8812 | 78% | ✓ |
| h2 | hard | 1 | 1 | 2347 | 0 | 2351 | 8692 | 78% | ✓ |
| h3 | hard | 0 | 1 | 2348 | 0 | 2350 | 9869 | 58% | ✓ |
| h4 | hard | 0 | 0 | 2342 | 0 | 2343 | 8489 | 78% | ✓ |
| h5 | hard | 1 | 1 | 2347 | 0 | 2349 | 9331 | 84% | ✓ |
| h6 | hard | 1 | 2 | 2350 | 0 | 2353 | 9560 | 91% | ✓ |
| h7 | hard | 0 | 1 | 2338 | 0 | 2340 | 9926 | 79% | ✓ |
| h8 | hard | 1 | 1 | 2347 | 0 | 2349 | 10160 | 83% | ✓ |
| h9 | hard | 1 | 1 | 2350 | 0 | 2352 | 10269 | 98% | ✓ |
| h10 | hard | 1 | 1 | 2347 | 0 | 2350 | 8864 | 67% | ✓ |

## 3. RSE Impact (delta)

| Metric | No RSE | With RSE | Delta |
|---|---|---|---|
| Keyword hit rate | 100% | 100% | +0% |
| Avg context chars | 9432 | 1231 | -8201 |
| Avg kw overlap | 86% | 59% | -27% |
| Avg total latency ms | 2348.9 | 3186.8 | +837.9 |
