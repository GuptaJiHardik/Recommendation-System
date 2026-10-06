# Phase 5 candidate analysis

## Reproduce

```text
uv run python -B -m src.rank_dataset build --db 'data\recommendations.db' --seed 42 --negative-ratio 3 --output-dir 'data\rank' --report 'reports\candidate_analysis.md'
```

History is strictly before each UTC cutoff; labels and exposure provenance use the following seven days, including the cutoff and excluding the outcome end. Models are refitted at every snapshot. Visitors in exported datasets have pre-cutoff events; visitors without future positives remain eligible for negative sampling. Validation is unsampled; final test is reserved.

The pool contains up to 20 collaborative, 20 content, and 10 popularity nominations, deduplicated by item ID. Rows are ordered by item ID, not recommendation quality. Missing source scores are 0.0 with flag 0; nominated zero scores have flag 1. Cold start uses only ten popularity nominations.

## Union retrieval

Recall uses recommendable future carts/purchases, excluding unavailable items and prior purchases. Macro recall averages eligible visitors with positives; micro recall divides retrieved positive pairs by all relevant pairs. Empty cohorts are unavailable.

| Role / cutoff | Visitors / evaluated | Pool min / mean / max | Positive pairs | Macro Recall@50 | Micro recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| train / 2025-02-17 | 43 / 7 | 29 / 35.58 / 42 | 5 / 7 | 0.714286 | 0.714286 |
| train / 2025-02-24 | 44 / 15 | 30 / 36.27 / 44 | 16 / 20 | 0.766667 | 0.800000 |
| train / 2025-03-03 | 44 / 13 | 32 / 35.73 / 41 | 17 / 19 | 0.884615 | 0.894737 |
| train / 2025-03-10 | 45 / 12 | 30 / 35.36 / 43 | 11 / 14 | 0.750000 | 0.785714 |
| validation / 2025-03-17 | 49 / 12 | 28 / 35.59 / 42 | 13 / 16 | 0.847222 | 0.812500 |

**Retrieval ceiling:** the ranker cannot rescue positives omitted from this pool. Phases 3/4 used 50 candidates per individual source; their earlier 100% validation recall is not a like-for-like comparison with this smaller budget. Inspect these misses before Phase 6; this phase does not tune generators or increase budgets.

### Missed recommendable positives

- 2025-02-17: `visitor_0027 / item_0016` (personalized), `visitor_0029 / item_0068` (personalized).
- 2025-02-24: `visitor_0007 / item_0035` (personalized), `visitor_0008 / item_0041` (personalized), `visitor_0034 / item_0085` (personalized), `visitor_0039 / item_0025` (personalized).
- 2025-03-03: `visitor_0017 / item_0025` (personalized), `visitor_0033 / item_0004` (personalized).
- 2025-03-10: `visitor_0019 / item_0024` (personalized), `visitor_0024 / item_0055` (personalized), `visitor_0041 / item_0049` (personalized), `visitor_0047 / item_0009` (cold_start), `visitor_0047 / item_0030` (cold_start), `visitor_0049 / item_0079` (cold_start), `visitor_0050 / item_0013` (cold_start).
- 2025-03-17: `visitor_0008 / item_0061` (personalized), `visitor_0015 / item_0093` (personalized), `visitor_0041 / item_0029` (personalized).

## Source contributions and overlap

All nomination/overlap counts below cover every historical visitor. Exclusive candidates have exactly one source; pairwise intersections include three-source candidates.

| Cutoff | Source | Macro Recall@50 | Nominations | Exclusive candidates | Positive nominations | Exclusive positives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2025-02-17 | collaborative | 0.571429 | 860 | 359 | 4 | 1 |
| 2025-02-17 | content | 0.571429 | 860 | 363 | 4 | 0 |
| 2025-02-17 | popularity | 0.142857 | 430 | 261 | 1 | 0 |
| 2025-02-24 | collaborative | 0.433333 | 880 | 390 | 10 | 3 |
| 2025-02-24 | content | 0.400000 | 880 | 402 | 9 | 4 |
| 2025-02-24 | popularity | 0.233333 | 440 | 271 | 4 | 2 |
| 2025-03-03 | collaborative | 0.576923 | 880 | 374 | 11 | 0 |
| 2025-03-03 | content | 0.846154 | 880 | 385 | 16 | 5 |
| 2025-03-03 | popularity | 0.153846 | 440 | 265 | 4 | 0 |
| 2025-03-10 | collaborative | 0.541667 | 900 | 359 | 8 | 2 |
| 2025-03-10 | content | 0.583333 | 900 | 391 | 8 | 3 |
| 2025-03-10 | popularity | 0.125000 | 450 | 269 | 2 | 0 |
| 2025-03-17 | collaborative | 0.819444 | 980 | 393 | 12 | 2 |
| 2025-03-17 | content | 0.638889 | 980 | 429 | 10 | 1 |
| 2025-03-17 | popularity | 0.166667 | 490 | 306 | 2 | 0 |

| Cutoff | Collab + content | Collab + popularity | Content + popularity |
| --- | ---: | ---: | ---: |
| 2025-02-17 | 451 | 123 | 119 |
| 2025-02-24 | 435 | 126 | 114 |
| 2025-03-03 | 453 | 133 | 122 |
| 2025-03-10 | 478 | 150 | 118 |
| 2025-03-17 | 522 | 155 | 119 |

### Cold start

- 2025-02-17: 1 observed visitors, 0 with recommendable positives; macro recall unavailable.
- 2025-02-24: 0 observed visitors, 0 with recommendable positives; macro recall unavailable.
- 2025-03-03: 1 observed visitors, 0 with recommendable positives; macro recall unavailable.
- 2025-03-10: 4 observed visitors, 3 with recommendable positives; macro recall 0.000000.
- 2025-03-17: 1 observed visitors, 0 with recommendable positives; macro recall unavailable.

## Training sampling

Seed: **42**. Requested negatives per positive: **3** per training snapshot. Keep all retrieved positives; sample exposed negatives first, then unobserved proxies. Exposure means a visitor/item impression in the outcome window; a view-only outcome still has label 0. A negative indicates no observed target action, not dislike. Sampled prevalence is not representative exposure prevalence; future model scores are ordering signals rather than calibrated probabilities.

| Cutoff | Available negatives / exposed | Kept positives | Sampled negatives / exposed / proxy | Achieved ratio | Shortfall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-02-17 | 1525 / 121 | 5 | 15 / 15 / 0 | 3.000000 | 0 |
| 2025-02-24 | 1580 / 232 | 16 | 48 / 48 / 0 | 3.000000 | 0 |
| 2025-03-03 | 1555 / 191 | 17 | 51 / 51 / 0 | 3.000000 | 0 |
| 2025-03-10 | 1580 / 196 | 11 | 33 / 33 / 0 | 3.000000 | 0 |

Exported rows: **196 training**, **1744 validation**.

## Feature and label trace

Visitor `visitor_0005`, item `item_0031`, snapshot **2025-02-17T00:00:00+00:00**.

| Feature | Value | Information time |
| --- | ---: | --- |
| collab_score | 4.0849685259 | Events/models strictly before cutoff |
| content_score | 0.4981164125 | Events/models strictly before cutoff |
| popularity_score | 0.0 | Events/models strictly before cutoff |
| from_collab | 1 | Events/models strictly before cutoff |
| from_content | 1 | Events/models strictly before cutoff |
| from_popularity | 0 | Events/models strictly before cutoff |
| visitor_view_count | 18 | Events/models strictly before cutoff |
| visitor_cart_count | 4 | Events/models strictly before cutoff |
| visitor_purchase_count | 1 | Events/models strictly before cutoff |
| days_since_last_activity | 1.5459375 | Events/models strictly before cutoff |
| visitor_category_affinity | 0.3714285714 | Events/models strictly before cutoff |
| item_interaction_count | 4 | Events/models strictly before cutoff |
| item_price | 84.41 | Catalog available at cutoff (static metadata) |
| visitor_item_price_gap | 28.8717142857 | Events/models strictly before cutoff; static available prices |

Last visitor activity: **2025-02-15T10:53:51+00:00**. Category `electronics` strength is 13 / 35; weighted mean historical price is 55.538286. Item creation: 2025-01-06T00:00:00+00:00.

Label: **1**, from target actions in **[2025-02-17T00:00:00+00:00, 2025-02-24T00:00:00+00:00)**. Target timestamps: 2025-02-18T16:16:05+00:00. Exposure provenance: `positive`; impression timestamps: 2025-02-18T16:15:00+00:00.

IDs, role, snapshot, outcome end, label, exposure flag, and provenance are metadata, excluded from FEATURE_COLUMNS. Recency and price gap are missing for empty histories; Phase 6 will impute and scale features. Catalog attributes are assumed static because this dataset has no price/category revision history. Synthetic exposure is incomplete evidence and does not establish real-world preference or business uplift.

**Checkpoint:** source flags describe nomination, not comparable score scales. Features describe the past; labels describe the following seven days. Improving ranking only reorders retrieved items, so missing relevant candidates require retrieval changes.
