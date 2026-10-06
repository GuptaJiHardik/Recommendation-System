# Phase 6 model comparison

## Reproduce

```text
uv run python -B -m src.train
```

Explicit configured model: **logistic**. Seed: 42; negatives per positive: 3.

Training: **196 rows / 49 positives**. Validation: **1744 unsampled rows / 13 positives**.

Training cutoffs: 2025-02-17, 2025-02-24, 2025-03-03, 2025-03-10.
Validation history is strictly before **2025-03-17T00:00:00+00:00**; target outcomes are in [2025-03-17T00:00:00+00:00, 2025-03-24T00:00:00+00:00). Final test remains reserved.

## Validation tuning

Imputation, scaling, and coefficients use training rows only. Selection uses MAP@10, then Recall@10, smaller C, then no class weighting. The saved pipeline is not refitted on validation labels.

| C | Class weight | Status | MAP@10 | Recall@10 |
| ---: | --- | --- | ---: | ---: |
| 0.01 | none | ok | 0.164616 | 0.527778 |
| 0.01 | balanced | ok | 0.156928 | 0.527778 |
| 0.1 | none | ok | 0.110185 | 0.527778 |
| 0.1 | balanced | ok | 0.083840 | 0.486111 |
| 1.0 | none | ok | 0.095172 | 0.486111 |
| 1.0 | balanced | ok | 0.092708 | 0.486111 |
| 10.0 | none | ok | 0.095172 | 0.486111 |
| 10.0 | balanced | ok | 0.089749 | 0.486111 |

Selected logistic configuration: `{'C': 0.01, 'class_weight': None}`.

## Same-pool comparisons

All methods reorder the same deduplicated 20 collaborative / 20 content / 10 popularity union. Purchased and unavailable items are excluded; no category cap is applied. Individual-source baselines use actual scores for every union item, not missing-nomination zeros. The logistic inputs retain Phase 5 nomination scores and source flags.

Blend: 0.4 collaborative + 0.4 content + 0.2 popularity, each source min–max normalized within the visitor's fixed pool. Constant sources contribute zero. Ties use item ID.

### Personalized

| Method | Visitors / evaluated | Candidate Recall@50 | Precision@10 | Recall@10 | MAP@10 | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| popularity | 49 / 12 | 0.847222 | 0.016667 | 0.166667 | 0.069444 | 0.110000 |
| collaborative | 49 / 12 | 0.847222 | 0.075000 | 0.652778 | 0.181812 | 0.650000 |
| content | 49 / 12 | 0.847222 | 0.050000 | 0.388889 | 0.102778 | 0.710000 |
| blend | 49 / 12 | 0.847222 | 0.066667 | 0.611111 | 0.218849 | 0.630000 |
| logistic | 49 / 12 | 0.847222 | 0.066667 | 0.527778 | 0.164616 | 0.640000 |

### Cold Start

| Method | Visitors / evaluated | Candidate Recall@50 | Precision@10 | Recall@10 | MAP@10 | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| popularity | 1 / 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| collaborative | 1 / 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| content | 1 / 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| blend | 1 / 0 | unavailable | unavailable | unavailable | unavailable | unavailable |
| logistic | 1 / 0 | unavailable | unavailable | unavailable | unavailable | unavailable |

Cold start always uses popularity because training rows have prior history. Only visitors with recommendable held-out carts/purchases enter quality averages; unavailable cohorts are not scored as zero.

### Scoring latency

Prepared feature/source-score rows; one warm-up and 20 repetitions per historical visitor. Times include scoring only, excluding sorting, retrieval, features, fitting, and metrics. Source baselines read prepared scores; this is not whole-request or HTTP latency.

| Method | Samples | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| popularity | 980 | 0.002600 | 0.006100 |
| collaborative | 980 | 0.002700 | 0.005500 |
| content | 980 | 0.002800 | 0.005700 |
| blend | 980 | 0.036500 | 0.074755 |
| logistic | 980 | 1.957100 | 3.172675 |

### Logistic differences from simpler methods

| Compared with | Δ Precision@10 | Δ Recall@10 | Δ MAP@10 | Δ Coverage |
| --- | ---: | ---: | ---: | ---: |
| popularity | 0.050000 | 0.361111 | 0.095172 | 0.530000 |
| collaborative | -0.008333 | -0.125000 | -0.017196 | -0.010000 |
| content | 0.016667 | 0.138889 | 0.061839 | -0.070000 |
| blend | 0.000000 | -0.083333 | -0.054233 | 0.010000 |

## Coefficients and learning checkpoint

Coefficients change log-odds per one training standard deviation of an imputed feature, holding other inputs fixed. Positive coefficients raise the score and negative coefficients lower it; correlated features and regularization prevent causal interpretations.

| Feature | Standardized coefficient | Interpretation |
| --- | ---: | --- |
| collab_score | 0.031092 | Stronger collaborative nomination raises the conditional ordering score. |
| content_score | 0.062360 | Stronger metadata similarity nomination raises the conditional ordering score. |
| popularity_score | -0.049463 | Increasing this feature lowers the conditional ordering score. |
| from_collab | 0.016239 | Increasing this feature raises the conditional ordering score. |
| from_content | 0.021823 | Increasing this feature raises the conditional ordering score. |
| from_popularity | -0.034880 | Increasing this feature lowers the conditional ordering score. |
| visitor_view_count | 0.053332 | Increasing this feature raises the conditional ordering score. |
| visitor_cart_count | -0.019747 | Increasing this feature lowers the conditional ordering score. |
| visitor_purchase_count | -0.008607 | Increasing this feature lowers the conditional ordering score. |
| days_since_last_activity | 0.028010 | Increasing this feature raises the conditional ordering score. |
| visitor_category_affinity | 0.007992 | Greater historical affinity to the item's category raises the conditional ordering score. |
| item_interaction_count | -0.033657 | Increasing this feature lowers the conditional ordering score. |
| item_price | -0.075493 | Increasing this feature lowers the conditional ordering score. |
| visitor_item_price_gap | -0.047337 | Increasing this feature lowers the conditional ordering score. |

Measured result: logistic MAP@10 (0.164616) is below the best simpler method, blend (0.218849). Model choice remains explicit.

## Trace one ranking

Visitor `visitor_0003`; relevant items: item_0034.

| Method | Top 10 in order |
| --- | --- |
| popularity | item_0008, item_0002, item_0003, item_0007, item_0012, item_0078, item_0011, item_0009, item_0047, item_0018 |
| collaborative | item_0044, item_0004, item_0019, item_0090, item_0009, item_0034, item_0069, item_0054, item_0029, item_0076 |
| content | item_0069, item_0029, item_0019, item_0039, item_0059, item_0024, item_0089, item_0004, item_0034, item_0014 |
| blend | item_0004, item_0019, item_0069, item_0009, item_0034, item_0044, item_0029, item_0024, item_0054, item_0039 |
| logistic | item_0069, item_0029, item_0019, item_0014, item_0024, item_0089, item_0004, item_0059, item_0034, item_0039 |

Configured model's first item: `item_0069`, score **0.310370**.

| Feature | Pre-cutoff value |
| --- | ---: |
| collab_score | 4.723511 |
| content_score | 0.904209 |
| popularity_score | 0.000000 |
| from_collab | 1.000000 |
| from_content | 1.000000 |
| from_popularity | 0.000000 |
| visitor_view_count | 10.000000 |
| visitor_cart_count | 3.000000 |
| visitor_purchase_count | 0.000000 |
| days_since_last_activity | 0.343079 |
| visitor_category_affinity | 0.894737 |
| item_interaction_count | 9.000000 |
| item_price | 26.940000 |
| visitor_item_price_gap | 8.277895 |

After training-fitted imputation/scaling, intercept + coefficient contributions = **-0.798390**; applying the logistic sigmoid gives the score above.

## Assumptions and limitations

Label 1 means an observed cart or purchase in the following seven days; absence of an action is an offline proxy, not proof of dislike. Exposure-first 3:1 sampling changes prevalence; class weighting is compared on unsampled validation, with no inverse-sampling correction. Scores are not calibrated purchase probabilities. Logistic regression fits this binary target with regularization; ridge regression's squared-error numeric objective is a weaker first choice.

IDs, labels, snapshot times, and future exposure provenance are excluded from model inputs. Feature/generator events are strictly before each cutoff; catalog metadata is static. The ranker cannot recover the missed relevant items outside its candidate pool. The small synthetic validation cohort and eight tuning choices limit confidence; offline improvements do not prove business uplift.

The bundle contains the full preprocessing pipeline and schema, bounded collaborative serving state, fitted content representation, and training provenance. Its generators use the validation cutoff; it is not a later serving refit. Raw visitor histories are read separately. Only trusted local pickle bundles should be loaded.

Versions: `{"numpy": "2.5.3", "pandas": "2.3.3", "python": "3.13.15", "scipy": "1.18.1", "sklearn": "1.9.1"}`.
