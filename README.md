# Recommendation MVP — Phases 0–4

Phase 0 creates a small exposure-linked e-commerce dataset for learning and inspection. Generated CSVs live in `data/synthetic/` and are ignored by Git. Phase 1 loads them into SQLite and recommends products by popularity. Phase 2 establishes chronological evaluation; Phase 3 adds item-to-item collaborative retrieval. Phase 4 adds metadata-based content retrieval. Phase 5 and later in `RECOMMENDATION_MVP_SPEC.md` are not implemented yet.

## Run

Requires Python 3.11 or newer; Phase 0 has no third-party dependencies. From the repository root:

```text
python src/generate_synthetic.py
python src/explore_data.py data/synthetic
```

If using `uv`, substitute `uv run --no-project --no-cache python` for `python`. The generator defaults to seed `42` and `data/synthetic/`; use `--seed` and `--output-dir` to change them. The profiler accepts any directory with the three CSV interfaces.

`src/generate_synthetic.py` writes products, impressions, and optional linked view, cart, and purchase actions. It simulates 12 weeks from 2025-01-06, 50 visitors with one or two preferred categories, varied visit counts, shared popular products, occasional exploration, five late visitors, and 15 products introduced in later weeks. A small item tail receives little exposure. Timestamps are UTC.

## Observed output (seed 42)

| Measure | Result |
| --- | ---: |
| Products / categories | 100 / 5 (20 per category) |
| Impressions / visitors with events | 3,763 / 50 |
| Views / carts / purchases | 1,030 / 228 / 65 |
| Impressions with any action | 1,030 (27.4%) |
| Impressions without action | 2,733 |
| Events per visitor | 5 minimum, 25.5 median, 51 maximum |
| Items with zero events | 4 |
| Final 21 days: carts / purchases | 58 / 11 |
| Final 7 days: carts / purchases | 24 / 4 |

The most active item has 46 events; several items have none or one. The profiler reports zero missing values and passes ID, reference, action-order, and creation-time checks. I inspected eight raw rows from each CSV: item prices and descriptions are simple but plausible; impression positions increase within a session; observed events point to matching impressions. A second run with seed `42` produced byte-identical hashes for all three files.

## Phase 0 checkpoint

- **One event row:** an action by one visitor on one item at a UTC time, in a session, linked to the impression that showed that item. A cart and purchase following a view are separate rows tied to the same impression.
- **View versus purchase:** a view may mean curiosity or accidental attention. A purchase requires stronger intent and commitment, so later models may assign it more implicit-feedback weight. Neither action is a numeric rating.
- **Sparsity and cold start:** most visitor–item pairs have no event, and 2,733 displayed items get no action. A new visitor has little or no history; a newly created item can have no interactions even though its category, price, and text are available.

**Limitation:** all behavior comes from deliberately simplified probabilities. Its exposure log and preferences are useful for testing a pipeline, but they cannot establish real shopper behavior or real-world recommendation quality. An unclicked impression is an uncertain negative, not proof of dislike.

## Phase 1 — implemented

Added strict CSV validation and deduplication, atomic SQLite import/read, six chronological snapshots, and weighted popularity recommendations with purchase filtering and unknown-visitor fallback.

Run from the project root (Pandas and pytest are declared in `pyproject.toml`):

```text
uv run --no-project --with "pandas>=2.2,<3" python -m src.data
uv run --no-project --with "pandas>=2.2,<3" python -m src.split
uv run --no-project --with "pandas>=2.2,<3" python -m src.popularity visitor_0001 visitor_0032 visitor_0050 visitor_unknown --as-of 2025-03-17T00:00:00Z --k 10
uv run --no-project --with "pandas>=2.2,<3" --with "pytest>=8,<10" python -m pytest tests/test_data.py -q
```

`src.data` validates IDs, UTC timestamps, prices, types, and product/impression references before writing `data/recommendations.db`. Exact duplicates are removed; other invalid rows fail with diagnostics and preserve the existing database. Imported: **100 items, 1,323 events, 3,763 impressions; zero duplicates**.

Snapshots use expanding history `< cutoff` and outcomes `[cutoff, cutoff + 7 days)`. All dates below are UTC midnight; the declared observation end is March 31, 2025. `src.split` also prints visitor, impression, and event-type counts.

| Role | Cutoff | Outcome end | History events | Outcome events |
| --- | --- | --- | ---: | ---: |
| Train | 2025-02-17 | 2025-02-24 | 621 | 83 |
| Train | 2025-02-24 | 2025-03-03 | 704 | 134 |
| Train | 2025-03-03 | 2025-03-10 | 838 | 112 |
| Train | 2025-03-10 | 2025-03-17 | 950 | 124 |
| Validation | 2025-03-17 | 2025-03-24 | 1,074 | 108 |
| Test | 2025-03-24 | 2025-03-31 | 1,182 | 141 |

Popularity sums pre-cutoff views/carts/purchases with weights **1/3/5**; ties use item ID. Future products are excluded, and eligible zero-score items fill short lists. Missing creation metadata means no known introduction cutoff. The four demo visitors each received 10 items with no prior purchases included:

| Visitor | First three item IDs |
| --- | --- |
| visitor_0001 | item_0008, item_0002, item_0003 |
| visitor_0032 | item_0002, item_0003, item_0007 |
| visitor_0050 | item_0008, item_0002, item_0003 |
| visitor_unknown | item_0008, item_0002, item_0003 |

**Observation/checkpoint:** `visitor_0032` purchased `item_0008`, so it is excluded. Random splits can expose future behavior; popularity provides a simple comparison baseline, and unknown visitors receive the global order. **Limitation:** these provisional weights favor popular items rather than personal preferences; quality metrics begin in Phase 2.

**Checks:** 41 tests pass; re-import preserves counts, all six windows match the table, and all four demo lists pass purchase-exclusion checks.

## Phase 2 — implemented

`src/evaluate.py` evaluates retrieval and ranking separately, with hand-calculated metric tests and the protocol/results in [reports/baseline.md](reports/baseline.md). Relevance is a distinct held-out cart or purchase on an eligible item, excluding prior purchases. Personalized averages require history and a recommendable positive; cold start is reported separately. Phase 2 had 77 passing tests. The final test window remains reserved.

## Phase 3 — implemented

`src/collaborative.py` sums pre-cutoff event weights (view=1, cart=3, purchase=5) into a float64 sparse visitor–item matrix. Item vectors are columns of that matrix; cosine similarity is their dot product divided by their lengths. Self-similarities are removed, and each source item retains at most 20 positive-similarity neighbors. Each candidate score sums `interaction_strength × similarity` over the visitor's source items. Scores are ordering signals, not probabilities.

Previously purchased items contribute evidence but cannot be recommended. Viewed/carted items remain eligible. Known visitors receive only positive-score collaborative candidates; short lists are not filled with popularity. Empty histories use cached popularity order, including eligible zero-score items. Score ties use item ID. Products created after the snapshot and events at/after it are excluded; missing creation metadata means no known introduction cutoff.

Run from the project root after importing the database:

```text
uv run --no-project --with "pandas>=2.2,<3" --with "numpy>=1.26,<3" --with "scipy>=1.12,<2" python -B -m src.collaborative recommend visitor_0001 visitor_0032 visitor_0050 visitor_unknown --as-of 2025-03-17T00:00:00Z --k 10
uv run --no-project --with "pandas>=2.2,<3" --with "numpy>=1.26,<3" --with "scipy>=1.12,<2" python -B -m src.collaborative recommend visitor_0045 --as-of 2025-03-17T00:00:00Z --k 50
uv run --no-project --with "pandas>=2.2,<3" --with "numpy>=1.26,<3" --with "scipy>=1.12,<2" python -B -m src.collaborative evaluate
uv run --no-project --with "pandas>=2.2,<3" --with "numpy>=1.26,<3" --with "scipy>=1.12,<2" --with "pytest>=8,<10" python -B -m pytest tests -q -p no:cacheprovider
```

Dependencies are also declared in `pyproject.toml`, so `uv run python -m src.collaborative ...` works with a project environment. Both subcommands accept `--db` and `--neighbors-per-item`. Evaluation always uses the existing validation snapshot and prints per-visitor candidates, metrics, and differences as JSON; it does not evaluate the final test snapshot. No weights or neighbor counts were tuned.

### Validation comparison

Measured on seed 42 with history strictly before **2025-03-17 UTC** and outcomes in **[2025-03-17, 2025-03-24)**, using the unchanged evaluator and full available 100-item catalog. There are 49 known visitors, 12 with recommendable positives (16 positive pairs); the remaining 37 are excluded from averages. The one observed cold-start visitor has no relevant outcome, so cold-start quality metrics are unavailable.

| Measure | Popularity | Collaborative |
| --- | ---: | ---: |
| Candidate Recall@50 | 0.652778 | 1.000000 |
| Precision@10 | 0.016667 | 0.075000 |
| Recall@10 | 0.166667 | 0.652778 |
| MAP@10 | 0.069444 | 0.181812 |
| Catalog coverage | 11% | 65% |
| Median latency (ms) | 2.070350 | 2.252800 |
| p95 latency (ms) | 2.350705 | 2.834590 |

One-time collaborative fitting took **43.226900 ms**. Latency follows Phase 2's warm-up and callback/top-10 timing protocol; fitting and initial data loading are excluded. Popularity queries SQLite while collaborative retrieval uses memory, so these are measurements of the current local paths, not isolated algorithm benchmarks or HTTP latency. Quality is deterministic for a fixed database; timings vary.

All 12 evaluated visitors received 50 candidates. Across all 49 known visitors, candidate counts ranged from 20 to 50 (mean 48.96). `visitor_0045` has one pre-cutoff view of `item_0060`: its 20 candidates start with `item_0055` (0.645497), `item_0093` (0.577350), and `item_0080` (0.445435). `visitor_0032` has 33 events: its first candidates are `item_0038`, `item_0033`, and `item_0001`; its purchased `item_0008` is excluded. Unknown visitors receive popularity, beginning with `item_0008`, `item_0002`, and `item_0003`.

### Trace one recommendation

For `visitor_0001`, the first candidate is `item_0020`, with score **6.095828**. Its full retained-source trace is:

| Source item | Interaction strength | Similarity | Contribution |
| --- | ---: | ---: | ---: |
| item_0040 | 5 | 0.793257 | 3.966284 |
| item_0075 | 4 | 0.338648 | 1.354592 |
| item_0010 | 1 | 0.471782 | 0.471782 |
| item_0005 | 1 | 0.303170 | 0.303170 |

The unrounded contributions sum to the candidate score. The recommendation command prints this trace for each nonempty personalized result; Python callers can use `model.explain(visitor_id, history, item_id)`.

### Artifact and learning checkpoint

Because collaborative candidate Recall@50 strictly exceeded popularity, evaluation atomically saved `artifacts/collaborative_validation.json` (excluded from Git). It contains schema version 1, UTC cutoff, weights, neighbor bound, eligible item IDs, bounded neighbors, and popularity scores. It contains no visitor–item matrix. `CollaborativeModel.load(path)` restores the serving model; callers provide validated pre-cutoff history. Loading rejects unsupported schemas and invalid scores. A non-improving or unavailable metric leaves any existing artifact untouched and reports that decision. This artifact represents the validation cutoff and is not a later serving refit.

The tiny test example has this weighted matrix:

| Visitor | a | b | c | zero |
| --- | ---: | ---: | ---: | ---: |
| u1 | 4 | 1 | 0 | 0 |
| u2 | 1 | 1 | 1 | 0 |

The 4 for `(u1, a)` is a view plus a cart. Cosine(a,b) is `5 / sqrt(34)`; cosine(a,c) is `1 / sqrt(17)`. For u1, c receives `4 / sqrt(17) + 1 / sqrt(2)`. Sparse storage keeps only nonzero strengths. An unobserved item has a zero vector and cannot be retrieved collaboratively; a future content model can use its metadata. Cosine measures shared behavior patterns, not a causal preference or a numeric rating.

**Observation:** collaborative retrieval captured every recommendable validation positive in the candidate pool, but top-10 recall remained 65.28%, leaving room for ranking improvements. **Limitation:** this small, generated cohort has deliberately shared preferences and reflects simulated exposure. It cannot establish real-world quality or business uplift. New items and isolated histories remain weak points; no popularity backfill hides short personalized lists. Similarity fitting processes sparse rows individually, but remains a pairwise computation rather than a large-catalog search index.

**Checks:** 99 tests pass, covering hand-calculated scores, sparse storage, bounds/ties, cutoff leakage, purchase exclusions, SQL fallback parity, CLI/evaluator integration, artifact round trips and atomic failure handling. The saved artifact also matched fresh-model candidates and passed purchase exclusions for all 49 known visitors.

## Phase 4 — implemented

`src/content_based.py` builds sparse item features from metadata and recommends items by cosine similarity to a weighted visitor profile. It exposes `fit_content_based(events, items, as_of_time)`, `model.generate(visitor_id, history, limit)`, and `model.explain(visitor_id, history, item_id)`. Generation returns the existing `item_id`, `score`, and `source` DataFrame interface; personalized rows use `source="content"`.

### Run

From the project root, using the existing imported database:

```text
uv sync
uv run python -B -m src.content_based recommend visitor_0001 visitor_0032 visitor_0050 visitor_0045 visitor_unknown --as-of 2025-03-17T00:00:00Z --k 10
uv run python -B -m src.content_based evaluate
uv run python -B -m pytest tests -q -p no:cacheprovider
```

Both subcommands accept `--db`. `recommend` prints JSON with product metadata and a source-item trace for the first personalized candidate. `evaluate` fits fresh content and collaborative models at the existing validation cutoff, compares them with popularity through the unchanged evaluator, and prints metrics, per-visitor candidate IDs, metric differences, category counts, and examples of retrieved items with no earlier events. It neither evaluates the final test snapshot nor writes model artifacts. The scikit-learn dependency is declared in `pyproject.toml`; the measured run used version 1.9.1.

### Features and scoring

Each of four blocks receives equal weight: one-hot category, one-hot brand, one-hot price bucket, and lowercase unigram TF-IDF of descriptions. TF-IDF uses smoothed IDF, L2 normalization, and no stop-word removal. Every nonempty block has unit L2 norm; the concatenated item vector is normalized again. This prevents the text block's larger vocabulary from dominating simply through its size. Blank descriptions have zero text vectors; if all descriptions have no usable tokens, the text block is disabled.

Price uses five catalog-wide quantile bins, fitted only on available products. Duplicate boundaries are removed, exact interior-boundary ties enter the higher bucket, and constant prices use one bucket. Validation edges are **10.03, 30.836, 51.668, 71.19, 88.356, 155.00**. The validation matrix has **100 rows and 77 columns**: 5 category, 8 brand, 5 price, and 59 text features. Sorted item IDs, fitted encoders/vectorizer, price boundaries, feature names, and block boundaries remain available on the model for inspection.

The profile is the weighted average of normalized item vectors, using event weights **view=1, add-to-cart=3, purchase=5**, aggregated by item. It is then normalized for cosine scoring. For item vectors `x_i` and strengths `w_i`, candidate `x` receives:

```text
profile = sum(w_i * x_i) / sum(w_i)
score = dot(profile, x) / norm(profile)
source contribution_i = w_i * dot(x_i, x) / norm(sum(w_j * x_j))
```

Source contributions sum to the candidate score. Content scores lie in [0, 1] and are similarity signals, not probabilities. Their scale differs from collaborative and popularity scores; comparing their raw magnitudes across generators would be misleading.

History must belong to the requested visitor and be strictly before the model cutoff. Future products do not enter encoders, price bins, TF-IDF vocabulary/IDF, or retrieval. Products created exactly at the cutoff are available; missing creation times retain the existing eligibility rule. Previously purchased items inform the profile but cannot be recommended. Viewed/carted items remain eligible, including the source item itself. Personalized generation returns only positive-score items, with item-ID tie breaking and no popularity backfill. Empty histories use pre-cutoff popularity, including eligible zero-score completion. Empty or exhausted catalogs return an empty result with the standard columns.

### Validation comparison

Seed 42; history before **2025-03-17 UTC**, outcomes in **[2025-03-17, 2025-03-24)**. The full available catalog contains 100 items. Personalized evaluation includes the same **12 visitors and 16 recommendable positive pairs** as Phase 3; 37 known visitors without recommendable positives are excluded from averages. The one observed cold-start visitor has no relevant outcome, so cold-start quality metrics remain unavailable. No feature or event weights were tuned.

| Measure | Popularity | Collaborative | Content |
| --- | ---: | ---: | ---: |
| Candidate Recall@50 | 0.652778 | 1.000000 | 1.000000 |
| Precision@10 | 0.016667 | 0.075000 | 0.050000 |
| Recall@10 | 0.166667 | 0.652778 | 0.388889 |
| MAP@10 | 0.069444 | 0.181812 | 0.102778 |
| Catalog coverage | 11% | 65% | 71% |
| Median latency (ms) | 2.537250 | 2.416750 | 3.051750 |
| p95 latency (ms) | 2.899595 | 3.428645 | 3.922590 |

One-time fitting took **84.337300 ms** for content and **46.712200 ms** for collaborative. Latency follows the existing warm-up and callback/top-10 protocol, excluding fitting, initial data loading, labels, and diagnostics. Popularity queries SQLite; both personalized models use memory. Timings vary by run and measure these local code paths, not HTTP latency. Every known visitor received 50 content candidates in this dataset; this is an observation rather than a guaranteed minimum pool size.

### Examples and new items

These are the first three candidates at the validation cutoff, shown as `item_id (score)`:

| Visitor | Content | Collaborative |
| --- | --- | --- |
| visitor_0001 | item_0040 (0.888677), item_0075 (0.862225), item_0005 (0.834102) | item_0020 (6.095828), item_0030 (4.927871), item_0088 (3.923962) |
| visitor_0032 | item_0018 (0.692907), item_0073 (0.625753), item_0038 (0.594896) | item_0038 (24.998652), item_0033 (24.011296), item_0001 (21.013689) |
| visitor_0050 | item_0013 (0.976379), item_0048 (0.976379), item_0028 (0.885781) | item_0041 (2.765050), item_0028 (1.925663), item_0016 (1.605910) |
| visitor_0045 | item_0060 (1.000000), item_0095 (0.594659), item_0098 (0.573613) | item_0055 (0.645497), item_0093 (0.577350), item_0080 (0.445435) |

Both models return popularity for `visitor_unknown`, beginning with item_0008 (62), item_0002 (61), and item_0003 (55). `visitor_0032`'s purchased item_0008 is excluded. `visitor_0045` has only one view of item_0060, so content recommends that still-eligible item with cosine 1.0 and then metadata-similar products.

Eight available products have no pre-cutoff events: item_0086, item_0089, item_0094, item_0095, item_0096, item_0097, item_0098, and item_0100. All eight appear somewhere in the evaluated personalized content pools. For example, **item_0089**, a Fable beauty lip balm priced at **29.50**, is seventh for `visitor_0003` with score **0.719841**, despite having no past interactions. The tiny unit example also demonstrates a metadata-matching unseen item retrieved by content but absent from collaborative retrieval.

### Trace one recommendation

`visitor_0001`'s first content candidate is **item_0040**, a Cedar sports water bottle priced at 64.36, with score **0.888677**. Its source contributions are:

| Source item | Interaction strength | Item cosine | Contribution |
| --- | ---: | ---: | ---: |
| item_0040 | 5 | 1.000000 | 0.462765 |
| item_0075 | 4 | 0.701663 | 0.259764 |
| item_0005 | 1 | 0.700891 | 0.064870 |
| item_0010 | 1 | 0.450891 | 0.041731 |
| item_0035 | 1 | 0.298266 | 0.027605 |
| item_0080 | 1 | 0.296143 | 0.027409 |
| item_0011 | 1 | 0.016496 | 0.001527 |
| item_0012 | 1 | 0.016496 | 0.001527 |
| item_0056 | 1 | 0.015972 | 0.001478 |

The unrounded contributions sum to **0.8886768635759622**. An interaction strength of 5 may reflect several weaker events; it does not necessarily mean a purchase. This item remains recommendable because the visitor has not purchased it.

### Observation, limitations, and checkpoint

**Observation:** content matches collaborative candidate recall and reaches more distinct top-10 products, but its top-10 recall and MAP are lower. A retrieved positive is still not guaranteed a high rank. Category concentration is visible: across the 12 evaluated visitors, the median largest-category share is **90%**, and five lists contain ten products from a single category. For `visitor_0003`, all ten are beauty products; `visitor_0008` splits five electronics/five fashion. No category cap is applied in this phase.

- **Why it helps new items:** an available item's category, brand, price, and description define its vector without needing an interaction column. The visitor still needs history for a personalized profile.
- **Why it can become narrow:** repeated behavior pulls the profile toward similar metadata. Here, descriptions repeat category and brand, reinforcing those preferences even with equal block weights.
- **How event weights matter:** a cart contributes three times a view and a purchase five times a view before averaging; repeated events accumulate. Stronger evidence changes the profile's direction, while uniformly multiplying all strengths leaves cosine scores unchanged. Purchase evidence remains useful even though the purchased product is filtered out.

**Limitations:** the text is short, synthetic, and templated. Shared boilerplate gives some unrelated items small positive similarities. Equal block weights and five quantile buckets are fixed hypotheses; prices within a bucket are indistinguishable, and crossing a boundary changes that feature abruptly. The later real dataset may lack usable descriptions, brands, or prices; do not manufacture those fields or assume this text representation transfers. These offline results on simulated exposure establish reproducible behavior, not business uplift. Content persistence, candidate union, learned ranking, and final diversity rules remain later-phase work.

**Checks:** **128 tests pass** (99 existing plus 29 Phase 4 cases), covering exact profiles/scores/traces, isolated feature effects, TF-IDF, quantile boundaries and ties, unseen-item retrieval, weights, purchase exclusion, deterministic ordering, cutoff leakage, empty inputs, invalid arguments, SQL fallback parity, diagnostics, and validation-only CLI behavior without artifact writes. Additional inspection verified purchase exclusions and top-10 explanation totals for all 49 known visitors. The final test window remains reserved.
