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

## Phase 5 — implemented

Phase 5 creates a bounded candidate union, a shared as-of-time feature function,
and snapshot datasets for Phase 6. The measured retrieval and sampling results,
missed positives, and a complete feature/label trace are in
[reports/candidate_analysis.md](reports/candidate_analysis.md).

### Run and outputs

After importing the synthetic data into SQLite, run from the project root:

```text
uv run python -B -m src.rank_dataset build
uv run python -B -m pytest tests -q -p no:cacheprovider
```

`build` accepts `--db`, `--seed` (default 42), `--negative-ratio` (default 3),
`--output-dir` (default `data/rank`), and `--report` (default
`reports/candidate_analysis.md`). It refits both generators independently at
each of the existing four training cutoffs and the validation cutoff. It does
not build or evaluate the final test snapshot, save model artifacts, or train a
ranker. Source data is opened read-only. Each output file is replaced atomically;
metadata is written last, but the entire output bundle is not one transaction.

The ignored output directory contains:

- `train.csv`: sampled training rows from the four older snapshots.
- `validation.csv`: every generated candidate for every visitor with pre-cutoff
  event history at the validation snapshot, without negative sampling.
- `metadata.json`: ordered feature schema, cutoffs, source budgets, weights,
  sampling configuration, class counts, retrieval diagnostics, and example traces.

On the current seed-42 database, the command produces **196 training rows
(49 positives + 147 negatives)** and **1,744 validation rows (13 positives)**.
All 147 sampled negatives have an impression in their outcome window; proxy
negatives are supported but unnecessary for this run. These are reproducible
observations, not hard-coded dataset requirements.

### Candidate and feature interfaces

`src.candidates.generate_candidates(visitor_id, history, collaborative_model,
content_model)` returns a DataFrame with up to 20 collaborative, 20 content,
and 10 popularity nominations, deduplicated by item ID. Models must have the
same cutoff and available catalog. Short personalized lists are not backfilled
beyond these budgets. Unknown visitors receive at most ten popularity candidates
with only popularity source flags, rather than attributing either retriever's
existing fallback to a personalized source. Prior purchases and unavailable
products remain excluded.

Each row retains `collab_score`, `content_score`, `popularity_score`, and
`from_collab`, `from_content`, `from_popularity`. An absent nomination uses
score **0.0** and flag **0**; a nominated zero-score popularity item has flag
**1**. The union is sorted by item ID for reproducibility and has no learned or
blended ranking. Raw generator scores have different scales.

`src.rank_features.prepare_rank_context(events, items, as_of_time)` prepares
historical aggregates once. `build_rank_features(visitor_id, candidates,
as_of_time, context)` returns `item_id` plus the ordered `FEATURE_COLUMNS`:

- Three nomination scores and three source flags.
- Separate visitor view, cart, and purchase event counts; fractional days since
  last activity.
- Visitor-category affinity: weighted strength for the candidate category divided
  by total visitor strength, using the existing view/cart/purchase weights 1/3/5.
- Item interaction count, available item price, and absolute price difference
  from the visitor's event-weighted mean historical item price.

Behavioral features and candidate fitting use only events strictly before the
cutoff. Creation times at the cutoff are available. Prices and categories are
assumed static because the catalog has no revision history. Empty-history visitor
counts and category affinity are zero; recency and price gap remain missing for
Phase 6's preprocessing. `item_id` is join metadata, excluded from model inputs.

### Labels, sampling, and learning checkpoint

`src.rank_dataset.build_snapshot_dataset(snapshot, events, items, impressions=None)`
returns all labeled rows and diagnostics. Each `(snapshot, visitor_id, item_id)`
is unique. A future cart or purchase in `[cutoff, cutoff + 7 days)` sets label 1;
repeated actions collapse into one label. View-only outcomes are label 0.
Visitors with past events but no future positives still contribute candidate
rows. Cold-start quality is reported separately using the existing evaluator.
Relevant items absent from the union are recorded as misses, never inserted.

`sample_training_rows(rows, negative_ratio=3, seed=42)` retains every retrieved
positive and samples up to three negatives per positive **across each snapshot**.
It first samples label-0 candidates shown in that outcome window, then fills a
shortage from unexposed label-0 candidates. The seed is applied independently to
each snapshot after sorting visitor/item pairs. A snapshot with no retrieved
positives contributes no sampled rows, with an explicit diagnostic. Exposure
flags and provenance are metadata only; future impressions never enter features.
Missing impressions require proxy negatives. An absent target action does not
prove dislike, and sampling changes class prevalence, so later scores should
not be described as calibrated purchase probabilities.

Validation union candidate **Recall@50 is 0.847222** (macro visitor recall),
retrieving **13/16** recommendable positive pairs (micro recall **0.8125**).
Pool sizes are **28–42**, with mean **35.59** across 49 historical visitors.
The earlier collaborative/content results used 50 candidates from each source
separately; their 100% recall is not a comparison at the same budget.
The three omitted pairs are listed in the analysis report. A ranker can only
reorder retrieved items, so it cannot repair these omissions. Inspect retrieval
limitations before proceeding to Phase 6.

The report traces one row's source scores, historical aggregates, available
metadata, and future label timestamps. IDs, snapshot role, outcome end, labels,
exposure flags, and provenance stay outside `FEATURE_COLUMNS`. Tests cover
hand-calculated features and recall, cutoff/label boundaries, future-data leakage,
source flags, sampling, proxy fallback, deterministic exports, cold start, and
final-test reservation.

**Checks:** **168 tests pass** (128 existing plus 40 Phase 5 cases). Rebuilding
the real dataset produced byte-identical training/validation CSVs, metadata,
and analysis report. The source database and existing model artifacts remained
unchanged. The final test snapshot remains reserved.

## Phase 6 — implemented

Phase 6 trains a regularized logistic ranker and compares five ranking methods
on the same Phase 5 candidate union. Complete tuning results, coefficient
interpretations, ranking examples, and limitations are in
[reports/model_comparison.md](reports/model_comparison.md).

### Run

From the project root, using the imported synthetic database:

```text
uv run python -B -m src.train
uv run python -B -m pytest tests -q -p no:cacheprovider
```

The training command accepts `--db`, `--seed` (42), `--negative-ratio` (3),
`--model logistic|blend` (logistic), `--artifact`
(`artifacts/model_bundle.pkl`), and `--report`
(`reports/model_comparison.md`). It rebuilds the four training snapshots and
unsampled validation rows in memory using Phase 5's existing functions. It does
not depend on potentially stale `data/rank` exports or rewrite those exports,
the candidate analysis, or the database. The final test window is not built or
evaluated. Each output is replaced atomically; the artifact and report are not
one transaction.

To deliberately choose the blend and retain the default logistic artifact:

```text
uv run python -B -m src.train --model blend --artifact artifacts/blend_bundle.pkl --report reports/blend_comparison.md
```

Model choice never changes automatically based on validation performance. Empty
or one-class training and unsuccessful logistic tuning cause a clear error when
logistic was requested, leaving its existing artifact intact. Explicit blend
selection can still produce an artifact/report when logistic is unavailable.

### Features, fitting, and serving interface

`src.ranker.fit_ranker(training_rows, C=1.0, class_weight=None, seed=42)` selects
the existing 14 `FEATURE_COLUMNS` by name. IDs, timestamps, labels, and future
exposure metadata are excluded. The complete pipeline uses training-fitted
median imputation (retaining all-missing columns), standard scaling, and
regularized logistic regression with the LBFGS solver and 2,000-iteration limit.
`Ranker.score(feature_rows)` preserves input-row order and returns
`predict_proba(... )[:, 1]`. Missing required features, duplicate feature names,
nonnumeric values, and infinities fail clearly; NaNs are imputed and empty
inference returns an empty array. Reordered columns retain identical scores.

Eight configurations cross `C = 0.01, 0.1, 1, 10` with no class weighting or
`balanced`. Validation MAP@10 selects the winner; ties use Recall@10, smaller C,
then no class weighting. Preprocessing and coefficients never fit validation
rows. The selected pipeline remains fitted on older snapshots, while saved
candidate generators are fitted at the validation cutoff, **2025-03-17 UTC**.

The ignored bundle stores the entire ranker pipeline/schema, bounded
collaborative serving state without its training visitor–item matrix, the fitted
content representation, explicit model choice, and cutoff/sampling/package
provenance. Load only trusted local pickle files. A direct Python inference path
for Phase 7 is:

```python
from src.data import load_events, load_items
from src.rank_features import prepare_rank_context
from src.ranker import load_bundle

bundle = load_bundle("artifacts/model_bundle.pkl", expected_model="logistic")
context = prepare_rank_context(
    load_events(), load_items(), bundle.metadata["serving_cutoff"]
)
visitor = "visitor_0001"
history = context.histories.get(visitor, context.history.iloc[:0])
recommendations = bundle.recommend(visitor, history, context).head(10)
```

The loader rejects incompatible schemas, mismatched generator cutoffs/catalogs,
and an artifact that does not match the requested model. Missing artifacts fail;
there is no silent model switch. Unknown visitors receive up to ten popularity
candidates because missing-history features were not represented in training.
Known visitors receive the whole ordered candidate pool before the caller
selects top 10; category diversity rules and HTTP serving remain Phase 7 work.

### Validation findings and learning checkpoint

Seed 42 produces the existing **196 training rows (49 positives)** and **1,744
unsampled validation rows (13 retrieved positives)**. All five methods share
**0.847222 candidate Recall@50**, covering 13 of 16 recommendable positive pairs.
Quality averages include the same **12 historical visitors**. Cold-start quality
is unavailable because its one observed visitor has no recommendable outcome.

| Method | Precision@10 | Recall@10 | MAP@10 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.016667 | 0.166667 | 0.069444 | 11% |
| Collaborative | 0.075000 | 0.652778 | 0.181812 | 65% |
| Content | 0.050000 | 0.388889 | 0.102778 | 71% |
| Fixed blend | 0.066667 | 0.611111 | 0.218849 | 63% |
| Logistic | 0.066667 | 0.527778 | 0.164616 | 64% |

The winning logistic configuration is **C=0.01 with no class weighting**. It
beats popularity and content on MAP@10 but trails collaborative ranking and the
blend. This measured result is retained; no claim of ML superiority is made.
Individual-source baselines score every union item with that source's actual
score, while logistic features preserve nomination scores/flags. The fixed blend
uses 0.4 collaborative + 0.4 content + 0.2 popularity after per-visitor,
per-source min–max normalization; constant sources contribute zero. Score ties
use ascending item ID. No candidates or missed positives are added by ranking.

Label 1 means a cart or purchase in the seven days following the snapshot.
Negatives mean no observed target action, not dislike. The 3:1 exposure-first
sampling changes class prevalence, so logistic outputs are **ordering scores,
not calibrated purchase probabilities**. Logistic regression's binary objective
and regularization suit this label better than ridge's squared-error numeric
objective. Balanced class weights were tested on unsampled validation and did
not improve the selected configuration.

Coefficients describe conditional log-odds changes per training standard
deviation after imputation/scaling. Collaborative nomination score (**+0.0311**),
content nomination score (**+0.0624**), and historical category affinity
(**+0.0080**) raise the ordering score when other features are fixed. Item price
(**−0.0755**) lowers it. These associations are neither causal effects nor
independent measures of feature importance. The report traces visitor_0003's
item_0069 features through the transformed linear score and sigmoid.

Scoring latency is measured separately on prepared rows with one warm-up and
20 repetitions per historical visitor. It excludes sorting, retrieval, feature
construction, fitting, and HTTP work; source baselines simply read their
prepared scores. Values vary per run and are recorded in the report. The small
synthetic cohort, candidate omissions, negative-label uncertainty, and validation
tuning limit confidence; offline metrics do not prove business uplift.

**Checks:** **205 tests pass** (168 existing plus 37 Phase 6 cases), covering
feature schemas, preprocessing/inference isolation, deterministic fitting,
fixed-pool comparisons, blend arithmetic, cold start, purchase/availability
exclusions, explicit model choice, one-class/nonconvergence behavior, atomic
artifact failure handling, saved-model parity, CLI reproducibility, and final
test reservation.
