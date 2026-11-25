# E3 Unit Dossier — `qwen3-4b-base_en` (Qwen3-4B-Base, EN, BBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **BBQ** (58,492 items, 11 categories) · Source: `results/e3/qwen3-4b-base_en/analysis/20261107_153055_items`.

## 1. Set Summary and Target-Category Localization

**Why this check:** summarizes how strongly each selected (and ablation) neuron set concentrates answer changes (`pred != base_pred`) in its target category versus the mean of the other categories, together with its layer profile and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | ---: | :--- | :--- |
| `real_gender20` | real | Gender_identity | 20 | 397 / 5,672 (7.00%) | [6.38, 7.63] | 7.48% | 0.94× | 4,353 / 58,492 (7.44%) | A100-SXM4-40GB | L32:1, L34:1, L35:18 |
| `real_race20` | real | Race_ethnicity | 20 | 371 / 6,880 (5.39%) | [4.90, 5.94] | 6.33% | 0.85× | 3,695 / 58,492 (6.32%) | A100-SXM4-40GB | L28:1, L35:19 |
| `real_religion20` | real | Religion | 20 | 79 / 1,200 (6.58%) | [5.17, 8.00] | 7.98% | 0.83× | 4,746 / 58,492 (8.11%) | A100-SXM4-40GB | L35:20 |
| `real_gender5` | real | Gender_identity | 5 | 245 / 5,672 (4.32%) | [3.83, 4.85] | 5.22% | 0.83× | 3,132 / 58,492 (5.35%) | A100-SXM4-40GB | L35:5 |
| `real_race5` | real | Race_ethnicity | 5 | 199 / 6,880 (2.89%) | [2.51, 3.31] | 3.03% | 0.96× | 2,123 / 58,492 (3.63%) | A100-SXM4-40GB | L35:5 |
| `real_religion5` | real | Religion | 5 | 49 / 1,200 (4.08%) | [3.08, 5.25] | 3.89% | 1.05× | 2,560 / 58,492 (4.38%) | A100-SXM4-40GB | L35:5 |

*Legend — **Kind**: `real` (selected top-K) or `ablation` (exploratory subset); **Target Flips**: changed answers (`pred != base_pred`) in target category; **95% CI (%)**: paired item-bootstrap interval (`B = 1,000`); **Other Mean**: mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **Layer Profile**: `L<layer>:<count>` of zeroed neurons.*

## 2. Selected vs. Layer-Matched Random Neurons

**Why this check:** tests whether the BiasScore-selected neurons change substantially more target-category answers than uniform random neuron sets drawn from the exact same MLP layers.

| Comparison | Target Cat. | Selected Flips | Random Flips (min–max, mean) | Real/Rand | Rank | GPUs |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| Gender·20 (seeds 0-4) | Gender_identity | 397 / 5,672 (7.00%) | 28–64 (0.49–1.13%, mean 0.76%) | 9.19× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |
| Religion·20 (seeds 0-4) | Religion | 79 / 1,200 (6.58%) | 8–16 (0.67–1.33%, mean 0.93%) | 7.05× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |

*Legend — **Selected Flips**: target-category changed answers (`changed / n_items`, `%`, `pred != base_pred`); **Random Flips (min–max, mean)**: range across the layer-matched random seeds reported as `min–max changed items (min–max%, mean %)` over the same `n_items` target denominator; **Real/Rand**: `Selected % / Random mean %`; **Rank**: position of selected set among selected + random (`1` = highest).*

## 3. Changed Answers Across All Categories (`Set × Category`)

**Why this table:** exposes the full `set × category` flip matrix (`pred != base_pred`, all items pooled; random families summarized by mean) so off-target spillover across individual categories can be inspected directly.

| Category | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `rand_gender20` (mean) | `rand_religion20` (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 3,680 | 324 (8.80%) | 156 (4.24%) | 205 (5.57%) | 91 (2.47%) | 319 (8.67%) | 102 (2.77%) | 0.74% [0.65–0.87] | 0.62% [0.38–1.09] |
| Disability_status | 1,556 | 100 (6.43%) | 64 (4.11%) | 69 (4.43%) | 49 (3.15%) | 104 (6.68%) | 44 (2.83%) | 0.71% [0.51–1.09] | 0.51% [0.32–0.58] |
| Gender_identity | 5,672 | 397 (7.00%) | 245 (4.32%) | 331 (5.84%) | 144 (2.54%) | 384 (6.77%) | 238 (4.20%) | 0.76% [0.49–1.13] | 0.46% [0.14–0.81] |
| Nationality | 3,080 | 250 (8.12%) | 187 (6.07%) | 209 (6.79%) | 95 (3.08%) | 292 (9.48%) | 135 (4.38%) | 0.71% [0.42–0.91] | 0.62% [0.42–0.88] |
| Physical_appearance | 1,576 | 129 (8.19%) | 68 (4.31%) | 82 (5.20%) | 31 (1.97%) | 140 (8.88%) | 48 (3.05%) | 0.63% [0.57–0.70] | 0.46% [0.25–0.70] |
| Race_ethnicity | 6,880 | 408 (5.93%) | 361 (5.25%) | 371 (5.39%) | 199 (2.89%) | 448 (6.51%) | 258 (3.75%) | 0.74% [0.62–0.84] | 0.51% [0.33–0.70] |
| Race_x_SES | 11,160 | 831 (7.45%) | 616 (5.52%) | 700 (6.27%) | 683 (6.12%) | 1,002 (8.98%) | 677 (6.07%) | 0.94% [0.83–1.04] | 0.79% [0.56–1.41] |
| Race_x_gender | 15,960 | 1,275 (7.99%) | 964 (6.04%) | 1,158 (7.26%) | 594 (3.72%) | 1,460 (9.15%) | 749 (4.69%) | 1.08% [0.73–1.71] | 0.75% [0.46–1.22] |
| Religion | 1,200 | 72 (6.00%) | 64 (5.33%) | 79 (6.58%) | 34 (2.83%) | 79 (6.58%) | 49 (4.08%) | 1.28% [1.00–1.50] | 0.93% [0.67–1.33] |
| SES | 6,864 | 491 (7.15%) | 354 (5.16%) | 410 (5.97%) | 189 (2.75%) | 448 (6.53%) | 227 (3.31%) | 0.68% [0.58–0.76] | 0.54% [0.25–1.06] |
| Sexual_orientation | 864 | 76 (8.80%) | 53 (6.13%) | 81 (9.38%) | 14 (1.62%) | 70 (8.10%) | 33 (3.82%) | 0.83% [0.58–1.04] | 0.62% [0.12–1.39] |

*Legend — Cells show `changed (flips_pct)` (`pred != base_pred`, pooled `ambig + disambig`); random-family columns show `mean% [min–max]` across seeds.*

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 13.42% / 4.18% | 6.85% / 1.63% | 8.86% / 2.28% | 4.40% / 0.54% | 13.59% / 3.75% | 4.40% / 1.14% |
| Disability_status | 10.54% / 2.31% | 6.81% / 1.41% | 7.58% / 1.29% | 5.66% / 0.64% | 11.05% / 2.31% | 4.76% / 0.90% |
| Gender_identity | 10.68% / 3.28% | 7.26% / 1.38% | 8.99% / 2.68% | 4.13% / 0.85% | 10.68% / 2.86% | 6.81% / 1.59% |
| Nationality | 11.95% / 4.29% | 9.09% / 3.05% | 10.06% / 3.51% | 4.68% / 1.49% | 14.68% / 4.29% | 6.36% / 2.40% |
| Physical_appearance | 10.91% / 5.46% | 5.84% / 2.79% | 7.87% / 2.54% | 2.03% / 1.90% | 11.93% / 5.84% | 4.06% / 2.03% |
| Race_ethnicity | 10.03% / 1.83% | 9.10% / 1.40% | 9.22% / 1.57% | 5.12% / 0.67% | 10.70% / 2.33% | 6.57% / 0.93% |
| Race_x_SES | 7.56% / 5.77% | 5.07% / 4.82% | 6.49% / 4.84% | 6.27% / 5.16% | 9.30% / 6.90% | 6.86% / 4.52% |
| Race_x_gender | 8.65% / 1.32% | 7.53% / 1.03% | 7.81% / 1.12% | 4.30% / 0.80% | 10.09% / 1.47% | 5.60% / 0.63% |
| Religion | 9.50% / 2.50% | 7.67% / 3.00% | 10.00% / 3.17% | 3.83% / 1.83% | 10.83% / 2.33% | 6.83% / 1.33% |
| SES | 12.30% / 2.01% | 8.92% / 1.40% | 10.40% / 1.54% | 4.60% / 0.90% | 11.39% / 1.66% | 5.68% / 0.93% |
| Sexual_orientation | 15.28% / 2.31% | 10.88% / 1.39% | 16.20% / 2.55% | 2.78% / 0.46% | 13.89% / 2.31% | 6.94% / 0.69% |

*Legend — Cells show `ambig% / disambig%` (S/A/U state changes from `direction.csv`).*

</details>

## 4. Neuron and Changed-Item Overlap Between Sets

**Why this check:** tests whether distinct target categories share the same high-BiasScore neurons and whether different interventions flip the same underlying items (pairwise shared neurons, neuron Jaccard, and changed-item Jaccard).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `real_gender20` vs. `real_religion20` | 16 / 20,20 | 0.667 | 0.649 | Gender_identity: 0.644 | Religion: 0.606 | L35.3, L35.6, L35.16, L35.36, L35.57, L35.69, L35.76, L35.123, L35.188, L35.235, L35.248, L35.284, L35.363, L35.935, L35.1158 ... |
| `real_gender20` vs. `real_race20` | 10 / 20,20 | 0.333 | 0.586 | Gender_identity: 0.655 | Race_ethnicity: 0.583 | L35.3, L35.16, L35.36, L35.69, L35.188, L35.235, L35.284, L35.363, L35.935, L35.3421 |
| `real_race20` vs. `real_religion20` | 10 / 20,20 | 0.333 | 0.520 | Race_ethnicity: 0.508 | Religion: 0.463 | L35.3, L35.16, L35.36, L35.69, L35.188, L35.235, L35.284, L35.363, L35.935, L35.3421 |
| `real_race20` vs. `real_gender5` | 5 / 20,5 | 0.250 | 0.584 | Race_ethnicity: 0.702 | Gender_identity: 0.569 | L35.16, L35.36, L35.284, L35.363, L35.935 |
| `real_gender20` vs. `real_gender5` | 5 / 20,5 | 0.250 | 0.421 | Gender_identity: 0.436 | Gender_identity: 0.436 | L35.16, L35.36, L35.284, L35.363, L35.935 |
| `real_religion20` vs. `real_gender5` | 5 / 20,5 | 0.250 | 0.374 | Religion: 0.312 | Gender_identity: 0.453 | L35.16, L35.36, L35.284, L35.363, L35.935 |
| `real_race20` vs. `real_religion5` | 5 / 20,5 | 0.250 | 0.370 | Race_ethnicity: 0.413 | Religion: 0.362 | L35.16, L35.36, L35.363, L35.935, L35.3421 |
| `real_gender20` vs. `real_religion5` | 5 / 20,5 | 0.250 | 0.305 | Gender_identity: 0.377 | Religion: 0.163 | L35.16, L35.36, L35.363, L35.935, L35.3421 |
| `real_religion20` vs. `real_religion5` | 5 / 20,5 | 0.250 | 0.288 | Religion: 0.174 | Religion: 0.174 | L35.16, L35.36, L35.363, L35.935, L35.3421 |
| `real_race20` vs. `real_race5` | 5 / 20,5 | 0.250 | 0.213 | Race_ethnicity: 0.226 | Race_ethnicity: 0.226 | L35.16, L35.935, L35.2005, L35.3490, L35.3824 |
| `real_gender5` vs. `real_religion5` | 4 / 5,5 | 0.667 | 0.404 | Gender_identity: 0.495 | Religion: 0.412 | L35.16, L35.36, L35.363, L35.935 |
| `real_race5` vs. `real_religion5` | 2 / 5,5 | 0.250 | 0.345 | Race_ethnicity: 0.340 | Religion: 0.431 | L35.16, L35.935 |
| `real_gender5` vs. `real_race5` | 2 / 5,5 | 0.250 | 0.249 | Gender_identity: 0.118 | Race_ethnicity: 0.258 | L35.16, L35.935 |
| `real_gender20` vs. `real_race5` | 2 / 20,5 | 0.087 | 0.181 | Gender_identity: 0.053 | Race_ethnicity: 0.167 | L35.16, L35.935 |
| `real_religion20` vs. `real_race5` | 2 / 20,5 | 0.087 | 0.161 | Religion: 0.087 | Race_ethnicity: 0.145 | L35.16, L35.935 |

*Legend — **Shared Neurons**: identical `(layer, index)` neurons (`shared / size_A, size_B`); **Neuron Jaccard**: `|A ∩ B| / |A ∪ B|` over zeroed neurons; **Items Jacc (All / Target A / Target B)**: Jaccard overlap of changed items across all items or restricted to each set's target category.*

## 5. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

**Why this table:** reports secondary directional metrics (BBQ/EsBBQ bias-score deltas, stereotype transitions `A→S`/`S→A`, and accuracy transitions) for descriptive completeness; these are secondary to answer-change localization.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | Disamb Bias Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: |
| Gender_identity | `real_gender20` | ambig | 303 / 2,836 (10.68%) | 140 / 163 | -0.81% | 0 / 0 | - |
| Gender_identity | `real_gender20` | disambig | 93 / 2,836 (3.28%) | 43 / 50 | -0.25% | 60 / 33 | -0.49 [-1.90, 0.85] |
| Gender_identity | `real_gender5` | ambig | 206 / 2,836 (7.26%) | 98 / 108 | -0.35% | 0 / 0 | - |
| Gender_identity | `real_gender5` | disambig | 39 / 2,836 (1.38%) | 19 / 20 | -0.04% | 22 / 17 | -0.07 [-0.92, 0.78] |
| Gender_identity | `real_race20` | ambig | 255 / 2,836 (8.99%) | 131 / 124 | 0.25% | 0 / 0 | - |
| Gender_identity | `real_race20` | disambig | 76 / 2,836 (2.68%) | 36 / 40 | -0.14% | 50 / 26 | -0.28 [-1.55, 0.92] |
| Gender_identity | `real_race5` | ambig | 117 / 2,836 (4.13%) | 57 / 60 | -0.11% | 0 / 0 | - |
| Gender_identity | `real_race5` | disambig | 24 / 2,836 (0.85%) | 13 / 11 | 0.07% | 11 / 13 | 0.14 [-0.49, 0.85] |
| Gender_identity | `real_religion20` | ambig | 303 / 2,836 (10.68%) | 137 / 166 | -1.02% | 0 / 0 | - |
| Gender_identity | `real_religion20` | disambig | 81 / 2,836 (2.86%) | 40 / 41 | -0.04% | 52 / 29 | -0.07 [-1.34, 1.20] |
| Gender_identity | `real_religion5` | ambig | 193 / 2,836 (6.81%) | 106 / 87 | 0.67% | 0 / 0 | - |
| Gender_identity | `real_religion5` | disambig | 45 / 2,836 (1.59%) | 20 / 25 | -0.18% | 25 / 20 | -0.35 [-1.27, 0.56] |
| Race_ethnicity | `real_gender20` | ambig | 345 / 3,440 (10.03%) | 173 / 158 | 0.44% | 12 / 2 | - |
| Race_ethnicity | `real_gender20` | disambig | 63 / 3,440 (1.83%) | 34 / 26 | 0.23% | 45 / 17 | 0.56 [-0.29, 1.38] |
| Race_ethnicity | `real_gender5` | ambig | 313 / 3,440 (9.10%) | 147 / 139 | 0.23% | 3 / 24 | - |
| Race_ethnicity | `real_gender5` | disambig | 48 / 3,440 (1.40%) | 25 / 20 | 0.15% | 34 / 13 | 0.21 [-0.53, 0.94] |
| Race_ethnicity | `real_race20` | ambig | 317 / 3,440 (9.22%) | 158 / 137 | 0.61% | 3 / 19 | - |
| Race_ethnicity | `real_race20` | disambig | 54 / 3,440 (1.57%) | 33 / 19 | 0.41% | 39 / 15 | 0.76 [-0.00, 1.55] |
| Race_ethnicity | `real_race5` | ambig | 176 / 3,440 (5.12%) | 79 / 71 | 0.23% | 1 / 25 | - |
| Race_ethnicity | `real_race5` | disambig | 23 / 3,440 (0.67%) | 11 / 9 | 0.06% | 12 / 10 | 0.03 [-0.47, 0.53] |
| Race_ethnicity | `real_religion20` | ambig | 368 / 3,440 (10.70%) | 179 / 176 | 0.09% | 10 / 3 | - |
| Race_ethnicity | `real_religion20` | disambig | 80 / 3,440 (2.33%) | 46 / 29 | 0.49% | 59 / 20 | 1.02 [0.06, 2.02] |
| Race_ethnicity | `real_religion5` | ambig | 226 / 3,440 (6.57%) | 109 / 105 | 0.12% | 3 / 9 | - |
| Race_ethnicity | `real_religion5` | disambig | 32 / 3,440 (0.93%) | 19 / 10 | 0.26% | 21 / 10 | 0.61 [-0.00, 1.29] |
| Religion | `real_gender20` | ambig | 57 / 600 (9.50%) | 19 / 16 | 0.50% | 4 / 18 | - |
| Religion | `real_gender20` | disambig | 15 / 600 (2.50%) | 9 / 2 | 1.17% | 9 / 6 | 2.79 [0.50, 5.13] |
| Religion | `real_gender5` | ambig | 46 / 600 (7.67%) | 16 / 20 | -0.67% | 1 / 9 | - |
| Religion | `real_gender5` | disambig | 18 / 600 (3.00%) | 12 / 5 | 1.17% | 12 / 6 | 2.66 [-0.03, 5.91] |
| Religion | `real_race20` | ambig | 60 / 600 (10.00%) | 26 / 17 | 1.50% | 2 / 15 | - |
| Religion | `real_race20` | disambig | 19 / 600 (3.17%) | 12 / 6 | 1.00% | 11 / 8 | 2.31 [-0.35, 5.46] |
| Religion | `real_race5` | ambig | 23 / 600 (3.83%) | 7 / 11 | -0.67% | 2 / 3 | - |
| Religion | `real_race5` | disambig | 11 / 600 (1.83%) | 2 / 5 | -0.50% | 6 / 5 | -1.39 [-3.32, 0.59] |
| Religion | `real_religion20` | ambig | 65 / 600 (10.83%) | 22 / 21 | 0.17% | 2 / 20 | - |
| Religion | `real_religion20` | disambig | 14 / 600 (2.33%) | 6 / 3 | 0.50% | 6 / 8 | 1.18 [-0.91, 3.45] |
| Religion | `real_religion5` | ambig | 41 / 600 (6.83%) | 17 / 16 | 0.17% | 3 / 5 | - |
| Religion | `real_religion5` | disambig | 8 / 600 (1.33%) | 4 / 4 | 0.00% | 4 / 4 | 0.00 [-1.78, 2.15] |

*Legend — **A→S / S→A**: flips from anti-stereotype to stereotyped person and vice versa; **Net→Stereo**: `(A→S − S→A) / n_items` (`%`); **Corr→Inc / Inc→Corr**: accuracy transitions; **Bias Δ (95% CI)**: change in benchmark bias score vs. baseline with paired item-bootstrap CI.*

## 6. Baseline Log-Likelihood Margin Mechanism

**Why this check:** verifies whether changed answers come selectively from baseline-undecided items (low `top − second` margin) rather than confident predictions, comparing median margins and flip rates across deciles 1–10.

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `real_gender20` | real | 4,353 | 0.344 | 2.742 |
| `real_race20` | real | 3,695 | 0.281 | 2.672 |
| `real_religion20` | real | 4,746 | 0.375 | 2.750 |
| `real_gender5` | real | 3,132 | 0.250 | 2.629 |
| `real_race5` | real | 2,123 | 0.156 | 2.562 |
| `real_religion5` | real | 2,560 | 0.188 | 2.609 |
| `rand_gender20 (mean n=5)` | random_family_mean | 510.4 | 0.065 | 2.475 |
| `rand_religion20 (mean n=5)` | random_family_mean | 380.4 | 0.066 | 2.453 |

| Decile | Margin Range | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `rand_gender20 (mean n=5)` | `rand_religion20 (mean n=5)` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.375] | 41.01% | 35.49% | 38.12% | 28.75% | 41.93% | 32.74% | 8.60% | 6.00% |
| 2 | [0.375, 0.773] | 21.41% | 13.39% | 18.09% | 5.47% | 23.78% | 8.31% | 0.05% | 0.24% |
| 3 | [0.773, 1.248] | 8.89% | 4.24% | 5.78% | 1.93% | 10.81% | 2.34% | 0.07% | 0.12% |
| 4 | [1.248, 1.750] | 2.51% | 0.34% | 0.99% | 0.14% | 3.66% | 0.38% | 0.00% | 0.09% |
| 5 | [1.750, 2.422] | 0.48% | 0.09% | 0.14% | 0.00% | 0.82% | 0.00% | 0.00% | 0.05% |
| 6 | [2.422, 3.281] | 0.12% | 0.00% | 0.05% | 0.00% | 0.14% | 0.00% | 0.00% | 0.00% |
| 7 | [3.281, 4.391] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 8 | [4.391, 5.672] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 9 | [5.672, 7.363] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 10 | [7.363, 17.688] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) of changed vs. unchanged items; **Decile**: baseline-margin decile (`1` = smallest margin / most undecided, `10` = largest margin).*

## 7. Granular Breakdowns (Demographic Groups and Topics)

**Why this table:** checks whether answer changes concentrate in specific demographic subgroups (`answer_info`) or social topics (`relevant_social_value`, EsBBQ) within each target category.

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender_identity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | 4,616 | 7.13% | 4.48% | 5.74% | 2.64% | 6.65% | 4.44% |
| M | 4,616 | 7.13% | 4.48% | 5.74% | 2.64% | 6.65% | 4.44% |
| boy | 40 | 10.00% | 2.50% | 2.50% | 5.00% | 2.50% | 5.00% |
| girl | 40 | 5.00% | 2.50% | 2.50% | 7.50% | 2.50% | 5.00% |
| man | 152 | 9.87% | 0.00% | 9.87% | 0.66% | 9.21% | 3.95% |
| nonTrans | 464 | 7.97% | 5.17% | 7.97% | 1.51% | 8.19% | 3.45% |
| nonTrans_F | 200 | 1.00% | 2.00% | 1.50% | 2.00% | 4.50% | 1.00% |
| nonTrans_M | 200 | 5.50% | 4.50% | 5.00% | 3.50% | 7.50% | 3.50% |
| trans | 464 | 7.97% | 5.17% | 7.97% | 1.51% | 8.19% | 3.45% |
| trans_F | 200 | 5.50% | 4.50% | 5.00% | 3.50% | 7.50% | 3.50% |
| trans_M | 200 | 1.00% | 2.00% | 1.50% | 2.00% | 4.50% | 1.00% |
| woman | 152 | 9.87% | 0.00% | 9.87% | 1.32% | 9.21% | 3.95% |

### Race_ethnicity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| African | 200 | 3.50% | 5.50% | 5.50% | 2.50% | 4.00% | 2.50% |
| African American | 520 | 3.46% | 4.23% | 3.65% | 1.54% | 3.08% | 2.31% |
| Arab | 208 | 3.37% | 4.81% | 4.33% | 1.44% | 3.85% | 2.88% |
| Asian | 312 | 3.21% | 2.88% | 3.53% | 1.28% | 2.88% | 2.88% |
| Black | 520 | 2.50% | 2.69% | 2.88% | 1.73% | 2.50% | 1.92% |
| Caucasian | 128 | 11.72% | 6.25% | 6.25% | 0.78% | 8.59% | 5.47% |
| European | 184 | 5.43% | 5.98% | 5.98% | 4.35% | 4.89% | 4.89% |
| F-Arab | 1,056 | 8.14% | 6.44% | 6.25% | 3.79% | 9.38% | 4.17% |
| F-Asian | 904 | 6.08% | 4.65% | 4.87% | 3.65% | 6.75% | 3.65% |
| F-Black | 1,904 | 6.51% | 5.09% | 5.30% | 2.52% | 8.35% | 3.20% |
| F-Latino | 1,112 | 5.49% | 4.95% | 5.04% | 3.06% | 7.28% | 3.78% |
| F-White | 784 | 6.89% | 6.38% | 6.51% | 1.91% | 7.40% | 2.81% |
| Hispanic | 328 | 5.49% | 4.27% | 5.18% | 1.83% | 5.79% | 1.52% |
| Jewish | 168 | 2.98% | 3.57% | 3.57% | 0.60% | 3.57% | 1.79% |
| Latino | 304 | 6.58% | 6.58% | 5.26% | 4.61% | 6.91% | 5.26% |
| M-Arab | 824 | 7.16% | 5.70% | 5.70% | 4.37% | 7.65% | 5.70% |
| M-Asian | 848 | 7.08% | 6.72% | 7.43% | 3.18% | 7.43% | 4.83% |
| M-Black | 1,032 | 5.81% | 5.04% | 5.72% | 3.20% | 5.81% | 3.39% |
| M-Latino | 936 | 6.94% | 5.88% | 6.09% | 3.42% | 6.73% | 4.81% |
| M-White | 600 | 5.00% | 5.17% | 5.33% | 2.00% | 4.83% | 4.67% |
| Middle Eastern | 224 | 3.57% | 5.80% | 5.36% | 4.02% | 2.68% | 5.36% |
| Native American | 240 | 2.92% | 2.92% | 4.17% | 3.33% | 4.17% | 2.50% |
| Roma | 168 | 5.36% | 3.57% | 2.98% | 1.19% | 4.76% | 2.98% |
| South American | 136 | 8.82% | 8.09% | 8.09% | 5.88% | 8.82% | 5.88% |
| White | 120 | 2.50% | 5.00% | 4.17% | 1.67% | 3.33% | 4.17% |

### Religion — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atheist | 288 | 5.56% | 6.60% | 5.21% | 2.43% | 6.25% | 4.17% |
| Buddhist | 112 | 5.36% | 3.57% | 4.46% | 3.57% | 3.57% | 6.25% |
| Catholic | 264 | 6.44% | 4.92% | 6.06% | 4.17% | 7.20% | 5.68% |
| Christian | 312 | 5.45% | 4.81% | 8.65% | 1.60% | 6.73% | 3.85% |
| Hindu | 264 | 7.20% | 6.82% | 7.95% | 6.06% | 7.58% | 6.82% |
| Jewish | 280 | 5.71% | 5.00% | 6.07% | 1.07% | 5.71% | 3.21% |
| Mormon | 264 | 7.95% | 6.06% | 8.33% | 3.79% | 9.47% | 2.65% |
| Muslim | 328 | 6.71% | 5.18% | 6.10% | 1.83% | 7.32% | 3.05% |
| Protestant | 144 | 2.08% | 3.47% | 2.78% | 0.69% | 2.08% | 1.39% |
| Sikh | 144 | 4.86% | 4.86% | 7.64% | 3.47% | 5.56% | 4.17% |

</details>
