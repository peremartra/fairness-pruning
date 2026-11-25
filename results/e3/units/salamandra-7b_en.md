# E3 Unit Dossier — `salamandra-7b_en` (Salamandra-7B, EN, BBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **BBQ** (58,492 items, 11 categories) · Source: `results/e3/salamandra-7b_en/analysis/20261107_153940_items`.

## 1. Set Summary and Target-Category Localization

**Why this check:** summarizes how strongly each selected (and ablation) neuron set concentrates answer changes (`pred != base_pred`) in its target category versus the mean of the other categories, together with its layer profile and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | ---: | :--- | :--- |
| `real_gender20` | real | Gender_identity | 20 | 928 / 5,672 (16.36%) | [15.39, 17.28] | 4.31% | 3.79× | 3,860 / 58,492 (6.60%) | A100-SXM4-40GB | L22:1, L24:1, L26:1, L27:2, L28:2, L29:2, L30:3, L31:8 |
| `real_race20` | real | Race_ethnicity | 20 | 519 / 6,880 (7.54%) | [6.95, 8.18] | 4.14% | 1.82× | 3,131 / 58,492 (5.35%) | A100-SXM4-40GB | L29:1, L30:1, L31:18 |
| `real_religion20` | real | Religion | 20 | 163 / 1,200 (13.58%) | [11.75, 15.59] | 5.49% | 2.48× | 3,866 / 58,492 (6.61%) | A100-SXM4-40GB | L23:1, L25:1, L27:1, L31:17 |
| `real_gender5` | real | Gender_identity | 5 | 913 / 5,672 (16.10%) | [15.14, 17.00] | 2.95% | 5.45× | 2,953 / 58,492 (5.05%) | A100-SXM4-40GB | L27:1, L29:2, L31:2 |
| `real_race5` | real | Race_ethnicity | 5 | 351 / 6,880 (5.10%) | [4.61, 5.58] | 2.84% | 1.79× | 2,255 / 58,492 (3.86%) | A100-SXM4-40GB | L31:5 |
| `real_religion5` | real | Religion | 5 | 113 / 1,200 (9.42%) | [7.75, 11.08] | 3.41% | 2.76× | 2,471 / 58,492 (4.22%) | A100-SXM4-40GB | L25:1, L31:4 |

*Legend — **Kind**: `real` (selected top-K) or `ablation` (exploratory subset); **Target Flips**: changed answers (`pred != base_pred`) in target category; **95% CI (%)**: paired item-bootstrap interval (`B = 1,000`); **Other Mean**: mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **Layer Profile**: `L<layer>:<count>` of zeroed neurons.*

## 2. Selected vs. Layer-Matched Random Neurons

**Why this check:** tests whether the BiasScore-selected neurons change substantially more target-category answers than uniform random neuron sets drawn from the exact same MLP layers.

| Comparison | Target Cat. | Selected Flips | Random Flips (min–max, mean) | Real/Rand | Rank | GPUs |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| Gender·20 (seeds 0-4) | Gender_identity | 928 / 5,672 (16.36%) | 41–81 (0.72–1.43%, mean 0.93%) | 17.58× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |
| Religion·20 (seeds 0-4) | Religion | 163 / 1,200 (13.58%) | 11–19 (0.92–1.58%, mean 1.23%) | 11.01× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |

*Legend — **Selected Flips**: target-category changed answers (`changed / n_items`, `%`, `pred != base_pred`); **Random Flips (min–max, mean)**: range across the layer-matched random seeds reported as `min–max changed items (min–max%, mean %)` over the same `n_items` target denominator; **Real/Rand**: `Selected % / Random mean %`; **Rank**: position of selected set among selected + random (`1` = highest).*

## 3. Changed Answers Across All Categories (`Set × Category`)

**Why this table:** exposes the full `set × category` flip matrix (`pred != base_pred`, all items pooled; random families summarized by mean) so off-target spillover across individual categories can be inspected directly.

| Category | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `rand_gender20` (mean) | `rand_religion20` (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 3,680 | 122 (3.32%) | 103 (2.80%) | 116 (3.15%) | 95 (2.58%) | 107 (2.91%) | 79 (2.15%) | 1.30% [1.17–1.66] | 1.40% [1.14–1.71] |
| Disability_status | 1,556 | 29 (1.86%) | 15 (0.96%) | 27 (1.74%) | 19 (1.22%) | 39 (2.51%) | 11 (0.71%) | 0.80% [0.58–0.90] | 0.68% [0.51–1.03] |
| Gender_identity | 5,672 | 928 (16.36%) | 913 (16.10%) | 244 (4.30%) | 216 (3.81%) | 233 (4.11%) | 128 (2.26%) | 0.93% [0.72–1.43] | 0.99% [0.67–1.60] |
| Nationality | 3,080 | 83 (2.69%) | 52 (1.69%) | 124 (4.03%) | 80 (2.60%) | 378 (12.27%) | 275 (8.93%) | 1.06% [0.75–1.69] | 1.31% [0.97–2.05] |
| Physical_appearance | 1,576 | 68 (4.31%) | 58 (3.68%) | 50 (3.17%) | 25 (1.59%) | 44 (2.79%) | 17 (1.08%) | 0.99% [0.57–1.27] | 1.09% [0.82–1.27] |
| Race_ethnicity | 6,880 | 428 (6.22%) | 268 (3.90%) | 519 (7.54%) | 351 (5.10%) | 615 (8.94%) | 415 (6.03%) | 1.05% [0.90–1.42] | 1.17% [0.81–1.83] |
| Race_x_SES | 11,160 | 580 (5.20%) | 319 (2.86%) | 626 (5.61%) | 451 (4.04%) | 631 (5.65%) | 367 (3.29%) | 1.47% [1.20–1.81] | 1.53% [1.24–2.11] |
| Race_x_gender | 15,960 | 1,334 (8.36%) | 1,053 (6.60%) | 1,098 (6.88%) | 835 (5.23%) | 1,320 (8.27%) | 911 (5.71%) | 0.97% [0.87–1.23] | 1.04% [0.73–1.67] |
| Religion | 1,200 | 50 (4.17%) | 16 (1.33%) | 56 (4.67%) | 30 (2.50%) | 163 (13.58%) | 113 (9.42%) | 1.35% [0.92–1.92] | 1.23% [0.92–1.58] |
| SES | 6,864 | 203 (2.96%) | 122 (1.78%) | 232 (3.38%) | 127 (1.85%) | 311 (4.53%) | 138 (2.01%) | 0.84% [0.70–1.22] | 1.05% [0.79–1.60] |
| Sexual_orientation | 864 | 35 (4.05%) | 34 (3.94%) | 39 (4.51%) | 26 (3.01%) | 25 (2.89%) | 17 (1.97%) | 1.67% [1.27–2.78] | 1.57% [1.16–2.08] |

*Legend — Cells show `changed (flips_pct)` (`pred != base_pred`, pooled `ambig + disambig`); random-family columns show `mean% [min–max]` across seeds.*

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 5.05% / 1.58% | 4.67% / 0.92% | 5.38% / 0.92% | 4.35% / 0.82% | 4.78% / 1.03% | 3.70% / 0.60% |
| Disability_status | 3.34% / 0.39% | 1.54% / 0.39% | 2.83% / 0.64% | 1.93% / 0.51% | 3.73% / 1.29% | 1.29% / 0.13% |
| Gender_identity | 24.26% / 8.43% | 24.12% / 8.04% | 6.28% / 2.29% | 5.50% / 2.08% | 6.03% / 2.15% | 3.39% / 1.09% |
| Nationality | 4.03% / 1.36% | 2.53% / 0.84% | 5.78% / 2.27% | 3.64% / 1.56% | 16.95% / 7.60% | 12.27% / 5.58% |
| Physical_appearance | 4.31% / 4.31% | 4.06% / 3.30% | 4.95% / 1.40% | 2.66% / 0.51% | 3.68% / 1.90% | 1.40% / 0.76% |
| Race_ethnicity | 8.87% / 3.58% | 5.61% / 2.18% | 11.34% / 3.75% | 7.50% / 2.70% | 12.56% / 5.32% | 8.66% / 3.40% |
| Race_x_SES | 5.41% / 3.92% | 3.05% / 1.99% | 5.32% / 4.62% | 3.98% / 3.35% | 6.02% / 3.98% | 3.42% / 2.26% |
| Race_x_gender | 7.38% / 1.60% | 5.51% / 1.07% | 7.81% / 1.85% | 5.68% / 1.42% | 9.21% / 2.43% | 6.65% / 1.79% |
| Religion | 5.17% / 3.17% | 1.17% / 1.50% | 5.50% / 3.83% | 3.50% / 1.50% | 17.67% / 9.50% | 12.50% / 6.33% |
| SES | 4.90% / 1.02% | 3.00% / 0.55% | 5.74% / 1.02% | 3.15% / 0.55% | 7.84% / 1.22% | 3.55% / 0.47% |
| Sexual_orientation | 4.86% / 3.24% | 4.86% / 3.01% | 6.48% / 2.55% | 4.40% / 1.62% | 4.40% / 1.39% | 3.94% / 0.00% |

*Legend — Cells show `ambig% / disambig%` (S/A/U state changes from `direction.csv`).*

</details>

## 4. Neuron and Changed-Item Overlap Between Sets

**Why this check:** tests whether distinct target categories share the same high-BiasScore neurons and whether different interventions flip the same underlying items (pairwise shared neurons, neuron Jaccard, and changed-item Jaccard).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `real_gender20` vs. `real_gender5` | 5 / 20,5 | 0.250 | 0.581 | Gender_identity: 0.871 | Gender_identity: 0.871 | L27.9621, L29.4153, L29.10805, L31.3907, L31.8000 |
| `real_religion20` vs. `real_religion5` | 5 / 20,5 | 0.250 | 0.498 | Religion: 0.586 | Religion: 0.586 | L25.4990, L31.720, L31.1724, L31.7356, L31.10538 |
| `real_race20` vs. `real_race5` | 5 / 20,5 | 0.250 | 0.468 | Race_ethnicity: 0.508 | Race_ethnicity: 0.508 | L31.5601, L31.6742, L31.6990, L31.7042, L31.7356 |
| `real_race20` vs. `real_religion20` | 5 / 20,20 | 0.143 | 0.179 | Race_ethnicity: 0.156 | Religion: 0.165 | L31.230, L31.3778, L31.6742, L31.7295, L31.7356 |
| `real_gender20` vs. `real_religion20` | 2 / 20,20 | 0.053 | 0.238 | Gender_identity: 0.196 | Religion: 0.203 | L31.65, L31.10813 |
| `real_religion20` vs. `real_race5` | 2 / 20,5 | 0.087 | 0.134 | Religion: 0.055 | Race_ethnicity: 0.108 | L31.6742, L31.7356 |
| `real_gender20` vs. `real_race20` | 1 / 20,20 | 0.026 | 0.193 | Gender_identity: 0.142 | Race_ethnicity: 0.266 | L31.3907 |
| `real_race20` vs. `real_gender5` | 1 / 20,5 | 0.042 | 0.182 | Race_ethnicity: 0.288 | Gender_identity: 0.141 | L31.3907 |
| `real_race20` vs. `real_religion5` | 1 / 20,5 | 0.042 | 0.139 | Race_ethnicity: 0.111 | Religion: 0.158 | L31.7356 |
| `real_race5` vs. `real_religion5` | 1 / 5,5 | 0.111 | 0.127 | Race_ethnicity: 0.104 | Religion: 0.067 | L31.7356 |
| `real_gender20` vs. `real_race5` | 0 / 20,5 | 0.000 | 0.173 | Gender_identity: 0.078 | Race_ethnicity: 0.279 | - |
| `real_gender20` vs. `real_religion5` | 0 / 20,5 | 0.000 | 0.172 | Gender_identity: 0.083 | Religion: 0.164 | - |
| `real_gender5` vs. `real_race5` | 0 / 5,5 | 0.000 | 0.168 | Gender_identity: 0.076 | Race_ethnicity: 0.346 | - |
| `real_religion20` vs. `real_gender5` | 0 / 20,5 | 0.000 | 0.155 | Religion: 0.053 | Gender_identity: 0.195 | - |
| `real_gender5` vs. `real_religion5` | 0 / 5,5 | 0.000 | 0.121 | Gender_identity: 0.082 | Religion: 0.057 | - |

*Legend — **Shared Neurons**: identical `(layer, index)` neurons (`shared / size_A, size_B`); **Neuron Jaccard**: `|A ∩ B| / |A ∪ B|` over zeroed neurons; **Items Jacc (All / Target A / Target B)**: Jaccard overlap of changed items across all items or restricted to each set's target category.*

## 5. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

**Why this table:** reports secondary directional metrics (BBQ/EsBBQ bias-score deltas, stereotype transitions `A→S`/`S→A`, and accuracy transitions) for descriptive completeness; these are secondary to answer-change localization.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | Disamb Bias Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: |
| Gender_identity | `real_gender20` | ambig | 688 / 2,836 (24.26%) | 323 / 365 | -1.48% | 0 / 0 | - |
| Gender_identity | `real_gender20` | disambig | 239 / 2,836 (8.43%) | 116 / 123 | -0.25% | 129 / 110 | -0.49 [-2.54, 1.76] |
| Gender_identity | `real_gender5` | ambig | 684 / 2,836 (24.12%) | 323 / 361 | -1.34% | 0 / 0 | - |
| Gender_identity | `real_gender5` | disambig | 228 / 2,836 (8.04%) | 107 / 121 | -0.49% | 123 / 105 | -0.99 [-2.96, 1.20] |
| Gender_identity | `real_race20` | ambig | 178 / 2,836 (6.28%) | 91 / 87 | 0.14% | 0 / 0 | - |
| Gender_identity | `real_race20` | disambig | 65 / 2,836 (2.29%) | 34 / 31 | 0.11% | 30 / 35 | 0.21 [-0.85, 1.27] |
| Gender_identity | `real_race5` | ambig | 156 / 2,836 (5.50%) | 79 / 77 | 0.07% | 0 / 0 | - |
| Gender_identity | `real_race5` | disambig | 59 / 2,836 (2.08%) | 28 / 31 | -0.11% | 30 / 29 | -0.21 [-1.20, 0.85] |
| Gender_identity | `real_religion20` | ambig | 171 / 2,836 (6.03%) | 83 / 88 | -0.18% | 0 / 0 | - |
| Gender_identity | `real_religion20` | disambig | 61 / 2,836 (2.15%) | 31 / 30 | 0.04% | 28 / 33 | 0.07 [-0.99, 1.13] |
| Gender_identity | `real_religion5` | ambig | 96 / 2,836 (3.39%) | 53 / 43 | 0.35% | 0 / 0 | - |
| Gender_identity | `real_religion5` | disambig | 31 / 2,836 (1.09%) | 13 / 18 | -0.18% | 15 / 16 | -0.35 [-1.13, 0.35] |
| Race_ethnicity | `real_gender20` | ambig | 305 / 3,440 (8.87%) | 144 / 146 | -0.06% | 15 / 0 | - |
| Race_ethnicity | `real_gender20` | disambig | 123 / 3,440 (3.58%) | 48 / 73 | -0.73% | 72 / 51 | -1.46 [-2.68, -0.26] |
| Race_ethnicity | `real_gender5` | ambig | 193 / 3,440 (5.61%) | 90 / 94 | -0.12% | 8 / 1 | - |
| Race_ethnicity | `real_gender5` | disambig | 75 / 3,440 (2.18%) | 27 / 47 | -0.58% | 41 / 34 | -1.19 [-2.21, -0.26] |
| Race_ethnicity | `real_race20` | ambig | 390 / 3,440 (11.34%) | 174 / 192 | -0.52% | 0 / 24 | - |
| Race_ethnicity | `real_race20` | disambig | 129 / 3,440 (3.75%) | 49 / 78 | -0.84% | 70 / 59 | -1.75 [-3.00, -0.49] |
| Race_ethnicity | `real_race5` | ambig | 258 / 3,440 (7.50%) | 128 / 123 | 0.15% | 1 / 6 | - |
| Race_ethnicity | `real_race5` | disambig | 93 / 3,440 (2.70%) | 34 / 58 | -0.70% | 51 / 42 | -1.43 [-2.53, -0.38] |
| Race_ethnicity | `real_religion20` | ambig | 432 / 3,440 (12.56%) | 225 / 197 | 0.81% | 10 / 0 | - |
| Race_ethnicity | `real_religion20` | disambig | 183 / 3,440 (5.32%) | 79 / 100 | -0.61% | 116 / 67 | -1.17 [-2.76, 0.41] |
| Race_ethnicity | `real_religion5` | ambig | 298 / 3,440 (8.66%) | 149 / 141 | 0.23% | 5 / 3 | - |
| Race_ethnicity | `real_religion5` | disambig | 117 / 3,440 (3.40%) | 52 / 63 | -0.32% | 69 / 48 | -0.64 [-1.89, 0.58] |
| Religion | `real_gender20` | ambig | 31 / 600 (5.17%) | 15 / 11 | 0.67% | 5 / 0 | - |
| Religion | `real_gender20` | disambig | 19 / 600 (3.17%) | 7 / 8 | -0.17% | 9 / 9 | -0.08 [-2.66, 2.75] |
| Religion | `real_gender5` | ambig | 7 / 600 (1.17%) | 5 / 2 | 0.50% | 0 / 0 | - |
| Religion | `real_gender5` | disambig | 9 / 600 (1.50%) | 6 / 2 | 0.67% | 6 / 2 | 1.55 [-0.35, 3.51] |
| Religion | `real_race20` | ambig | 33 / 600 (5.50%) | 20 / 11 | 1.50% | 0 / 2 | - |
| Religion | `real_race20` | disambig | 23 / 600 (3.83%) | 10 / 12 | -0.33% | 15 / 8 | -0.51 [-3.87, 2.66] |
| Religion | `real_race5` | ambig | 21 / 600 (3.50%) | 12 / 8 | 0.67% | 0 / 1 | - |
| Religion | `real_race5` | disambig | 9 / 600 (1.50%) | 3 / 4 | -0.17% | 6 / 2 | 0.00 [-1.93, 1.93] |
| Religion | `real_religion20` | ambig | 106 / 600 (17.67%) | 50 / 52 | -0.33% | 4 / 0 | - |
| Religion | `real_religion20` | disambig | 57 / 600 (9.50%) | 27 / 26 | 0.17% | 44 / 12 | 0.27 [-4.74, 5.57] |
| Religion | `real_religion5` | ambig | 75 / 600 (12.50%) | 35 / 38 | -0.50% | 2 / 0 | - |
| Religion | `real_religion5` | disambig | 38 / 600 (6.33%) | 17 / 21 | -0.67% | 28 / 10 | -1.40 [-5.91, 3.11] |

*Legend — **A→S / S→A**: flips from anti-stereotype to stereotyped person and vice versa; **Net→Stereo**: `(A→S − S→A) / n_items` (`%`); **Corr→Inc / Inc→Corr**: accuracy transitions; **Bias Δ (95% CI)**: change in benchmark bias score vs. baseline with paired item-bootstrap CI.*

## 6. Baseline Log-Likelihood Margin Mechanism

**Why this check:** verifies whether changed answers come selectively from baseline-undecided items (low `top − second` margin) rather than confident predictions, comparing median margins and flip rates across deciles 1–10.

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `real_gender20` | real | 3,860 | 0.250 | 1.750 |
| `real_race20` | real | 3,131 | 0.188 | 1.719 |
| `real_religion20` | real | 3,866 | 0.250 | 1.746 |
| `real_gender5` | real | 2,953 | 0.250 | 1.703 |
| `real_race5` | real | 2,255 | 0.133 | 1.672 |
| `real_religion5` | real | 2,471 | 0.188 | 1.688 |
| `rand_gender20 (mean n=5)` | random_family_mean | 641.8 | 0.059 | 1.620 |
| `rand_religion20 (mean n=5)` | random_family_mean | 693.0 | 0.064 | 1.621 |

| Decile | Margin Range | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `rand_gender20 (mean n=5)` | `rand_religion20 (mean n=5)` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.250] | 34.74% | 26.24% | 32.91% | 26.58% | 34.80% | 25.23% | 10.84% | 11.58% |
| 2 | [0.250, 0.531] | 18.36% | 14.07% | 13.27% | 8.22% | 15.44% | 7.06% | 0.11% | 0.27% |
| 3 | [0.531, 0.844] | 9.51% | 8.33% | 5.32% | 3.18% | 7.39% | 4.62% | 0.01% | 0.00% |
| 4 | [0.844, 1.184] | 2.92% | 1.85% | 1.74% | 0.51% | 4.17% | 3.18% | 0.01% | 0.00% |
| 5 | [1.188, 1.594] | 0.44% | 0.00% | 0.26% | 0.05% | 2.65% | 1.66% | 0.00% | 0.00% |
| 6 | [1.594, 2.109] | 0.02% | 0.00% | 0.03% | 0.00% | 1.32% | 0.46% | 0.00% | 0.00% |
| 7 | [2.109, 2.805] | 0.00% | 0.00% | 0.00% | 0.00% | 0.32% | 0.03% | 0.00% | 0.00% |
| 8 | [2.809, 3.859] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 9 | [3.859, 5.445] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 10 | [5.445, 13.469] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) of changed vs. unchanged items; **Decile**: baseline-margin decile (`1` = smallest margin / most undecided, `10` = largest margin).*

## 7. Granular Breakdowns (Demographic Groups and Topics)

**Why this table:** checks whether answer changes concentrate in specific demographic subgroups (`answer_info`) or social topics (`relevant_social_value`, EsBBQ) within each target category.

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender_identity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | 4,616 | 17.85% | 17.59% | 4.74% | 4.18% | 4.33% | 2.23% |
| M | 4,616 | 17.85% | 17.59% | 4.74% | 4.18% | 4.33% | 2.23% |
| boy | 40 | 20.00% | 20.00% | 2.50% | 2.50% | 5.00% | 2.50% |
| girl | 40 | 20.00% | 20.00% | 2.50% | 2.50% | 5.00% | 7.50% |
| man | 152 | 21.05% | 21.05% | 1.97% | 2.63% | 1.97% | 4.61% |
| nonTrans | 464 | 1.94% | 1.51% | 1.51% | 1.72% | 2.80% | 1.51% |
| nonTrans_F | 200 | 13.00% | 13.00% | 3.50% | 3.00% | 4.50% | 2.50% |
| nonTrans_M | 200 | 14.00% | 13.50% | 4.00% | 2.50% | 3.50% | 2.00% |
| trans | 464 | 1.94% | 1.51% | 1.51% | 1.72% | 2.80% | 1.51% |
| trans_F | 200 | 14.00% | 13.50% | 4.00% | 2.50% | 3.50% | 2.00% |
| trans_M | 200 | 13.00% | 13.00% | 3.50% | 3.00% | 4.50% | 2.50% |
| woman | 152 | 22.37% | 22.37% | 0.66% | 1.32% | 0.66% | 4.61% |

### Race_ethnicity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| African | 200 | 3.50% | 2.50% | 11.00% | 3.50% | 10.00% | 5.50% |
| African American | 520 | 4.81% | 1.35% | 6.54% | 2.88% | 7.31% | 5.58% |
| Arab | 208 | 8.17% | 2.40% | 9.62% | 3.37% | 26.92% | 24.04% |
| Asian | 312 | 3.53% | 1.92% | 13.14% | 1.92% | 8.65% | 6.09% |
| Black | 520 | 5.00% | 3.65% | 6.92% | 3.08% | 9.04% | 6.54% |
| Caucasian | 128 | 6.25% | 0.78% | 7.03% | 3.12% | 7.81% | 7.81% |
| European | 184 | 4.89% | 1.09% | 9.24% | 2.72% | 8.15% | 3.26% |
| F-Arab | 1,056 | 6.91% | 4.83% | 6.06% | 4.73% | 13.45% | 7.48% |
| F-Asian | 904 | 9.29% | 7.85% | 8.19% | 5.86% | 5.53% | 2.88% |
| F-Black | 1,904 | 9.40% | 7.77% | 9.24% | 6.83% | 8.82% | 3.62% |
| F-Latino | 1,112 | 7.28% | 6.29% | 7.37% | 4.86% | 8.54% | 4.68% |
| F-White | 784 | 12.37% | 10.97% | 12.76% | 11.86% | 5.99% | 1.79% |
| Hispanic | 328 | 12.50% | 1.52% | 4.88% | 2.13% | 7.93% | 8.23% |
| Jewish | 168 | 4.76% | 0.00% | 7.74% | 2.38% | 4.76% | 3.57% |
| Latino | 304 | 3.29% | 1.64% | 4.93% | 2.63% | 7.24% | 2.96% |
| M-Arab | 824 | 3.88% | 0.97% | 6.31% | 4.85% | 18.57% | 15.90% |
| M-Asian | 848 | 2.48% | 0.59% | 5.66% | 5.90% | 7.90% | 6.37% |
| M-Black | 1,032 | 2.23% | 0.58% | 5.43% | 4.26% | 5.23% | 4.36% |
| M-Latino | 936 | 4.06% | 0.85% | 7.91% | 5.56% | 6.94% | 6.20% |
| M-White | 600 | 3.33% | 1.17% | 5.33% | 5.67% | 5.50% | 5.00% |
| Middle Eastern | 224 | 5.36% | 2.23% | 6.70% | 3.12% | 14.29% | 16.52% |
| Native American | 240 | 5.83% | 3.75% | 7.08% | 2.08% | 11.25% | 6.67% |
| Roma | 168 | 3.57% | 1.19% | 3.57% | 2.38% | 6.55% | 5.36% |
| South American | 136 | 5.88% | 0.74% | 7.35% | 1.47% | 7.35% | 2.21% |
| White | 120 | 5.00% | 3.33% | 7.50% | 4.17% | 5.83% | 5.00% |

### Religion — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atheist | 288 | 4.17% | 2.43% | 2.43% | 2.43% | 10.07% | 5.90% |
| Buddhist | 112 | 2.68% | 2.68% | 8.93% | 0.89% | 11.61% | 3.57% |
| Catholic | 264 | 3.79% | 1.14% | 5.68% | 3.03% | 8.71% | 6.44% |
| Christian | 312 | 5.45% | 0.32% | 2.88% | 2.88% | 8.01% | 6.09% |
| Hindu | 264 | 6.06% | 0.00% | 5.68% | 1.89% | 15.91% | 7.20% |
| Jewish | 280 | 3.93% | 2.14% | 3.93% | 2.50% | 13.93% | 7.50% |
| Mormon | 264 | 4.92% | 0.76% | 6.06% | 3.79% | 12.88% | 9.09% |
| Muslim | 328 | 3.35% | 1.52% | 3.96% | 1.83% | 25.91% | 23.78% |
| Protestant | 144 | 2.78% | 1.39% | 5.56% | 2.78% | 9.72% | 7.64% |
| Sikh | 144 | 2.08% | 2.08% | 5.56% | 2.08% | 15.28% | 11.11% |

</details>
