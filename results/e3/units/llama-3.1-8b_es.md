# E3 Unit Dossier — `llama-3.1-8b_es` (Llama-3.1-8B, ES, EsBBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **EsBBQ** (27,320 items, 10 categories) · Source: `results/esbbq_zeroed_llama8b/analysis/20261124_141737_items_a100base`.

## 1. Set Summary and Target-Category Localization

**Why this check:** summarizes how strongly each selected (and ablation) neuron set concentrates answer changes (`pred != base_pred`) in its target category versus the mean of the other categories, together with its layer profile and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | ---: | :--- | :--- |
| `exp11` | real | Gender | 20 | 626 / 4,832 (12.96%) | [11.92, 13.91] | 5.31% | 2.44× | 1,853 / 27,320 (6.78%) | A100-SXM4-80GB | L29:2, L30:2, L31:16 |
| `exp13` | real | Religion | 20 | 43 / 648 (6.64%) | [4.63, 8.80] | 5.03% | 1.32× | 1,354 / 27,320 (4.96%) | A100-SXM4-80GB | L21:1, L24:1, L28:1, L31:17 |
| `exp14` | real | RaceEthnicity | 20 | 335 / 3,716 (9.02%) | [8.02, 9.99] | 8.65% | 1.04× | 2,418 / 27,320 (8.85%) | A100-SXM4-80GB | L31:20 |
| `exp10` | real | Gender | 5 | 113 / 4,832 (2.34%) | [1.95, 2.79] | 3.29% | 0.71× | 881 / 27,320 (3.22%) | A100-SXM4-40GB | L29:1, L31:4 |
| `exp12` | real | RaceEthnicity | 5 | 114 / 3,716 (3.07%) | [2.53, 3.63] | 2.84% | 1.08× | 812 / 27,320 (2.97%) | A100-SXM4-40GB | L31:5 |
| `exp2` | real | Religion | 5 | 24 / 648 (3.70%) | [2.16, 5.25] | 2.42% | 1.53× | 644 / 27,320 (2.36%) | A100-SXM4-40GB | L28:1, L31:4 |

*Legend — **Kind**: `real` (selected top-K) or `ablation` (exploratory subset); **Target Flips**: changed answers (`pred != base_pred`) in target category; **95% CI (%)**: paired item-bootstrap interval (`B = 1,000`); **Other Mean**: mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **Layer Profile**: `L<layer>:<count>` of zeroed neurons.*

## 2. Changed Answers Across All Categories (`Set × Category`)

**Why this table:** exposes the full `set × category` flip matrix (`pred != base_pred`, all items pooled; random families summarized by mean) so off-target spillover across individual categories can be inspected directly.

| Category | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 4,068 | 103 (2.53%) | 162 (3.98%) | 121 (2.97%) | 191 (4.70%) | 255 (6.27%) | 454 (11.16%) |
| DisabilityStatus | 2,832 | 45 (1.59%) | 72 (2.54%) | 61 (2.15%) | 134 (4.73%) | 110 (3.88%) | 245 (8.65%) |
| Gender | 4,832 | 71 (1.47%) | 113 (2.34%) | 209 (4.33%) | 221 (4.57%) | 626 (12.96%) | 258 (5.34%) |
| LGBTQIA | 2,000 | 53 (2.65%) | 102 (5.10%) | 70 (3.50%) | 112 (5.60%) | 119 (5.95%) | 227 (11.35%) |
| Nationality | 504 | 15 (2.98%) | 13 (2.58%) | 10 (1.98%) | 30 (5.95%) | 30 (5.95%) | 33 (6.55%) |
| PhysicalAppearance | 3,528 | 76 (2.15%) | 103 (2.92%) | 90 (2.55%) | 168 (4.76%) | 139 (3.94%) | 363 (10.29%) |
| RaceEthnicity | 3,716 | 162 (4.36%) | 169 (4.55%) | 114 (3.07%) | 254 (6.84%) | 326 (8.77%) | 335 (9.02%) |
| Religion | 648 | 24 (3.70%) | 17 (2.62%) | 24 (3.70%) | 43 (6.64%) | 21 (3.24%) | 48 (7.41%) |
| SES | 4,204 | 72 (1.71%) | 101 (2.40%) | 91 (2.16%) | 158 (3.76%) | 170 (4.04%) | 374 (8.90%) |
| SpanishRegion | 988 | 23 (2.33%) | 29 (2.94%) | 22 (2.23%) | 43 (4.35%) | 57 (5.77%) | 81 (8.20%) |

*Legend — Cells show `changed (flips_pct)` (`pred != base_pred`, pooled `ambig + disambig`); random-family columns show `mean% [min–max]` across seeds.*

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 3.56% / 2.05% | 4.41% / 3.78% | 3.79% / 2.59% | 6.66% / 3.78% | 8.28% / 5.33% | 10.14% / 11.64% |
| DisabilityStatus | 2.05% / 1.37% | 2.48% / 2.57% | 2.69% / 1.89% | 5.82% / 4.20% | 4.42% / 3.62% | 12.39% / 6.83% |
| Gender | 1.93% / 1.26% | 3.46% / 1.83% | 6.18% / 3.49% | 5.12% / 4.33% | 18.35% / 10.52% | 7.78% / 4.24% |
| LGBTQIA | 2.43% / 2.74% | 6.08% / 4.71% | 3.12% / 3.65% | 5.90% / 5.48% | 6.42% / 5.76% | 13.19% / 10.60% |
| Nationality | 3.57% / 2.68% | 3.57% / 2.08% | 1.19% / 2.38% | 7.14% / 5.36% | 9.52% / 4.17% | 8.93% / 5.36% |
| PhysicalAppearance | 3.66% / 1.40% | 5.36% / 1.70% | 3.66% / 2.00% | 8.76% / 2.76% | 7.99% / 1.91% | 14.88% / 7.99% |
| RaceEthnicity | 5.37% / 3.86% | 6.68% / 3.50% | 4.07% / 2.57% | 8.63% / 5.95% | 11.89% / 7.23% | 10.91% / 8.08% |
| Religion | 3.24% / 3.94% | 2.78% / 2.55% | 5.56% / 2.78% | 5.09% / 7.41% | 3.24% / 3.24% | 11.57% / 5.32% |
| SES | 2.10% / 1.52% | 2.54% / 2.34% | 2.61% / 1.95% | 4.42% / 3.43% | 5.22% / 3.47% | 9.86% / 8.43% |
| SpanishRegion | 5.56% / 0.75% | 6.17% / 1.36% | 4.63% / 1.05% | 8.95% / 2.11% | 13.58% / 1.96% | 18.83% / 3.01% |

*Legend — Cells show `ambig% / disambig%` (S/A/U state changes from `direction.csv`).*

</details>

## 3. Neuron and Changed-Item Overlap Between Sets

**Why this check:** tests whether distinct target categories share the same high-BiasScore neurons and whether different interventions flip the same underlying items (pairwise shared neurons, neuron Jaccard, and changed-item Jaccard).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `exp13` vs. `exp14` | 6 / 20,20 | 0.176 | 0.265 | Religion: 0.138 | RaceEthnicity: 0.437 | L31.273, L31.2398, L31.5897, L31.6950, L31.13336, L31.14276 |
| `exp11` vs. `exp10` | 5 / 20,5 | 0.250 | 0.266 | Gender: 0.088 | Gender: 0.088 | L29.8403, L31.77, L31.2566, L31.7783, L31.13336 |
| `exp13` vs. `exp2` | 5 / 20,5 | 0.250 | 0.253 | Religion: 0.457 | Religion: 0.457 | L28.7507, L31.33, L31.5897, L31.6950, L31.12768 |
| `exp14` vs. `exp12` | 5 / 20,5 | 0.250 | 0.177 | RaceEthnicity: 0.194 | RaceEthnicity: 0.194 | L31.2398, L31.5897, L31.8118, L31.10439, L31.12004 |
| `exp11` vs. `exp14` | 4 / 20,20 | 0.111 | 0.200 | Gender: 0.119 | RaceEthnicity: 0.078 | L31.1695, L31.2398, L31.12252, L31.13336 |
| `exp11` vs. `exp13` | 4 / 20,20 | 0.111 | 0.156 | Gender: 0.022 | Religion: 0.362 | L31.2398, L31.2564, L31.6466, L31.13336 |
| `exp13` vs. `exp12` | 2 / 20,5 | 0.087 | 0.212 | Religion: 0.175 | RaceEthnicity: 0.143 | L31.2398, L31.5897 |
| `exp14` vs. `exp2` | 2 / 20,5 | 0.087 | 0.114 | RaceEthnicity: 0.301 | Religion: 0.143 | L31.5897, L31.6950 |
| `exp13` vs. `exp10` | 1 / 20,5 | 0.042 | 0.238 | Religion: 0.304 | Gender: 0.206 | L31.13336 |
| `exp12` vs. `exp2` | 1 / 5,5 | 0.111 | 0.196 | RaceEthnicity: 0.160 | Religion: 0.143 | L31.5897 |
| `exp14` vs. `exp10` | 1 / 20,5 | 0.042 | 0.178 | RaceEthnicity: 0.105 | Gender: 0.216 | L31.13336 |
| `exp11` vs. `exp12` | 1 / 20,5 | 0.042 | 0.155 | Gender: 0.173 | RaceEthnicity: 0.100 | L31.2398 |
| `exp10` vs. `exp12` | 0 / 5,5 | 0.000 | 0.227 | Gender: 0.253 | RaceEthnicity: 0.137 | - |
| `exp10` vs. `exp2` | 0 / 5,5 | 0.000 | 0.155 | Gender: 0.102 | Religion: 0.519 | - |
| `exp11` vs. `exp2` | 0 / 20,5 | 0.000 | 0.116 | Gender: 0.092 | Religion: 0.406 | - |

*Legend — **Shared Neurons**: identical `(layer, index)` neurons (`shared / size_A, size_B`); **Neuron Jaccard**: `|A ∩ B| / |A ∪ B|` over zeroed neurons; **Items Jacc (All / Target A / Target B)**: Jaccard overlap of changed items across all items or restricted to each set's target category.*

## 4. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

**Why this table:** reports secondary directional metrics (BBQ/EsBBQ bias-score deltas, stereotype transitions `A→S`/`S→A`, and accuracy transitions) for descriptive completeness; these are secondary to answer-change localization.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | EsBBQ Bias Δ (95% CI) | BBQ-Formula Disamb Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: | ---: |
| Gender | `exp10` | ambig | 52 / 1,504 (3.46%) | 22 / 29 | -0.47% | 0 / 1 | -0.86 [-2.73, 0.87] | - |
| Gender | `exp10` | disambig | 61 / 3,328 (1.83%) | 31 / 30 | 0.03% | 36 / 25 | 0.06 [-0.90, 0.90] | 0.06 [-0.90, 0.90] |
| Gender | `exp11` | ambig | 276 / 1,504 (18.35%) | 136 / 138 | -0.13% | 0 / 2 | -0.13 [-4.39, 4.19] | - |
| Gender | `exp11` | disambig | 350 / 3,328 (10.52%) | 196 / 152 | 1.32% | 177 / 173 | 2.76 [0.66, 5.01] | 2.71 [0.63, 4.93] |
| Gender | `exp12` | ambig | 93 / 1,504 (6.18%) | 45 / 48 | -0.20% | 0 / 0 | -0.40 [-3.06, 2.00] | - |
| Gender | `exp12` | disambig | 116 / 3,328 (3.49%) | 56 / 60 | -0.12% | 63 / 53 | -0.24 [-1.56, 0.94] | -0.24 [-1.56, 0.96] |
| Gender | `exp13` | ambig | 77 / 1,504 (5.12%) | 33 / 44 | -0.73% | 0 / 0 | -1.46 [-3.86, 0.80] | - |
| Gender | `exp13` | disambig | 144 / 3,328 (4.33%) | 65 / 79 | -0.42% | 77 / 67 | -0.84 [-2.26, 0.46] | -0.84 [-2.28, 0.42] |
| Gender | `exp14` | ambig | 117 / 1,504 (7.78%) | 63 / 51 | 0.80% | 0 / 3 | 1.80 [-1.00, 4.32] | - |
| Gender | `exp14` | disambig | 141 / 3,328 (4.24%) | 71 / 68 | 0.09% | 71 / 70 | 0.30 [-1.12, 1.62] | 0.24 [-1.17, 1.53] |
| Gender | `exp2` | ambig | 29 / 1,504 (1.93%) | 13 / 16 | -0.20% | 0 / 0 | -0.40 [-1.86, 0.93] | - |
| Gender | `exp2` | disambig | 42 / 3,328 (1.26%) | 20 / 22 | -0.06% | 15 / 27 | -0.12 [-0.85, 0.65] | -0.12 [-0.84, 0.66] |
| RaceEthnicity | `exp10` | ambig | 82 / 1,228 (6.68%) | 40 / 34 | 0.49% | 4 / 4 | 0.81 [-1.88, 3.42] | - |
| RaceEthnicity | `exp10` | disambig | 87 / 2,488 (3.50%) | 41 / 45 | -0.16% | 41 / 46 | -0.24 [-1.70, 1.17] | -0.29 [-1.79, 1.17] |
| RaceEthnicity | `exp11` | ambig | 146 / 1,228 (11.89%) | 70 / 60 | 0.81% | 1 / 15 | 1.47 [-2.36, 4.97] | - |
| RaceEthnicity | `exp11` | disambig | 180 / 2,488 (7.23%) | 87 / 85 | 0.08% | 90 / 86 | -0.16 [-2.25, 1.86] | -0.16 [-2.30, 1.96] |
| RaceEthnicity | `exp12` | ambig | 50 / 1,228 (4.07%) | 26 / 17 | 0.73% | 5 / 2 | 1.55 [-0.57, 3.66] | - |
| RaceEthnicity | `exp12` | disambig | 64 / 2,488 (2.57%) | 28 / 33 | -0.20% | 32 / 32 | -0.32 [-1.53, 0.80] | -0.37 [-1.58, 0.75] |
| RaceEthnicity | `exp13` | ambig | 106 / 1,228 (8.63%) | 40 / 52 | -0.98% | 13 / 1 | -1.30 [-4.48, 1.63] | - |
| RaceEthnicity | `exp13` | disambig | 148 / 2,488 (5.95%) | 75 / 68 | 0.28% | 75 / 73 | 0.80 [-1.13, 2.66] | 0.70 [-1.28, 2.58] |
| RaceEthnicity | `exp14` | ambig | 134 / 1,228 (10.91%) | 59 / 55 | 0.33% | 6 / 14 | 0.00 [-3.50, 3.26] | - |
| RaceEthnicity | `exp14` | disambig | 201 / 2,488 (8.08%) | 97 / 89 | 0.32% | 110 / 87 | -0.08 [-2.36, 2.09] | 0.22 [-2.07, 2.41] |
| RaceEthnicity | `exp2` | ambig | 66 / 1,228 (5.37%) | 24 / 38 | -1.14% | 4 / 0 | -2.12 [-4.72, 0.16] | - |
| RaceEthnicity | `exp2` | disambig | 96 / 2,488 (3.86%) | 52 / 41 | 0.44% | 41 / 54 | 0.88 [-0.61, 2.37] | 0.87 [-0.71, 2.42] |
| Religion | `exp10` | ambig | 6 / 216 (2.78%) | 2 / 2 | 0.00% | 2 / 0 | -0.93 [-5.09, 2.78] | - |
| Religion | `exp10` | disambig | 11 / 432 (2.55%) | 5 / 6 | -0.23% | 7 / 4 | -0.46 [-3.38, 2.75] | -0.47 [-3.27, 2.80] |
| Religion | `exp11` | ambig | 7 / 216 (3.24%) | 3 / 2 | 0.46% | 2 / 0 | 0.00 [-4.17, 4.63] | - |
| Religion | `exp11` | disambig | 14 / 432 (3.24%) | 5 / 9 | -0.93% | 7 / 7 | -1.85 [-5.10, 1.75] | -1.87 [-5.14, 1.85] |
| Religion | `exp12` | ambig | 12 / 216 (5.56%) | 4 / 4 | 0.00% | 4 / 0 | 0.00 [-5.56, 5.56] | - |
| Religion | `exp12` | disambig | 12 / 432 (2.78%) | 8 / 4 | 0.93% | 7 / 5 | 1.85 [-1.39, 5.08] | 1.87 [-1.40, 5.12] |
| Religion | `exp13` | ambig | 11 / 216 (5.09%) | 4 / 2 | 0.93% | 5 / 0 | 1.39 [-3.24, 6.48] | - |
| Religion | `exp13` | disambig | 32 / 432 (7.41%) | 11 / 17 | -1.39% | 16 / 16 | -1.85 [-6.85, 3.21] | -2.27 [-7.18, 2.47] |
| Religion | `exp14` | ambig | 25 / 216 (11.57%) | 7 / 3 | 1.85% | 0 / 15 | 5.09 [-1.39, 11.57] | - |
| Religion | `exp14` | disambig | 23 / 432 (5.32%) | 16 / 7 | 2.08% | 13 / 10 | 4.17 [-0.03, 8.41] | 4.21 [0.00, 8.48] |
| Religion | `exp2` | ambig | 7 / 216 (3.24%) | 3 / 2 | 0.46% | 2 / 0 | 0.00 [-4.17, 4.17] | - |
| Religion | `exp2` | disambig | 17 / 432 (3.94%) | 6 / 10 | -0.93% | 9 / 8 | -2.31 [-5.88, 1.41] | -2.09 [-5.60, 1.65] |

*Legend — **A→S / S→A**: flips from anti-stereotype to stereotyped person and vice versa; **Net→Stereo**: `(A→S − S→A) / n_items` (`%`); **Corr→Inc / Inc→Corr**: accuracy transitions; **Bias Δ (95% CI)**: change in benchmark bias score vs. baseline with paired item-bootstrap CI.*

## 5. Baseline Log-Likelihood Margin Mechanism

**Why this check:** verifies whether changed answers come selectively from baseline-undecided items (low `top − second` margin) rather than confident predictions, comparing median margins and flip rates across deciles 1–10.

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `exp2` | real | 644 | 0.062 | 1.000 |
| `exp10` | real | 881 | 0.094 | 1.000 |
| `exp12` | real | 812 | 0.062 | 1.000 |
| `exp13` | real | 1,354 | 0.125 | 1.004 |
| `exp11` | real | 1,853 | 0.125 | 1.031 |
| `exp14` | real | 2,418 | 0.172 | 1.062 |

| Decile | Margin Range | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.141] | 19.84% | 24.93% | 25.81% | 32.65% | 38.10% | 40.15% |
| 2 | [0.141, 0.344] | 2.27% | 5.09% | 3.92% | 12.81% | 17.50% | 26.39% |
| 3 | [0.344, 0.500] | 0.84% | 1.10% | 0.00% | 2.42% | 8.35% | 12.04% |
| 4 | [0.500, 0.750] | 0.44% | 0.26% | 0.00% | 1.10% | 2.31% | 6.00% |
| 5 | [0.750, 0.969] | 0.15% | 0.29% | 0.00% | 0.51% | 1.06% | 2.01% |
| 6 | [0.969, 1.234] | 0.04% | 0.37% | 0.00% | 0.07% | 0.37% | 0.92% |
| 7 | [1.234, 1.516] | 0.00% | 0.15% | 0.00% | 0.00% | 0.11% | 0.62% |
| 8 | [1.516, 1.984] | 0.00% | 0.07% | 0.00% | 0.00% | 0.04% | 0.37% |
| 9 | [1.984, 2.750] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 10 | [2.750, 16.094] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) of changed vs. unchanged items; **Decile**: baseline-margin decile (`1` = smallest margin / most undecided, `10` = largest margin).*

## 6. Granular Breakdowns (Demographic Groups and Topics)

**Why this table:** checks whether answer changes concentrate in specific demographic subgroups (`answer_info`) or social topics (`relevant_social_value`, EsBBQ) within each target category.

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f | 4,832 | 1.47% | 2.34% | 4.33% | 4.57% | 12.96% | 5.34% |
| m | 4,832 | 1.47% | 2.34% | 4.33% | 4.57% | 12.96% | 5.34% |

### RaceEthnicity — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| africano | 124 | 0.81% | 3.23% | 1.61% | 6.45% | 4.03% | 9.68% |
| asiático | 264 | 7.20% | 4.92% | 4.55% | 10.98% | 12.12% | 9.09% |
| blanco | 3,716 | 4.36% | 4.55% | 3.07% | 6.84% | 8.77% | 9.02% |
| de Europa del Este | 396 | 0.76% | 2.53% | 3.79% | 3.79% | 12.63% | 6.31% |
| gitano | 372 | 2.15% | 2.96% | 1.88% | 5.65% | 3.76% | 12.10% |
| indio | 264 | 13.26% | 3.03% | 3.41% | 17.05% | 7.20% | 14.02% |
| latino | 672 | 1.34% | 11.16% | 3.57% | 3.57% | 10.71% | 7.74% |
| negro | 748 | 3.61% | 3.21% | 3.34% | 3.74% | 8.69% | 7.62% |
| árabe | 876 | 6.85% | 2.74% | 2.28% | 9.59% | 7.88% | 9.47% |

### Religion — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ateo | 216 | 1.85% | 0.93% | 1.85% | 4.17% | 2.78% | 7.87% |
| católico | 408 | 3.68% | 2.70% | 3.68% | 5.88% | 2.45% | 8.09% |
| judío | 48 | 4.17% | 2.08% | 4.17% | 10.42% | 2.08% | 4.17% |
| musulmán | 144 | 6.25% | 6.25% | 4.17% | 9.72% | 4.86% | 6.94% |
| no creyente | 216 | 3.24% | 1.39% | 5.09% | 4.63% | 2.78% | 8.80% |
| testigo de Jehová | 264 | 4.17% | 3.03% | 3.79% | 9.09% | 4.55% | 5.68% |

</details>

<details>
<summary>Expand EsBBQ topic breakdown (`relevant_social_value`)</summary>

### Gender — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cocinar | 240 | 1.25% | 2.92% | 7.08% | 6.25% | 18.75% | 5.83% |
| conducción temeraria o imprudente | 60 | 1.67% | 0.00% | 1.67% | 5.00% | 3.33% | 6.67% |
| debilidad, fragilidad, menos fuerza | 60 | 6.67% | 6.67% | 6.67% | 8.33% | 21.67% | 6.67% |
| emocional | 120 | 0.00% | 0.83% | 2.50% | 5.00% | 12.50% | 4.17% |
| gustos asociados a un género: coches | 120 | 1.67% | 0.00% | 0.83% | 1.67% | 7.50% | 0.83% |
| gustos asociados a un género: color azul | 160 | 2.50% | 1.25% | 3.12% | 5.00% | 16.25% | 6.25% |
| gustos asociados a un género: color rosa | 160 | 0.62% | 1.25% | 5.62% | 8.75% | 13.12% | 9.38% |
| gustos asociados a un género: fútbol | 72 | 0.00% | 1.39% | 2.78% | 2.78% | 5.56% | 4.17% |
| gustos asociados a un género: muñecas | 120 | 0.83% | 1.67% | 0.83% | 1.67% | 5.00% | 0.83% |
| inestabilidad emocional | 120 | 0.83% | 4.17% | 4.17% | 8.33% | 13.33% | 6.67% |
| inferioridad | 120 | 1.67% | 1.67% | 7.50% | 5.83% | 12.50% | 12.50% |
| ira, agresividad, violencia | 160 | 1.88% | 3.12% | 4.38% | 1.88% | 14.37% | 5.00% |
| mala conducción | 180 | 1.67% | 1.67% | 3.33% | 5.00% | 10.00% | 5.00% |
| malas entre ellas | 36 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.33% |
| mayor carga familiar | 240 | 2.08% | 3.33% | 6.67% | 5.83% | 13.33% | 5.83% |
| menor inteligencia | 60 | 1.67% | 1.67% | 6.67% | 5.00% | 11.67% | 6.67% |
| menor proactividad, liderazgo | 60 | 0.00% | 6.67% | 6.67% | 3.33% | 8.33% | 5.00% |
| no llorar | 180 | 2.22% | 2.22% | 6.67% | 4.44% | 16.11% | 6.11% |
| no ser alto cargo | 120 | 2.50% | 4.17% | 3.33% | 2.50% | 10.00% | 4.17% |
| peor con la tecnología | 120 | 1.67% | 1.67% | 2.50% | 3.33% | 8.33% | 4.17% |
| peor en STEM | 240 | 2.08% | 4.17% | 6.67% | 5.83% | 17.50% | 7.92% |
| peor en deportes | 72 | 2.78% | 0.00% | 5.56% | 0.00% | 13.89% | 4.17% |
| peor en matemáticas | 280 | 1.43% | 2.14% | 4.64% | 4.64% | 12.50% | 3.93% |
| persona cerrada emocionalmente, falta de empatía | 240 | 0.83% | 2.92% | 3.75% | 4.17% | 12.50% | 2.50% |
| persona exagerada | 180 | 1.11% | 0.56% | 3.33% | 5.00% | 16.11% | 5.00% |
| priorizar la familia sobre la carrera profesional, mayor carga familiar | 200 | 0.50% | 1.50% | 2.00% | 2.50% | 9.50% | 2.00% |
| priorizar la maternidad sobre la carrera profesional | 100 | 2.00% | 3.00% | 4.00% | 8.00% | 14.00% | 2.00% |
| profesión feminizada: maestra | 120 | 0.83% | 1.67% | 1.67% | 3.33% | 6.67% | 4.17% |
| profesión feminizada: secretaria | 232 | 1.29% | 4.31% | 6.03% | 5.17% | 18.97% | 6.47% |
| profesión masculinizada: informática | 120 | 2.50% | 0.83% | 3.33% | 4.17% | 10.83% | 5.00% |
| profesión masculinizada: ingeniería | 120 | 0.83% | 2.50% | 3.33% | 2.50% | 8.33% | 5.00% |
| querer tener hijos | 120 | 1.67% | 2.50% | 4.17% | 3.33% | 16.67% | 4.17% |
| sensible | 60 | 0.00% | 0.00% | 3.33% | 1.67% | 25.00% | 1.67% |
| violencia de género (maltratador) | 180 | 1.67% | 2.22% | 5.00% | 7.22% | 11.67% | 11.11% |
| víctima de violencia de género | 60 | 0.00% | 3.33% | 0.00% | 0.00% | 13.33% | 6.67% |

### RaceEthnicity — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acoso / agresión sexual | 12 | 0.00% | 25.00% | 0.00% | 8.33% | 25.00% | 50.00% |
| alcoholismo | 72 | 0.00% | 12.50% | 1.39% | 2.78% | 15.28% | 8.33% |
| analfabetismo tecnológico | 24 | 0.00% | 12.50% | 4.17% | 8.33% | 12.50% | 16.67% |
| ayudas sociales | 504 | 6.15% | 4.56% | 4.56% | 8.93% | 11.51% | 9.33% |
| bajo rendimiento académico | 432 | 4.63% | 1.39% | 2.08% | 6.02% | 7.64% | 6.94% |
| buscar dinero en relaciones | 72 | 2.78% | 18.06% | 4.17% | 6.94% | 18.06% | 6.94% |
| casarse y tener relaciones sexuales a edades tempranas | 24 | 8.33% | 0.00% | 0.00% | 4.17% | 0.00% | 4.17% |
| consumo de drogas | 96 | 2.08% | 6.25% | 8.33% | 2.08% | 5.21% | 13.54% |
| delincuencia, robar | 216 | 2.78% | 7.41% | 2.78% | 5.56% | 8.33% | 12.50% |
| docilidad | 72 | 6.94% | 1.39% | 0.00% | 9.72% | 11.11% | 9.72% |
| español como lengua extranjera | 168 | 3.57% | 7.14% | 6.55% | 8.33% | 8.93% | 14.88% |
| falta de acceso a la tecnología | 24 | 0.00% | 0.00% | 0.00% | 4.17% | 0.00% | 0.00% |
| falta de derechos de la mujer, machismo | 144 | 8.33% | 0.00% | 1.39% | 6.94% | 6.25% | 7.64% |
| hipersexualización, prostitución | 72 | 1.39% | 9.72% | 0.00% | 2.78% | 18.06% | 2.78% |
| inmigración ilegal | 132 | 2.27% | 6.06% | 2.27% | 3.03% | 12.12% | 6.82% |
| ira, agresividad, violencia | 120 | 1.67% | 0.83% | 2.50% | 5.00% | 9.17% | 5.83% |
| mal comportamiento, persona problemática | 120 | 3.33% | 9.17% | 4.17% | 10.00% | 9.17% | 12.50% |
| mal olor | 72 | 2.78% | 4.17% | 4.17% | 8.33% | 5.56% | 9.72% |
| mala gente, no ser de fiar | 24 | 0.00% | 0.00% | 4.17% | 4.17% | 4.17% | 12.50% |
| mayor presión parental | 48 | 0.00% | 2.08% | 0.00% | 4.17% | 2.08% | 0.00% |
| meritocracia | 744 | 6.32% | 3.49% | 3.36% | 8.20% | 10.08% | 8.33% |
| pobreza | 152 | 1.32% | 0.66% | 1.97% | 3.95% | 2.63% | 10.53% |
| promiscuidad | 36 | 0.00% | 2.78% | 0.00% | 2.78% | 2.78% | 0.00% |
| sumisión | 24 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| terrorismo | 168 | 7.14% | 4.17% | 1.19% | 8.93% | 6.55% | 9.52% |
| vagancia, no querer trabajar | 96 | 1.04% | 6.25% | 3.12% | 2.08% | 1.04% | 8.33% |
| violencia de género | 48 | 4.17% | 10.42% | 4.17% | 16.67% | 2.08% | 16.67% |

### Religion — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abuso sexual infantil | 48 | 8.33% | 8.33% | 8.33% | 10.42% | 8.33% | 6.25% |
| anticiencia | 336 | 2.68% | 1.49% | 4.17% | 4.76% | 2.68% | 8.63% |
| avaricia | 48 | 4.17% | 2.08% | 4.17% | 10.42% | 2.08% | 4.17% |
| celebración de tradiciones religiosas | 48 | 8.33% | 8.33% | 2.08% | 14.58% | 4.17% | 0.00% |
| falta de derechos de la mujer, machismo | 60 | 1.67% | 0.00% | 0.00% | 3.33% | 3.33% | 3.33% |
| persona sentenciosa, moralista | 48 | 2.08% | 0.00% | 2.08% | 4.17% | 4.17% | 10.42% |
| poligamia | 24 | 4.17% | 8.33% | 4.17% | 8.33% | 4.17% | 12.50% |
| terrorismo | 36 | 5.56% | 2.78% | 2.78% | 11.11% | 0.00% | 11.11% |

</details>
