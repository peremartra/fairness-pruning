# E3 Unit Dossier — `llama-3.1-8b_en` (Llama-3.1-8B, EN, BBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **BBQ** (58,492 items, 11 categories) · Source: `results/e3/llama-3.1-8b_en/analysis/20261108_144541_items`.

## 1. Set Summary and Target-Category Localization

Changed answers (`pred != base_pred`, `changed / n_items`, `%`) for each real and ablation set in its target category versus the mean of the other categories, with layer distribution and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | Rank | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | :--- | :--- |
| `real_gender20` | real | Gender_identity | 20 | 1,361 / 5,672 (24.00%) | [22.88, 25.09] | 6.28% | 3.82× | 1/11 | 5,291 / 58,492 (9.05%) | A100-SXM4-80GB | L27:2, L30:3, L31:15 |
| `real_race20` | real | Race_ethnicity | 20 | 519 / 6,880 (7.54%) | [6.96, 8.17] | 5.43% | 1.39× | 3/11 | 3,399 / 58,492 (5.81%) | A100-SXM4-80GB | L31:20 |
| `real_religion20` | real | Religion | 20 | 120 / 1,200 (10.00%) | [8.25, 11.67] | 6.50% | 1.54× | 2/11 | 4,027 / 58,492 (6.88%) | A100-SXM4-80GB | L30:1, L31:19 |
| `real_gender5` | real | Gender_identity | 5 | 1,323 / 5,672 (23.33%) | [22.23, 24.40] | 4.39% | 5.32× | 1/11 | 4,598 / 58,492 (7.86%) | A100-SXM4-80GB | L27:1, L31:4 |
| `real_race5` | real | Race_ethnicity | 5 | 415 / 6,880 (6.03%) | [5.49, 6.60] | 3.78% | 1.59× | 3/11 | 2,375 / 58,492 (4.06%) | A100-SXM4-80GB | L31:5 |
| `real_religion5` | real | Religion | 5 | 118 / 1,200 (9.83%) | [8.17, 11.42] | 3.55% | 2.77× | 1/11 | 2,487 / 58,492 (4.25%) | A100-SXM4-80GB | L31:5 |
| `abl_race4` | ablation | Race_ethnicity | 4 | 155 / 6,880 (2.25%) | [1.92, 2.62] | 1.86% | 1.21× | 3/11 | 1,148 / 58,492 (1.96%) | A100-SXM4-40GB | L31:4 |
| `abl_religion4` | ablation | Religion | 4 | 74 / 1,200 (6.17%) | [4.83, 7.50] | 2.24% | 2.75× | 1/11 | 1,434 / 58,492 (2.45%) | A100-SXM4-40GB | L31:4 |
| `abl_shared1` | ablation | Race+Religion | 1 | - | - | - | - | - | 2,051 / 58,492 (3.51%) | A100-SXM4-40GB | L31:1 |

*Legend — **Kind**: `real` (selected top-K) or `ablation` (exploratory subset); **Target Flips**: changed answers (`pred != base_pred`) in target category; **95% CI (%)**: paired item-bootstrap interval (`B = 1,000`); **Other Mean**: mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **Rank**: target position among all categories; **Layer Profile**: `L<layer>:<count>` of zeroed neurons.*

## 2. Selected vs. Layer-Matched Random Neurons

Direct contrast of selected sets against uniform random neuron sets drawn from the exact same layers.

| Comparison | Target Cat. | Selected Flips | Random Flips (min–max, mean) | Real/Rand | Rank | Other Cats (Real vs. Rand) | Ratio (Real vs. Rand) | GPUs |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | :--- |
| Gender·20 (seeds 0-4) | Gender_identity | 1,361 / 5,672 (24.00%) | 41–76 (0.72–1.34%, mean 0.93%) | 25.68× | 1 of 6 | 6.28% vs. 1.23% | 3.82× vs. 0.75× | A100-SXM4-80GB / A100-SXM4-40GB |
| Religion·20 (seeds 0-9) | Religion | 120 / 1,200 (10.00%) | 10–38 (0.83–3.17%, mean 1.37%) | 7.32× | 1 of 11 | 6.50% vs. 1.41% | 1.54× vs. 0.97× | A100-SXM4-80GB / A100-SXM4-40GB |

*Legend — **Selected/Random Flips**: target-category changed answers (`pred != base_pred`); **Real/Rand**: `Selected % / Random mean %`; **Rank**: position of selected set among selected + random (`1` = highest); **Other Cats / Ratio**: mean flip `%` across non-target categories and `Target / Other` ratio.*

## 3. Changed Answers Across All Categories (`Set × Category`)

Full matrix of changed answers (`pred != base_pred`, `changed (flips_pct)`) across every benchmark category (all items pooled; random families summarized by mean).

| Category | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `abl_religion4` | `abl_shared1` | `abl_race4` | `rand_gender20` (mean) | `rand_religion20` (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 3,680 | 183 (4.97%) | 106 (2.88%) | 171 (4.65%) | 76 (2.07%) | 194 (5.27%) | 83 (2.26%) | 97 (2.64%) | 41 (1.11%) | 84 (2.28%) | 1.66% [1.41–2.26] | 2.20% [1.30–6.28] |
| Disability_status | 1,556 | 49 (3.15%) | 18 (1.16%) | 31 (1.99%) | 17 (1.09%) | 57 (3.66%) | 16 (1.03%) | 16 (1.03%) | 12 (0.77%) | 17 (1.09%) | 0.67% [0.45–0.90] | 0.81% [0.39–1.48] |
| Gender_identity | 5,672 | 1,361 (24.00%) | 1,323 (23.33%) | 208 (3.67%) | 105 (1.85%) | 178 (3.14%) | 105 (1.85%) | 81 (1.43%) | 60 (1.06%) | 89 (1.57%) | 0.93% [0.72–1.34] | 1.17% [0.72–2.13] |
| Nationality | 3,080 | 115 (3.73%) | 68 (2.21%) | 313 (10.16%) | 274 (8.90%) | 367 (11.92%) | 236 (7.66%) | 63 (2.05%) | 227 (7.37%) | 110 (3.57%) | 1.42% [1.10–1.92] | 1.40% [0.94–2.31] |
| Physical_appearance | 1,576 | 115 (7.30%) | 57 (3.62%) | 62 (3.93%) | 29 (1.84%) | 76 (4.82%) | 32 (2.03%) | 36 (2.28%) | 16 (1.02%) | 23 (1.46%) | 1.40% [1.27–1.65] | 1.61% [1.08–2.03] |
| Race_ethnicity | 6,880 | 684 (9.94%) | 638 (9.27%) | 519 (7.54%) | 415 (6.03%) | 607 (8.82%) | 426 (6.19%) | 175 (2.54%) | 402 (5.84%) | 155 (2.25%) | 1.38% [1.16–1.58] | 1.58% [1.10–2.27] |
| Race_x_SES | 11,160 | 741 (6.64%) | 653 (5.85%) | 526 (4.71%) | 347 (3.11%) | 629 (5.64%) | 433 (3.88%) | 286 (2.56%) | 296 (2.65%) | 219 (1.96%) | 1.29% [1.11–1.38] | 1.53% [1.14–2.20] |
| Race_x_gender | 15,960 | 1,611 (10.09%) | 1,470 (9.21%) | 1,114 (6.98%) | 881 (5.52%) | 1,329 (8.33%) | 873 (5.47%) | 444 (2.78%) | 799 (5.01%) | 325 (2.04%) | 1.15% [1.03–1.30] | 1.49% [1.00–2.06] |
| Religion | 1,200 | 50 (4.17%) | 17 (1.42%) | 126 (10.50%) | 122 (10.17%) | 120 (10.00%) | 118 (9.83%) | 74 (6.17%) | 116 (9.67%) | 19 (1.58%) | 1.25% [0.92–1.58] | 1.37% [0.83–3.17] |
| SES | 6,864 | 310 (4.52%) | 202 (2.94%) | 300 (4.37%) | 92 (1.34%) | 405 (5.90%) | 138 (2.01%) | 135 (1.97%) | 73 (1.06%) | 92 (1.34%) | 0.80% [0.66–1.01] | 0.96% [0.60–1.35] |
| Sexual_orientation | 864 | 72 (8.33%) | 46 (5.32%) | 29 (3.36%) | 17 (1.97%) | 65 (7.52%) | 27 (3.12%) | 27 (3.12%) | 9 (1.04%) | 15 (1.74%) | 1.34% [0.81–2.43] | 1.32% [0.93–1.74] |

*Legend — Cells show `changed (flips_pct)` (`pred != base_pred`, pooled `ambig + disambig`); random-family columns show `mean% [min–max]` across seeds.*

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `abl_religion4` | `abl_shared1` | `abl_race4` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 6.58% / 3.37% | 3.48% / 2.28% | 6.41% / 2.88% | 2.93% / 1.20% | 7.28% / 3.26% | 2.77% / 1.74% | 3.42% / 1.85% | 1.63% / 0.60% | 3.32% / 1.25% |
| Disability_status | 5.27% / 1.03% | 1.80% / 0.51% | 2.70% / 1.29% | 1.67% / 0.51% | 5.27% / 2.06% | 1.67% / 0.39% | 1.80% / 0.26% | 1.16% / 0.39% | 1.54% / 0.64% |
| Gender_identity | 33.25% / 14.67% | 32.40% / 14.21% | 5.08% / 2.19% | 2.57% / 1.13% | 4.23% / 2.05% | 2.26% / 1.45% | 1.80% / 1.06% | 1.34% / 0.78% | 2.05% / 1.06% |
| Nationality | 5.52% / 1.95% | 3.44% / 0.97% | 15.45% / 4.87% | 13.38% / 4.42% | 18.05% / 5.78% | 11.69% / 3.64% | 3.12% / 0.97% | 11.04% / 3.70% | 5.00% / 2.14% |
| Physical_appearance | 9.52% / 5.08% | 4.19% / 3.05% | 5.96% / 1.90% | 2.54% / 1.14% | 6.47% / 3.17% | 3.17% / 0.89% | 3.68% / 0.89% | 1.27% / 0.76% | 2.16% / 0.76% |
| Race_ethnicity | 12.82% / 7.06% | 12.09% / 6.45% | 10.41% / 4.68% | 8.37% / 3.69% | 12.01% / 5.64% | 8.87% / 3.52% | 3.60% / 1.48% | 8.34% / 3.34% | 3.28% / 1.22% |
| Race_x_SES | 6.45% / 4.39% | 5.61% / 4.09% | 4.89% / 2.65% | 3.62% / 1.65% | 5.82% / 3.30% | 3.78% / 2.49% | 2.28% / 1.59% | 3.10% / 1.54% | 2.19% / 0.88% |
| Race_x_gender | 9.72% / 4.00% | 9.37% / 3.56% | 6.97% / 2.13% | 5.84% / 1.65% | 8.13% / 2.86% | 5.61% / 1.60% | 2.93% / 0.78% | 5.31% / 1.44% | 2.28% / 0.51% |
| Religion | 5.83% / 2.50% | 1.83% / 1.00% | 13.67% / 7.33% | 13.50% / 6.83% | 13.00% / 7.00% | 12.67% / 7.00% | 7.67% / 4.67% | 12.83% / 6.50% | 1.67% / 1.50% |
| SES | 5.74% / 3.29% | 3.09% / 2.80% | 6.47% / 2.27% | 1.98% / 0.70% | 9.00% / 2.80% | 2.77% / 1.25% | 2.86% / 1.08% | 1.25% / 0.87% | 2.01% / 0.67% |
| Sexual_orientation | 11.81% / 4.86% | 6.94% / 3.70% | 3.94% / 2.78% | 2.78% / 1.16% | 10.19% / 4.86% | 3.94% / 2.31% | 4.40% / 1.85% | 1.85% / 0.23% | 2.55% / 0.93% |

*Legend — Cells show `ambig% / disambig%` (S/A/U state changes from `direction.csv`).*

</details>

## 4. Neuron and Changed-Item Overlap Between Sets

Pairwise count of shared zeroed neurons, neuron Jaccard, and changed-item Jaccard (all items and within each set's target category).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `real_race20` vs. `real_religion20` | 6 / 20,20 | 0.176 | 0.383 | Race_ethnicity: 0.436 | Religion: 0.464 | L31.311, L31.1209, L31.2398, L31.5975, L31.6950, L31.13336 |
| `real_gender20` vs. `real_gender5` | 5 / 20,5 | 0.250 | 0.609 | Gender_identity: 0.908 | Gender_identity: 0.908 | L27.1904, L31.2566, L31.2951, L31.6657, L31.14107 |
| `real_race20` vs. `real_race5` | 5 / 20,5 | 0.250 | 0.439 | Race_ethnicity: 0.541 | Race_ethnicity: 0.541 | L31.1495, L31.2100, L31.6466, L31.6532, L31.6950 |
| `real_religion20` vs. `real_religion5` | 5 / 20,5 | 0.250 | 0.398 | Religion: 0.653 | Religion: 0.653 | L31.33, L31.311, L31.2398, L31.2564, L31.6950 |
| `real_religion5` vs. `abl_religion4` | 4 / 5,4 | 0.800 | 0.319 | Religion: 0.263 | Religion: 0.263 | L31.33, L31.311, L31.2398, L31.2564 |
| `real_race5` vs. `abl_race4` | 4 / 5,4 | 0.800 | 0.260 | Race_ethnicity: 0.215 | Race_ethnicity: 0.215 | L31.1495, L31.2100, L31.6466, L31.6532 |
| `real_religion20` vs. `abl_religion4` | 4 / 20,4 | 0.200 | 0.164 | Religion: 0.244 | Religion: 0.244 | L31.33, L31.311, L31.2398, L31.2564 |
| `real_race20` vs. `abl_race4` | 4 / 20,4 | 0.200 | 0.164 | Race_ethnicity: 0.162 | Race_ethnicity: 0.162 | L31.1495, L31.2100, L31.6466, L31.6532 |
| `real_race20` vs. `real_religion5` | 3 / 20,5 | 0.136 | 0.434 | Race_ethnicity: 0.519 | Religion: 0.544 | L31.311, L31.2398, L31.6950 |
| `real_gender20` vs. `real_religion20` | 3 / 20,20 | 0.081 | 0.127 | Gender_identity: 0.060 | Religion: 0.214 | L31.33, L31.5975, L31.14027 |
| `real_race20` vs. `abl_religion4` | 2 / 20,4 | 0.091 | 0.172 | Race_ethnicity: 0.143 | Religion: 0.105 | L31.311, L31.2398 |
| `real_gender20` vs. `real_race20` | 2 / 20,20 | 0.053 | 0.125 | Gender_identity: 0.090 | Race_ethnicity: 0.100 | L31.5975, L31.14107 |
| `real_race5` vs. `abl_shared1` | 1 / 5,1 | 0.200 | 0.612 | Race_ethnicity: 0.742 | Race+Religion: - | L31.6950 |
| `real_religion5` vs. `abl_shared1` | 1 / 5,1 | 0.200 | 0.546 | Religion: 0.592 | Race+Religion: - | L31.6950 |
| `real_race5` vs. `real_religion5` | 1 / 5,5 | 0.111 | 0.501 | Race_ethnicity: 0.659 | Religion: 0.548 | L31.6950 |
| `real_race20` vs. `abl_shared1` | 1 / 20,1 | 0.050 | 0.397 | Race_ethnicity: 0.522 | Race+Religion: - | L31.6950 |
| `real_religion20` vs. `abl_shared1` | 1 / 20,1 | 0.050 | 0.350 | Religion: 0.466 | Race+Religion: - | L31.6950 |
| `real_religion20` vs. `real_race5` | 1 / 20,5 | 0.042 | 0.349 | Religion: 0.449 | Race_ethnicity: 0.477 | L31.6950 |
| `real_race20` vs. `real_gender5` | 1 / 20,5 | 0.042 | 0.130 | Race_ethnicity: 0.130 | Gender_identity: 0.087 | L31.14107 |
| `real_gender20` vs. `abl_religion4` | 1 / 20,4 | 0.043 | 0.120 | Gender_identity: 0.028 | Religion: 0.512 | L31.33 |
| `real_gender20` vs. `real_religion5` | 1 / 20,5 | 0.042 | 0.096 | Gender_identity: 0.038 | Religion: 0.200 | L31.33 |
| `abl_race4` vs. `abl_religion4` | 0 / 4,4 | 0.000 | 0.203 | Race_ethnicity: 0.231 | Religion: 0.081 | - |
| `abl_race4` vs. `abl_shared1` | 0 / 4,1 | 0.000 | 0.164 | Race_ethnicity: 0.151 | Race+Religion: - | - |
| `real_religion5` vs. `abl_race4` | 0 / 5,4 | 0.000 | 0.161 | Religion: 0.087 | Race_ethnicity: 0.155 | - |
| `abl_religion4` vs. `abl_shared1` | 0 / 4,1 | 0.000 | 0.151 | Religion: 0.118 | Race+Religion: - | - |
| `real_race5` vs. `abl_religion4` | 0 / 5,4 | 0.000 | 0.147 | Race_ethnicity: 0.130 | Religion: 0.095 | - |
| `real_religion20` vs. `abl_race4` | 0 / 20,4 | 0.000 | 0.116 | Religion: 0.086 | Race_ethnicity: 0.114 | - |
| `real_religion20` vs. `real_gender5` | 0 / 20,5 | 0.000 | 0.114 | Religion: 0.062 | Gender_identity: 0.064 | - |
| `real_gender5` vs. `abl_religion4` | 0 / 5,4 | 0.000 | 0.111 | Gender_identity: 0.029 | Religion: 0.083 | - |
| `real_gender5` vs. `real_religion5` | 0 / 5,5 | 0.000 | 0.104 | Gender_identity: 0.039 | Religion: 0.071 | - |
| `real_gender5` vs. `abl_race4` | 0 / 5,4 | 0.000 | 0.086 | Gender_identity: 0.022 | Race_ethnicity: 0.083 | - |
| `real_gender20` vs. `abl_race4` | 0 / 20,4 | 0.000 | 0.086 | Gender_identity: 0.020 | Race_ethnicity: 0.090 | - |
| `real_gender5` vs. `real_race5` | 0 / 5,5 | 0.000 | 0.078 | Gender_identity: 0.030 | Race_ethnicity: 0.077 | - |
| `real_gender5` vs. `abl_shared1` | 0 / 5,1 | 0.000 | 0.077 | Gender_identity: 0.023 | Race+Religion: - | - |
| `real_gender20` vs. `real_race5` | 0 / 20,5 | 0.000 | 0.070 | Gender_identity: 0.028 | Race_ethnicity: 0.060 | - |
| `real_gender20` vs. `abl_shared1` | 0 / 20,1 | 0.000 | 0.063 | Gender_identity: 0.022 | Race+Religion: - | - |

*Legend — **Shared Neurons**: identical `(layer, index)` neurons (`shared / size_A, size_B`); **Neuron Jaccard**: `|A ∩ B| / |A ∪ B|` over zeroed neurons; **Items Jacc (All / Target A / Target B)**: Jaccard overlap of changed items across all items or restricted to each set's target category.*

## 5. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

Secondary benchmark metrics for target categories (`disambig` and `ambig` contexts, state-based S/A/U transitions); descriptive and not primary localization measures.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | Disamb Bias Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: |
| Gender_identity | `abl_race4` | ambig | 58 / 2,836 (2.05%) | 30 / 28 | 0.07% | 0 / 0 | - |
| Gender_identity | `abl_race4` | disambig | 30 / 2,836 (1.06%) | 17 / 13 | 0.14% | 18 / 12 | 0.28 [-0.42, 1.06] |
| Gender_identity | `abl_religion4` | ambig | 51 / 2,836 (1.80%) | 25 / 26 | -0.04% | 0 / 0 | - |
| Gender_identity | `abl_religion4` | disambig | 30 / 2,836 (1.06%) | 20 / 10 | 0.35% | 17 / 13 | 0.71 [-0.00, 1.41] |
| Gender_identity | `abl_shared1` | ambig | 38 / 2,836 (1.34%) | 15 / 23 | -0.28% | 0 / 0 | - |
| Gender_identity | `abl_shared1` | disambig | 22 / 2,836 (0.78%) | 13 / 9 | 0.14% | 15 / 7 | 0.28 [-0.35, 0.92] |
| Gender_identity | `real_gender20` | ambig | 943 / 2,836 (33.25%) | 415 / 528 | -3.98% | 0 / 0 | - |
| Gender_identity | `real_gender20` | disambig | 416 / 2,836 (14.67%) | 202 / 214 | -0.42% | 307 / 110 | -0.85 [-3.67, 1.90] |
| Gender_identity | `real_gender5` | ambig | 919 / 2,836 (32.40%) | 408 / 511 | -3.63% | 0 / 0 | - |
| Gender_identity | `real_gender5` | disambig | 403 / 2,836 (14.21%) | 199 / 204 | -0.18% | 298 / 105 | -0.35 [-3.24, 2.54] |
| Gender_identity | `real_race20` | ambig | 144 / 2,836 (5.08%) | 62 / 82 | -0.71% | 0 / 0 | - |
| Gender_identity | `real_race20` | disambig | 62 / 2,836 (2.19%) | 35 / 27 | 0.28% | 40 / 23 | 0.56 [-0.49, 1.62] |
| Gender_identity | `real_race5` | ambig | 73 / 2,836 (2.57%) | 34 / 39 | -0.18% | 0 / 0 | - |
| Gender_identity | `real_race5` | disambig | 32 / 2,836 (1.13%) | 21 / 11 | 0.35% | 19 / 13 | 0.71 [-0.07, 1.41] |
| Gender_identity | `real_religion20` | ambig | 120 / 2,836 (4.23%) | 55 / 65 | -0.35% | 0 / 0 | - |
| Gender_identity | `real_religion20` | disambig | 58 / 2,836 (2.05%) | 30 / 28 | 0.07% | 31 / 27 | 0.14 [-0.85, 1.13] |
| Gender_identity | `real_religion5` | ambig | 64 / 2,836 (2.26%) | 30 / 34 | -0.14% | 0 / 0 | - |
| Gender_identity | `real_religion5` | disambig | 41 / 2,836 (1.45%) | 23 / 18 | 0.18% | 24 / 17 | 0.35 [-0.56, 1.13] |
| Race_ethnicity | `abl_race4` | ambig | 113 / 3,440 (3.28%) | 56 / 57 | -0.03% | 0 / 0 | - |
| Race_ethnicity | `abl_race4` | disambig | 42 / 3,440 (1.22%) | 18 / 24 | -0.17% | 17 / 25 | -0.35 [-1.10, 0.35] |
| Race_ethnicity | `abl_religion4` | ambig | 124 / 3,440 (3.60%) | 56 / 68 | -0.35% | 0 / 0 | - |
| Race_ethnicity | `abl_religion4` | disambig | 51 / 3,440 (1.48%) | 30 / 21 | 0.26% | 26 / 25 | 0.52 [-0.29, 1.34] |
| Race_ethnicity | `abl_shared1` | ambig | 287 / 3,440 (8.34%) | 150 / 137 | 0.38% | 0 / 0 | - |
| Race_ethnicity | `abl_shared1` | disambig | 115 / 3,440 (3.34%) | 67 / 48 | 0.55% | 71 / 44 | 1.10 [-0.00, 2.38] |
| Race_ethnicity | `real_gender20` | ambig | 441 / 3,440 (12.82%) | 217 / 224 | -0.20% | 0 / 0 | - |
| Race_ethnicity | `real_gender20` | disambig | 243 / 3,440 (7.06%) | 109 / 134 | -0.73% | 142 / 101 | -1.45 [-3.31, 0.12] |
| Race_ethnicity | `real_gender5` | ambig | 416 / 3,440 (12.09%) | 206 / 210 | -0.12% | 0 / 0 | - |
| Race_ethnicity | `real_gender5` | disambig | 222 / 3,440 (6.45%) | 105 / 117 | -0.35% | 126 / 96 | -0.70 [-2.33, 0.87] |
| Race_ethnicity | `real_race20` | ambig | 358 / 3,440 (10.41%) | 185 / 173 | 0.35% | 0 / 0 | - |
| Race_ethnicity | `real_race20` | disambig | 161 / 3,440 (4.68%) | 93 / 68 | 0.73% | 97 / 64 | 1.45 [0.00, 2.91] |
| Race_ethnicity | `real_race5` | ambig | 288 / 3,440 (8.37%) | 148 / 140 | 0.23% | 0 / 0 | - |
| Race_ethnicity | `real_race5` | disambig | 127 / 3,440 (3.69%) | 69 / 58 | 0.32% | 80 / 47 | 0.64 [-0.58, 1.92] |
| Race_ethnicity | `real_religion20` | ambig | 413 / 3,440 (12.01%) | 200 / 213 | -0.38% | 0 / 0 | - |
| Race_ethnicity | `real_religion20` | disambig | 194 / 3,440 (5.64%) | 110 / 84 | 0.76% | 121 / 73 | 1.51 [-0.06, 3.08] |
| Race_ethnicity | `real_religion5` | ambig | 305 / 3,440 (8.87%) | 157 / 148 | 0.26% | 0 / 0 | - |
| Race_ethnicity | `real_religion5` | disambig | 121 / 3,440 (3.52%) | 70 / 51 | 0.55% | 78 / 43 | 1.10 [-0.12, 2.39] |
| Religion | `abl_race4` | ambig | 10 / 600 (1.67%) | 5 / 4 | 0.17% | 0 / 1 | - |
| Religion | `abl_race4` | disambig | 9 / 600 (1.50%) | 2 / 5 | -0.50% | 1 / 6 | -0.67 [-2.68, 0.87] |
| Religion | `abl_religion4` | ambig | 46 / 600 (7.67%) | 12 / 30 | -3.00% | 0 / 4 | - |
| Religion | `abl_religion4` | disambig | 28 / 600 (4.67%) | 11 / 14 | -0.50% | 17 / 8 | -0.50 [-3.88, 2.84] |
| Religion | `abl_shared1` | ambig | 77 / 600 (12.83%) | 33 / 42 | -1.50% | 0 / 2 | - |
| Religion | `abl_shared1` | disambig | 39 / 600 (6.50%) | 15 / 24 | -1.50% | 26 / 13 | -3.04 [-7.41, 1.01] |
| Religion | `real_gender20` | ambig | 35 / 600 (5.83%) | 13 / 21 | -1.33% | 0 / 1 | - |
| Religion | `real_gender20` | disambig | 15 / 600 (2.50%) | 6 / 7 | -0.17% | 8 / 5 | 0.00 [-2.51, 2.53] |
| Religion | `real_gender5` | ambig | 11 / 600 (1.83%) | 7 / 4 | 0.50% | 0 / 0 | - |
| Religion | `real_gender5` | disambig | 6 / 600 (1.00%) | 1 / 4 | -0.50% | 2 / 3 | -0.83 [-2.51, 0.53] |
| Religion | `real_race20` | ambig | 82 / 600 (13.67%) | 32 / 44 | -2.00% | 0 / 6 | - |
| Religion | `real_race20` | disambig | 44 / 600 (7.33%) | 16 / 25 | -1.50% | 31 / 12 | -2.51 [-7.03, 1.71] |
| Religion | `real_race5` | ambig | 81 / 600 (13.50%) | 34 / 45 | -1.83% | 0 / 2 | - |
| Religion | `real_race5` | disambig | 41 / 600 (6.83%) | 17 / 23 | -1.00% | 29 / 12 | -1.85 [-6.27, 2.21] |
| Religion | `real_religion20` | ambig | 78 / 600 (13.00%) | 27 / 46 | -3.17% | 0 / 5 | - |
| Religion | `real_religion20` | disambig | 42 / 600 (7.00%) | 19 / 20 | -0.17% | 34 / 7 | 0.20 [-4.37, 4.65] |
| Religion | `real_religion5` | ambig | 76 / 600 (12.67%) | 29 / 40 | -1.83% | 0 / 7 | - |
| Religion | `real_religion5` | disambig | 42 / 600 (7.00%) | 16 / 24 | -1.33% | 30 / 11 | -2.35 [-6.67, 1.87] |

*Legend — **A→S / S→A**: flips from anti-stereotype to stereotyped person and vice versa; **Net→Stereo**: `(A→S − S→A) / n_items` (`%`); **Corr→Inc / Inc→Corr**: accuracy transitions; **Bias Δ (95% CI)**: change in benchmark bias score vs. baseline with paired item-bootstrap CI.*

## 6. Baseline Log-Likelihood Margin Mechanism

Median baseline margin (`top − second`) of changed versus unchanged items (`ALL` categories) and flip rate across baseline-margin deciles 1–10 (decile 1 = most undecided).

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `real_gender20` | real | 5,291 | 0.250 | 1.250 |
| `real_race20` | real | 3,399 | 0.141 | 1.219 |
| `real_religion20` | real | 4,027 | 0.172 | 1.234 |
| `real_gender5` | real | 4,598 | 0.250 | 1.250 |
| `real_race5` | real | 2,375 | 0.125 | 1.188 |
| `real_religion5` | real | 2,487 | 0.125 | 1.188 |
| `abl_race4` | ablation | 1,148 | 0.062 | 1.148 |
| `abl_religion4` | ablation | 1,434 | 0.062 | 1.156 |
| `abl_shared1` | ablation | 2,051 | 0.125 | 1.180 |
| `rand_gender20 (mean n=5)` | random_family_mean | 694.2 | 0.037 | 1.133 |
| `rand_religion20 (mean n=10)` | random_family_mean | 840.6 | 0.051 | 1.138 |

| Decile | Margin Range | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `abl_religion4` | `abl_shared1` | `abl_race4` | `rand_gender20 (mean n=5)` | `rand_religion20 (mean n=10)` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.188] | 37.28% | 32.19% | 33.52% | 24.75% | 36.39% | 26.65% | 21.74% | 20.99% | 18.68% | 11.58% | 13.32% |
| 2 | [0.188, 0.375] | 21.29% | 16.82% | 14.70% | 8.80% | 17.47% | 9.52% | 2.39% | 7.88% | 0.89% | 0.15% | 0.50% |
| 3 | [0.375, 0.609] | 12.57% | 11.13% | 5.81% | 3.98% | 8.39% | 3.73% | 0.31% | 3.50% | 0.05% | 0.04% | 0.24% |
| 4 | [0.609, 0.844] | 9.56% | 8.72% | 2.44% | 1.76% | 3.66% | 1.61% | 0.07% | 1.66% | 0.00% | 0.03% | 0.14% |
| 5 | [0.844, 1.125] | 5.80% | 5.71% | 1.16% | 0.92% | 1.86% | 0.80% | 0.00% | 0.80% | 0.00% | 0.02% | 0.08% |
| 6 | [1.125, 1.484] | 2.89% | 3.09% | 0.43% | 0.34% | 0.91% | 0.21% | 0.00% | 0.21% | 0.00% | 0.02% | 0.06% |
| 7 | [1.484, 1.953] | 1.04% | 0.91% | 0.03% | 0.03% | 0.15% | 0.00% | 0.00% | 0.02% | 0.00% | 0.00% | 0.02% |
| 8 | [1.953, 2.641] | 0.03% | 0.03% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 9 | [2.641, 3.805] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.01% | 0.00% |
| 10 | [3.805, 13.938] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) of changed vs. unchanged items; **Decile**: baseline-margin decile (`1` = smallest margin / most undecided, `10` = largest margin).*

## 7. Granular Breakdowns (Demographic Groups and Topics)

Flip rates disaggregated by mentioned demographic group (`answer_info`) and, for EsBBQ, by item topic (`relevant_social_value`).

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender_identity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `abl_religion4` | `abl_shared1` | `abl_race4` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | 4,616 | 26.78% | 26.13% | 3.70% | 1.73% | 3.12% | 2.01% | 1.47% | 1.15% | 1.36% |
| M | 4,616 | 26.78% | 26.13% | 3.70% | 1.73% | 3.12% | 2.01% | 1.47% | 1.15% | 1.36% |
| boy | 40 | 7.50% | 5.00% | 10.00% | 2.50% | 2.50% | 7.50% | 5.00% | 0.00% | 5.00% |
| girl | 40 | 5.00% | 5.00% | 5.00% | 7.50% | 5.00% | 5.00% | 7.50% | 2.50% | 5.00% |
| man | 152 | 5.92% | 2.63% | 11.84% | 2.63% | 8.55% | 1.32% | 3.29% | 0.66% | 3.29% |
| nonTrans | 464 | 7.11% | 6.03% | 1.51% | 3.45% | 2.59% | 0.43% | 0.43% | 0.65% | 3.45% |
| nonTrans_F | 200 | 17.50% | 17.50% | 1.50% | 1.00% | 0.50% | 1.00% | 1.00% | 1.00% | 1.00% |
| nonTrans_M | 200 | 23.50% | 24.50% | 3.50% | 1.00% | 3.50% | 1.50% | 1.00% | 0.50% | 1.00% |
| trans | 464 | 7.11% | 6.03% | 1.51% | 3.45% | 2.59% | 0.43% | 0.43% | 0.65% | 3.45% |
| trans_F | 200 | 23.50% | 24.50% | 3.50% | 1.00% | 3.50% | 1.50% | 1.00% | 0.50% | 1.00% |
| trans_M | 200 | 17.50% | 17.50% | 1.50% | 1.00% | 0.50% | 1.00% | 1.00% | 1.00% | 1.00% |
| woman | 152 | 3.95% | 1.32% | 10.53% | 1.32% | 7.89% | 1.97% | 2.63% | 0.00% | 1.97% |

### Race_ethnicity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `abl_religion4` | `abl_shared1` | `abl_race4` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| African | 200 | 5.00% | 3.00% | 4.50% | 2.50% | 4.00% | 3.50% | 3.00% | 2.50% | 2.50% |
| African American | 520 | 6.15% | 3.46% | 4.23% | 1.73% | 5.58% | 3.08% | 1.92% | 2.12% | 2.31% |
| Arab | 208 | 4.33% | 1.92% | 4.81% | 4.33% | 4.33% | 2.88% | 1.92% | 3.37% | 2.40% |
| Asian | 312 | 7.05% | 2.88% | 5.13% | 4.81% | 5.77% | 2.88% | 2.24% | 5.77% | 4.49% |
| Black | 520 | 4.23% | 2.31% | 3.27% | 1.73% | 3.27% | 1.92% | 0.77% | 1.92% | 1.54% |
| Caucasian | 128 | 3.91% | 2.34% | 2.34% | 0.78% | 3.91% | 3.91% | 3.12% | 2.34% | 0.78% |
| European | 184 | 8.15% | 4.35% | 10.33% | 3.80% | 11.41% | 3.80% | 2.17% | 3.26% | 3.80% |
| F-Arab | 1,056 | 12.03% | 10.51% | 7.39% | 5.30% | 8.90% | 5.78% | 2.84% | 5.68% | 2.08% |
| F-Asian | 904 | 19.25% | 19.69% | 8.30% | 4.31% | 7.96% | 5.31% | 2.65% | 4.31% | 2.21% |
| F-Black | 1,904 | 13.03% | 13.29% | 8.09% | 5.99% | 9.14% | 6.36% | 2.73% | 5.67% | 2.26% |
| F-Latino | 1,112 | 11.69% | 10.43% | 6.38% | 4.23% | 7.55% | 4.59% | 2.07% | 3.96% | 1.89% |
| F-White | 784 | 21.05% | 23.21% | 7.14% | 3.83% | 8.42% | 4.72% | 3.19% | 3.70% | 2.55% |
| Hispanic | 328 | 3.96% | 3.96% | 10.06% | 7.01% | 6.10% | 6.10% | 3.35% | 5.49% | 3.35% |
| Jewish | 168 | 9.52% | 5.95% | 7.14% | 6.55% | 5.95% | 5.36% | 2.98% | 5.95% | 2.38% |
| Latino | 304 | 2.96% | 2.63% | 4.28% | 4.61% | 5.26% | 4.61% | 2.63% | 2.96% | 2.63% |
| M-Arab | 824 | 9.10% | 6.92% | 16.75% | 17.72% | 21.48% | 16.87% | 2.43% | 17.35% | 1.94% |
| M-Asian | 848 | 7.08% | 7.78% | 8.49% | 8.84% | 9.91% | 8.02% | 2.48% | 8.14% | 1.53% |
| M-Black | 1,032 | 6.10% | 6.78% | 8.53% | 7.46% | 10.76% | 7.46% | 2.62% | 7.27% | 2.13% |
| M-Latino | 936 | 9.62% | 7.69% | 8.76% | 8.44% | 11.54% | 8.97% | 3.42% | 8.12% | 2.56% |
| M-White | 600 | 8.00% | 6.50% | 6.00% | 5.50% | 7.00% | 5.00% | 1.67% | 5.50% | 1.83% |
| Middle Eastern | 224 | 4.02% | 3.12% | 4.02% | 4.91% | 5.36% | 4.02% | 2.23% | 4.46% | 2.23% |
| Native American | 240 | 5.42% | 3.33% | 2.50% | 2.50% | 3.75% | 2.92% | 3.33% | 2.92% | 2.50% |
| Roma | 168 | 2.98% | 5.36% | 3.57% | 4.76% | 11.90% | 5.36% | 1.19% | 4.17% | 2.38% |
| South American | 136 | 2.94% | 8.09% | 5.15% | 2.94% | 3.68% | 4.41% | 3.68% | 3.68% | 4.41% |
| White | 120 | 3.33% | 5.00% | 5.00% | 1.67% | 2.50% | 1.67% | 2.50% | 1.67% | 1.67% |

### Religion — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `abl_religion4` | `abl_shared1` | `abl_race4` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atheist | 288 | 4.51% | 0.35% | 8.33% | 7.29% | 9.38% | 7.99% | 6.60% | 6.94% | 0.69% |
| Buddhist | 112 | 4.46% | 2.68% | 11.61% | 11.61% | 9.82% | 10.71% | 8.93% | 9.82% | 0.00% |
| Catholic | 264 | 3.79% | 0.76% | 9.47% | 9.47% | 11.36% | 11.36% | 5.30% | 9.85% | 1.89% |
| Christian | 312 | 3.85% | 1.92% | 7.37% | 6.09% | 10.58% | 7.37% | 4.17% | 6.09% | 2.56% |
| Hindu | 264 | 4.55% | 0.38% | 18.18% | 17.80% | 10.98% | 14.02% | 5.30% | 17.05% | 0.76% |
| Jewish | 280 | 3.93% | 1.79% | 6.07% | 6.07% | 12.14% | 11.07% | 6.79% | 6.07% | 1.43% |
| Mormon | 264 | 5.68% | 1.89% | 11.36% | 11.36% | 6.82% | 7.58% | 7.95% | 10.23% | 2.65% |
| Muslim | 328 | 4.57% | 1.83% | 8.84% | 9.76% | 7.32% | 6.71% | 6.40% | 8.54% | 1.52% |
| Protestant | 144 | 2.08% | 2.08% | 5.56% | 4.17% | 6.25% | 6.94% | 4.86% | 5.56% | 0.69% |
| Sikh | 144 | 2.78% | 1.39% | 24.31% | 23.61% | 17.36% | 19.44% | 6.94% | 21.53% | 2.78% |

</details>
