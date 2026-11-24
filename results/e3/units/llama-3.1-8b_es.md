# E3 Unit Dossier — `llama-3.1-8b_es` (Llama-3.1-8B, ES, EsBBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **EsBBQ** (27,320 items, 10 categories) · Source: `results/esbbq_zeroed_llama8b/analysis/20261112_081632_items`.

## 1. Set Summary and Target-Category Localization

Changed answers (`pred != base_pred`, `changed / n_items`, `%`) for each real and ablation set in its target category versus the mean of the other categories, with layer distribution and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | Rank | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | :--- | :--- |
| `exp11` | real | Gender | 20 | 635 / 4,832 (13.14%) | [12.17, 14.09] | 5.39% | 2.44× | 1/10 | 1,876 / 27,320 (6.87%) | A100-SXM4-80GB | L29:2, L30:2, L31:16 |
| `exp13` | real | Religion | 20 | 44 / 648 (6.79%) | [4.78, 8.80] | 5.01% | 1.36× | 2/10 | 1,368 / 27,320 (5.01%) | A100-SXM4-80GB | L21:1, L24:1, L28:1, L31:17 |
| `exp14` | real | RaceEthnicity | 20 | 324 / 3,716 (8.72%) | [7.75, 9.63] | 8.58% | 1.02× | 6/10 | 2,419 / 27,320 (8.85%) | A100-SXM4-80GB | L31:20 |
| `exp10` | real | Gender | 5 | 138 / 4,832 (2.86%) | [2.44, 3.35] | 3.27% | 0.87× | 6/10 | 914 / 27,320 (3.35%) | A100-SXM4-40GB | L29:1, L31:4 |
| `exp12` | real | RaceEthnicity | 5 | 110 / 3,716 (2.96%) | [2.42, 3.50] | 3.35% | 0.88× | 6/10 | 895 / 27,320 (3.28%) | A100-SXM4-40GB | L31:5 |
| `exp2` | real | Religion | 5 | 23 / 648 (3.55%) | [2.16, 5.09] | 2.60% | 1.36× | 2/10 | 732 / 27,320 (2.68%) | A100-SXM4-40GB | L28:1, L31:4 |

## 2. Changed Answers Across All Categories (`Set × Category`)

Full matrix of changed answers (`pred != base_pred`, `changed (flips_pct)`) across every benchmark category (all items pooled; random families summarized by mean).

| Category | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 4,068 | 113 (2.78%) | 150 (3.69%) | 133 (3.27%) | 179 (4.40%) | 249 (6.12%) | 456 (11.21%) |
| DisabilityStatus | 2,832 | 53 (1.87%) | 66 (2.33%) | 73 (2.58%) | 129 (4.56%) | 117 (4.13%) | 252 (8.90%) |
| Gender | 4,832 | 100 (2.07%) | 138 (2.86%) | 230 (4.76%) | 220 (4.55%) | 635 (13.14%) | 267 (5.53%) |
| LGBTQIA | 2,000 | 49 (2.45%) | 102 (5.10%) | 75 (3.75%) | 117 (5.85%) | 124 (6.20%) | 224 (11.20%) |
| Nationality | 504 | 13 (2.58%) | 9 (1.79%) | 14 (2.78%) | 26 (5.16%) | 28 (5.56%) | 29 (5.75%) |
| PhysicalAppearance | 3,528 | 93 (2.64%) | 108 (3.06%) | 93 (2.64%) | 187 (5.30%) | 142 (4.02%) | 359 (10.18%) |
| RaceEthnicity | 3,716 | 174 (4.68%) | 175 (4.71%) | 110 (2.96%) | 267 (7.19%) | 327 (8.80%) | 324 (8.72%) |
| Religion | 648 | 23 (3.55%) | 18 (2.78%) | 31 (4.78%) | 44 (6.79%) | 24 (3.70%) | 47 (7.25%) |
| SES | 4,204 | 93 (2.21%) | 116 (2.76%) | 105 (2.50%) | 156 (3.71%) | 172 (4.09%) | 381 (9.06%) |
| SpanishRegion | 988 | 21 (2.13%) | 32 (3.24%) | 31 (3.14%) | 43 (4.35%) | 58 (5.87%) | 80 (8.10%) |

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 3.48% / 2.45% | 4.33% / 3.39% | 4.02% / 2.92% | 5.65% / 3.82% | 8.05% / 5.22% | 10.76% / 11.42% |
| DisabilityStatus | 2.48% / 1.58% | 2.69% / 2.15% | 3.34% / 2.21% | 5.71% / 3.99% | 4.96% / 3.73% | 12.72% / 7.04% |
| Gender | 2.59% / 1.83% | 3.86% / 2.40% | 6.32% / 4.06% | 4.99% / 4.36% | 18.35% / 10.79% | 7.91% / 4.45% |
| LGBTQIA | 2.78% / 2.32% | 6.42% / 4.56% | 3.82% / 3.72% | 6.08% / 5.76% | 7.64% / 5.62% | 14.06% / 10.04% |
| Nationality | 2.98% / 2.38% | 0.60% / 2.38% | 2.98% / 2.68% | 6.55% / 4.46% | 8.93% / 3.87% | 7.14% / 5.06% |
| PhysicalAppearance | 4.08% / 1.91% | 5.44% / 1.87% | 3.74% / 2.08% | 9.35% / 3.27% | 7.91% / 2.08% | 14.54% / 7.99% |
| RaceEthnicity | 5.62% / 4.22% | 6.92% / 3.62% | 4.15% / 2.37% | 8.96% / 6.31% | 11.89% / 7.27% | 10.67% / 7.76% |
| Religion | 2.78% / 3.94% | 2.31% / 3.01% | 6.02% / 4.17% | 4.63% / 7.87% | 2.78% / 4.17% | 12.04% / 4.86% |
| SES | 2.61% / 2.02% | 3.19% / 2.55% | 2.97% / 2.27% | 4.06% / 3.54% | 5.07% / 3.61% | 10.14% / 8.53% |
| SpanishRegion | 4.32% / 1.05% | 7.10% / 1.36% | 6.79% / 1.36% | 8.33% / 2.41% | 13.27% / 2.26% | 18.52% / 3.01% |

</details>

## 3. Neuron and Changed-Item Overlap Between Sets

Pairwise count of shared zeroed neurons, neuron Jaccard, and changed-item Jaccard (all items and within each set's target category).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `exp13` vs. `exp14` | 6 / 20,20 | 0.176 | 0.267 | Religion: 0.138 | RaceEthnicity: 0.438 | L31.273, L31.2398, L31.5897, L31.6950, L31.13336, L31.14276 |
| `exp11` vs. `exp10` | 5 / 20,5 | 0.250 | 0.276 | Gender: 0.111 | Gender: 0.111 | L29.8403, L31.77, L31.2566, L31.7783, L31.13336 |
| `exp13` vs. `exp2` | 5 / 20,5 | 0.250 | 0.275 | Religion: 0.457 | Religion: 0.457 | L28.7507, L31.33, L31.5897, L31.6950, L31.12768 |
| `exp14` vs. `exp12` | 5 / 20,5 | 0.250 | 0.190 | RaceEthnicity: 0.176 | RaceEthnicity: 0.176 | L31.2398, L31.5897, L31.8118, L31.10439, L31.12004 |
| `exp11` vs. `exp14` | 4 / 20,20 | 0.111 | 0.202 | Gender: 0.129 | RaceEthnicity: 0.071 | L31.1695, L31.2398, L31.12252, L31.13336 |
| `exp11` vs. `exp13` | 4 / 20,20 | 0.111 | 0.161 | Gender: 0.026 | Religion: 0.388 | L31.2398, L31.2564, L31.6466, L31.13336 |
| `exp13` vs. `exp12` | 2 / 20,5 | 0.087 | 0.234 | Religion: 0.230 | RaceEthnicity: 0.153 | L31.2398, L31.5897 |
| `exp14` vs. `exp2` | 2 / 20,5 | 0.087 | 0.128 | RaceEthnicity: 0.300 | Religion: 0.129 | L31.5897, L31.6950 |
| `exp12` vs. `exp2` | 1 / 5,5 | 0.111 | 0.250 | RaceEthnicity: 0.174 | Religion: 0.200 | L31.5897 |
| `exp13` vs. `exp10` | 1 / 20,5 | 0.042 | 0.247 | Religion: 0.319 | Gender: 0.239 | L31.13336 |
| `exp14` vs. `exp10` | 1 / 20,5 | 0.042 | 0.184 | RaceEthnicity: 0.102 | Gender: 0.258 | L31.13336 |
| `exp11` vs. `exp12` | 1 / 20,5 | 0.042 | 0.175 | Gender: 0.190 | RaceEthnicity: 0.095 | L31.2398 |
| `exp10` vs. `exp12` | 0 / 5,5 | 0.000 | 0.257 | Gender: 0.314 | RaceEthnicity: 0.140 | - |
| `exp10` vs. `exp2` | 0 / 5,5 | 0.000 | 0.192 | Gender: 0.227 | Religion: 0.519 | - |
| `exp11` vs. `exp2` | 0 / 20,5 | 0.000 | 0.138 | Gender: 0.119 | Religion: 0.424 | - |

## 4. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

Secondary benchmark metrics for target categories (`disambig` and `ambig` contexts, state-based S/A/U transitions); descriptive and not primary localization measures.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | EsBBQ Bias Δ (95% CI) | BBQ-Formula Disamb Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: | ---: |
| Gender | `exp10` | ambig | 58 / 1,504 (3.86%) | 26 / 32 | -0.40% | 0 / 0 | -0.80 [-2.80, 1.06] | - |
| Gender | `exp10` | disambig | 80 / 3,328 (2.40%) | 45 / 34 | 0.33% | 41 / 39 | 0.60 [-0.47, 1.62] | 0.63 [-0.42, 1.68] |
| Gender | `exp11` | ambig | 276 / 1,504 (18.35%) | 137 / 138 | -0.07% | 0 / 1 | -0.07 [-4.32, 4.19] | - |
| Gender | `exp11` | disambig | 359 / 3,328 (10.79%) | 206 / 152 | 1.62% | 177 / 182 | 3.31 [0.96, 5.76] | 3.28 [0.93, 5.71] |
| Gender | `exp12` | ambig | 95 / 1,504 (6.32%) | 46 / 48 | -0.13% | 1 / 0 | -0.33 [-2.86, 2.13] | - |
| Gender | `exp12` | disambig | 135 / 3,328 (4.06%) | 70 / 64 | 0.18% | 68 / 67 | 0.30 [-1.14, 1.56] | 0.33 [-1.11, 1.59] |
| Gender | `exp13` | ambig | 75 / 1,504 (4.99%) | 32 / 42 | -0.66% | 1 / 0 | -1.40 [-3.66, 0.86] | - |
| Gender | `exp13` | disambig | 145 / 3,328 (4.36%) | 70 / 74 | -0.12% | 73 / 72 | -0.30 [-1.69, 1.03] | -0.27 [-1.68, 1.08] |
| Gender | `exp14` | ambig | 119 / 1,504 (7.91%) | 65 / 52 | 0.86% | 0 / 2 | 1.86 [-0.80, 4.59] | - |
| Gender | `exp14` | disambig | 148 / 3,328 (4.45%) | 80 / 67 | 0.39% | 70 / 78 | 0.84 [-0.54, 2.29] | 0.81 [-0.57, 2.23] |
| Gender | `exp2` | ambig | 39 / 1,504 (2.59%) | 18 / 20 | -0.13% | 1 / 0 | -0.33 [-1.99, 1.33] | - |
| Gender | `exp2` | disambig | 61 / 3,328 (1.83%) | 34 / 26 | 0.24% | 20 / 41 | 0.42 [-0.53, 1.36] | 0.45 [-0.45, 1.38] |
| RaceEthnicity | `exp10` | ambig | 85 / 1,228 (6.92%) | 41 / 36 | 0.41% | 5 / 3 | 0.98 [-1.79, 3.66] | - |
| RaceEthnicity | `exp10` | disambig | 90 / 2,488 (3.62%) | 43 / 43 | 0.00% | 42 / 48 | 0.32 [-1.12, 1.77] | 0.16 [-1.29, 1.66] |
| RaceEthnicity | `exp11` | ambig | 146 / 1,228 (11.89%) | 70 / 62 | 0.65% | 1 / 13 | 1.63 [-2.20, 5.13] | - |
| RaceEthnicity | `exp11` | disambig | 181 / 2,488 (7.27%) | 89 / 83 | 0.24% | 90 / 87 | 0.40 [-1.70, 2.49] | 0.30 [-1.86, 2.48] |
| RaceEthnicity | `exp12` | ambig | 51 / 1,228 (4.15%) | 25 / 17 | 0.65% | 7 / 2 | 1.71 [-0.24, 3.75] | - |
| RaceEthnicity | `exp12` | disambig | 59 / 2,488 (2.37%) | 27 / 28 | -0.04% | 29 / 30 | 0.24 [-0.97, 1.45] | 0.08 [-1.16, 1.27] |
| RaceEthnicity | `exp13` | ambig | 110 / 1,228 (8.96%) | 41 / 53 | -0.98% | 15 / 1 | -1.14 [-4.23, 1.79] | - |
| RaceEthnicity | `exp13` | disambig | 157 / 2,488 (6.31%) | 80 / 69 | 0.44% | 79 / 78 | 1.37 [-0.64, 3.30] | 1.15 [-0.96, 3.11] |
| RaceEthnicity | `exp14` | ambig | 131 / 1,228 (10.67%) | 59 / 56 | 0.24% | 5 / 11 | 0.16 [-3.18, 3.58] | - |
| RaceEthnicity | `exp14` | disambig | 193 / 2,488 (7.76%) | 97 / 84 | 0.52% | 106 / 84 | 0.48 [-1.76, 2.69] | 0.68 [-1.57, 2.83] |
| RaceEthnicity | `exp2` | ambig | 69 / 1,228 (5.62%) | 24 / 39 | -1.22% | 6 / 0 | -1.95 [-4.48, 0.33] | - |
| RaceEthnicity | `exp2` | disambig | 105 / 2,488 (4.22%) | 58 / 43 | 0.60% | 45 / 59 | 1.45 [-0.27, 3.16] | 1.33 [-0.46, 3.09] |
| Religion | `exp10` | ambig | 5 / 216 (2.31%) | 2 / 2 | 0.00% | 1 / 0 | -0.46 [-4.17, 3.24] | - |
| Religion | `exp10` | disambig | 13 / 432 (3.01%) | 7 / 6 | 0.23% | 8 / 5 | 0.46 [-2.51, 3.83] | 0.47 [-2.37, 3.77] |
| Religion | `exp11` | ambig | 6 / 216 (2.78%) | 3 / 2 | 0.46% | 1 / 0 | 0.46 [-3.24, 4.63] | - |
| Religion | `exp11` | disambig | 18 / 432 (4.17%) | 8 / 10 | -0.46% | 9 / 9 | -0.93 [-4.61, 3.08] | -0.93 [-4.66, 3.24] |
| Religion | `exp12` | ambig | 13 / 216 (6.02%) | 5 / 5 | 0.00% | 3 / 0 | 0.46 [-5.56, 6.48] | - |
| Religion | `exp12` | disambig | 18 / 432 (4.17%) | 12 / 6 | 1.39% | 10 / 8 | 2.78 [-0.96, 6.51] | 2.80 [-0.93, 6.56] |
| Religion | `exp13` | ambig | 10 / 216 (4.63%) | 4 / 2 | 0.93% | 4 / 0 | 1.85 [-2.78, 6.94] | - |
| Religion | `exp13` | disambig | 34 / 432 (7.87%) | 13 / 17 | -0.93% | 17 / 17 | -0.93 [-5.96, 4.15] | -1.34 [-6.07, 3.54] |
| Religion | `exp14` | ambig | 26 / 216 (12.04%) | 7 / 3 | 1.85% | 0 / 16 | 5.56 [-0.94, 12.04] | - |
| Religion | `exp14` | disambig | 21 / 432 (4.86%) | 16 / 5 | 2.55% | 12 / 9 | 5.09 [0.89, 9.26] | 5.14 [0.93, 9.37] |
| Religion | `exp2` | ambig | 6 / 216 (2.78%) | 3 / 2 | 0.46% | 1 / 0 | 0.46 [-3.70, 4.64] | - |
| Religion | `exp2` | disambig | 17 / 432 (3.94%) | 7 / 9 | -0.46% | 9 / 8 | -1.39 [-4.94, 2.50] | -1.15 [-4.45, 2.60] |

## 5. Baseline Log-Likelihood Margin Mechanism

Median baseline margin (`top − second`) of changed versus unchanged items (`ALL` categories) and flip rate across baseline-margin deciles 1–10 (decile 1 = most undecided).

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `exp2` | real | 732 | 0.062 | 1.000 |
| `exp10` | real | 914 | 0.094 | 1.000 |
| `exp12` | real | 895 | 0.078 | 1.000 |
| `exp13` | real | 1,368 | 0.125 | 1.004 |
| `exp11` | real | 1,876 | 0.125 | 1.031 |
| `exp14` | real | 2,419 | 0.156 | 1.062 |

| Decile | Margin Range | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.141] | 22.58% | 25.66% | 27.53% | 33.42% | 38.65% | 40.41% |
| 2 | [0.141, 0.344] | 2.71% | 5.01% | 5.09% | 11.64% | 17.24% | 25.44% |
| 3 | [0.344, 0.500] | 0.92% | 1.54% | 0.15% | 3.44% | 7.94% | 12.12% |
| 4 | [0.500, 0.750] | 0.44% | 0.33% | 0.00% | 0.99% | 3.26% | 6.77% |
| 5 | [0.750, 0.969] | 0.11% | 0.29% | 0.00% | 0.51% | 0.92% | 1.79% |
| 6 | [0.969, 1.234] | 0.04% | 0.44% | 0.00% | 0.07% | 0.55% | 0.95% |
| 7 | [1.234, 1.516] | 0.00% | 0.11% | 0.00% | 0.00% | 0.07% | 0.70% |
| 8 | [1.516, 1.973] | 0.00% | 0.07% | 0.00% | 0.00% | 0.04% | 0.37% |
| 9 | [1.977, 2.750] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 10 | [2.750, 16.188] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## 6. Granular Breakdowns (Demographic Groups and Topics)

Flip rates disaggregated by mentioned demographic group (`answer_info`) and, for EsBBQ, by item topic (`relevant_social_value`).

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f | 4,832 | 2.07% | 2.86% | 4.76% | 4.55% | 13.14% | 5.53% |
| m | 4,832 | 2.07% | 2.86% | 4.76% | 4.55% | 13.14% | 5.53% |

### RaceEthnicity — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| africano | 124 | 4.03% | 3.23% | 3.23% | 8.06% | 2.42% | 6.45% |
| asiático | 264 | 6.82% | 4.55% | 4.17% | 10.61% | 11.74% | 8.71% |
| blanco | 3,716 | 4.68% | 4.71% | 2.96% | 7.19% | 8.80% | 8.72% |
| de Europa del Este | 396 | 1.01% | 2.27% | 3.54% | 4.04% | 12.37% | 6.06% |
| gitano | 372 | 2.69% | 3.49% | 1.88% | 5.65% | 3.76% | 12.10% |
| indio | 264 | 13.64% | 2.65% | 3.79% | 17.42% | 6.82% | 14.39% |
| latino | 672 | 1.79% | 10.71% | 3.42% | 4.17% | 10.71% | 7.29% |
| negro | 748 | 3.88% | 4.28% | 2.81% | 4.01% | 9.76% | 7.09% |
| árabe | 876 | 6.85% | 2.97% | 2.28% | 10.05% | 7.65% | 9.59% |

### Religion — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ateo | 216 | 1.39% | 0.46% | 2.31% | 3.70% | 2.31% | 7.41% |
| católico | 408 | 3.92% | 2.94% | 5.39% | 6.62% | 3.19% | 8.33% |
| judío | 48 | 4.17% | 2.08% | 4.17% | 10.42% | 2.08% | 4.17% |
| musulmán | 144 | 7.64% | 6.25% | 6.94% | 11.11% | 6.25% | 6.94% |
| no creyente | 216 | 2.31% | 2.31% | 6.02% | 4.63% | 3.70% | 8.80% |
| testigo de Jehová | 264 | 3.41% | 3.03% | 3.79% | 8.33% | 4.55% | 4.92% |

</details>

<details>
<summary>Expand EsBBQ topic breakdown (`relevant_social_value`)</summary>

### Gender — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cocinar | 240 | 2.08% | 2.92% | 7.08% | 5.42% | 19.58% | 5.83% |
| conducción temeraria o imprudente | 60 | 3.33% | 1.67% | 3.33% | 6.67% | 1.67% | 8.33% |
| debilidad, fragilidad, menos fuerza | 60 | 8.33% | 8.33% | 8.33% | 6.67% | 23.33% | 5.00% |
| emocional | 120 | 1.67% | 2.50% | 4.17% | 5.00% | 12.50% | 5.83% |
| gustos asociados a un género: coches | 120 | 1.67% | 1.67% | 0.83% | 3.33% | 7.50% | 2.50% |
| gustos asociados a un género: color azul | 160 | 1.88% | 1.88% | 2.50% | 5.62% | 15.62% | 5.62% |
| gustos asociados a un género: color rosa | 160 | 3.12% | 2.50% | 4.38% | 7.50% | 14.37% | 8.12% |
| gustos asociados a un género: fútbol | 72 | 0.00% | 1.39% | 2.78% | 2.78% | 5.56% | 4.17% |
| gustos asociados a un género: muñecas | 120 | 0.00% | 2.50% | 1.67% | 2.50% | 4.17% | 1.67% |
| inestabilidad emocional | 120 | 2.50% | 4.17% | 5.83% | 6.67% | 15.00% | 8.33% |
| inferioridad | 120 | 1.67% | 3.33% | 9.17% | 7.50% | 12.50% | 12.50% |
| ira, agresividad, violencia | 160 | 3.12% | 3.12% | 4.38% | 1.88% | 14.37% | 5.00% |
| mala conducción | 180 | 2.78% | 1.67% | 3.33% | 5.00% | 10.00% | 5.00% |
| malas entre ellas | 36 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.33% |
| mayor carga familiar | 240 | 2.08% | 2.50% | 6.67% | 5.83% | 13.33% | 5.83% |
| menor inteligencia | 60 | 1.67% | 1.67% | 6.67% | 5.00% | 11.67% | 6.67% |
| menor proactividad, liderazgo | 60 | 0.00% | 6.67% | 6.67% | 3.33% | 8.33% | 5.00% |
| no llorar | 180 | 1.67% | 2.78% | 6.11% | 3.89% | 15.56% | 5.56% |
| no ser alto cargo | 120 | 0.83% | 2.50% | 3.33% | 4.17% | 8.33% | 4.17% |
| peor con la tecnología | 120 | 0.83% | 2.50% | 1.67% | 4.17% | 7.50% | 5.00% |
| peor en STEM | 240 | 2.50% | 4.58% | 5.42% | 5.42% | 17.08% | 6.67% |
| peor en deportes | 72 | 2.78% | 0.00% | 5.56% | 0.00% | 13.89% | 4.17% |
| peor en matemáticas | 280 | 2.14% | 2.14% | 6.07% | 3.93% | 13.93% | 5.36% |
| persona cerrada emocionalmente, falta de empatía | 240 | 2.08% | 3.33% | 4.17% | 2.92% | 13.75% | 2.08% |
| persona exagerada | 180 | 0.56% | 1.11% | 3.89% | 5.56% | 15.56% | 5.56% |
| priorizar la familia sobre la carrera profesional, mayor carga familiar | 200 | 1.00% | 1.00% | 1.50% | 2.00% | 10.00% | 1.50% |
| priorizar la maternidad sobre la carrera profesional | 100 | 3.00% | 4.00% | 7.00% | 5.00% | 15.00% | 5.00% |
| profesión feminizada: maestra | 120 | 0.83% | 3.33% | 3.33% | 5.00% | 5.00% | 5.83% |
| profesión feminizada: secretaria | 232 | 3.45% | 5.60% | 7.33% | 5.60% | 20.26% | 6.90% |
| profesión masculinizada: informática | 120 | 2.50% | 0.83% | 3.33% | 4.17% | 10.83% | 5.00% |
| profesión masculinizada: ingeniería | 120 | 1.67% | 1.67% | 2.50% | 1.67% | 9.17% | 2.50% |
| querer tener hijos | 120 | 1.67% | 4.17% | 5.83% | 3.33% | 18.33% | 5.83% |
| sensible | 60 | 5.00% | 5.00% | 8.33% | 3.33% | 26.67% | 6.67% |
| violencia de género (maltratador) | 180 | 2.22% | 3.89% | 5.56% | 7.78% | 11.11% | 10.56% |
| víctima de violencia de género | 60 | 3.33% | 3.33% | 3.33% | 3.33% | 10.00% | 3.33% |

### RaceEthnicity — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acoso / agresión sexual | 12 | 0.00% | 25.00% | 0.00% | 8.33% | 25.00% | 50.00% |
| alcoholismo | 72 | 0.00% | 12.50% | 1.39% | 2.78% | 15.28% | 8.33% |
| analfabetismo tecnológico | 24 | 8.33% | 4.17% | 4.17% | 16.67% | 4.17% | 8.33% |
| ayudas sociales | 504 | 6.15% | 4.17% | 2.98% | 9.33% | 11.51% | 8.93% |
| bajo rendimiento académico | 432 | 5.79% | 2.08% | 2.78% | 6.25% | 7.87% | 7.64% |
| buscar dinero en relaciones | 72 | 5.56% | 18.06% | 6.94% | 9.72% | 15.28% | 9.72% |
| casarse y tener relaciones sexuales a edades tempranas | 24 | 4.17% | 4.17% | 4.17% | 8.33% | 4.17% | 8.33% |
| consumo de drogas | 96 | 4.17% | 6.25% | 6.25% | 2.08% | 5.21% | 11.46% |
| delincuencia, robar | 216 | 3.24% | 6.94% | 3.24% | 6.02% | 8.80% | 12.04% |
| docilidad | 72 | 6.94% | 1.39% | 0.00% | 9.72% | 11.11% | 9.72% |
| español como lengua extranjera | 168 | 2.98% | 6.55% | 7.14% | 7.74% | 8.33% | 13.10% |
| falta de acceso a la tecnología | 24 | 0.00% | 0.00% | 0.00% | 4.17% | 0.00% | 0.00% |
| falta de derechos de la mujer, machismo | 144 | 7.64% | 0.69% | 0.69% | 7.64% | 6.94% | 7.64% |
| hipersexualización, prostitución | 72 | 1.39% | 9.72% | 0.00% | 2.78% | 18.06% | 2.78% |
| inmigración ilegal | 132 | 1.52% | 5.30% | 1.52% | 3.79% | 12.88% | 6.06% |
| ira, agresividad, violencia | 120 | 0.83% | 1.67% | 3.33% | 5.83% | 8.33% | 6.67% |
| mal comportamiento, persona problemática | 120 | 1.67% | 9.17% | 2.50% | 9.17% | 8.33% | 10.83% |
| mal olor | 72 | 6.94% | 5.56% | 2.78% | 6.94% | 6.94% | 8.33% |
| mala gente, no ser de fiar | 24 | 4.17% | 4.17% | 8.33% | 8.33% | 8.33% | 16.67% |
| mayor presión parental | 48 | 0.00% | 2.08% | 0.00% | 4.17% | 2.08% | 0.00% |
| meritocracia | 744 | 6.18% | 3.63% | 2.96% | 8.06% | 10.22% | 7.93% |
| pobreza | 152 | 2.63% | 3.29% | 3.29% | 5.26% | 2.63% | 7.89% |
| promiscuidad | 36 | 0.00% | 2.78% | 0.00% | 2.78% | 2.78% | 0.00% |
| sumisión | 24 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| terrorismo | 168 | 7.74% | 4.76% | 1.79% | 9.52% | 5.95% | 10.12% |
| vagancia, no querer trabajar | 96 | 2.08% | 5.21% | 4.17% | 3.12% | 2.08% | 9.38% |
| violencia de género | 48 | 4.17% | 10.42% | 4.17% | 16.67% | 2.08% | 16.67% |

### Religion — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abuso sexual infantil | 48 | 4.17% | 4.17% | 8.33% | 6.25% | 4.17% | 10.42% |
| anticiencia | 336 | 1.79% | 1.79% | 5.06% | 4.46% | 2.98% | 8.33% |
| avaricia | 48 | 4.17% | 2.08% | 4.17% | 10.42% | 2.08% | 4.17% |
| celebración de tradiciones religiosas | 48 | 8.33% | 8.33% | 2.08% | 14.58% | 4.17% | 0.00% |
| falta de derechos de la mujer, machismo | 60 | 1.67% | 0.00% | 0.00% | 3.33% | 3.33% | 3.33% |
| persona sentenciosa, moralista | 48 | 2.08% | 0.00% | 2.08% | 4.17% | 4.17% | 10.42% |
| poligamia | 24 | 16.67% | 12.50% | 16.67% | 20.83% | 16.67% | 8.33% |
| terrorismo | 36 | 8.33% | 5.56% | 5.56% | 13.89% | 2.78% | 8.33% |

</details>
