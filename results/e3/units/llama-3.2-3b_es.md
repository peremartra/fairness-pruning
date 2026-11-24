# E3 Unit Dossier — `llama-3.2-3b_es` (Llama-3.2-3B, ES, EsBBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **EsBBQ** (27,320 items, 10 categories) · Source: `results/esbbq_zeroed_llama3b/analysis/20261112_081707_items`.

## 1. Set Summary and Target-Category Localization

Changed answers (`pred != base_pred`, `changed / n_items`, `%`) for each real and ablation set in its target category versus the mean of the other categories, with layer distribution and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | Rank | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | :--- | :--- |
| `exp11` | real | Gender | 20 | 816 / 4,832 (16.89%) | [15.85, 17.98] | 6.73% | 2.51× | 1/10 | 2,337 / 27,320 (8.55%) | L4 | L16:2, L18:1, L22:1, L24:1, L25:2, L26:3, L27:10 |
| `exp13` | real | Religion | 20 | 101 / 648 (15.59%) | [12.50, 18.36] | 10.96% | 1.42× | 3/10 | 3,055 / 27,320 (11.18%) | L4 | L19:1, L20:1, L21:1, L23:1, L24:1, L25:1, L26:1, L27:13 |
| `exp14` | real | RaceEthnicity | 20 | 529 / 3,716 (14.24%) | [13.13, 15.31] | 12.19% | 1.17× | 5/10 | 3,307 / 27,320 (12.10%) | L4 | L17:1, L23:1, L24:2, L25:2, L26:2, L27:12 |
| `exp10` | real | Gender | 5 | 74 / 4,832 (1.53%) | [1.20, 1.88] | 2.79% | 0.55× | 10/10 | 684 / 27,320 (2.50%) | L4 | L25:2, L26:1, L27:2 |
| `exp12` | real | RaceEthnicity | 5 | 298 / 3,716 (8.02%) | [7.13, 8.83] | 8.76% | 0.92× | 6/10 | 2,376 / 27,320 (8.70%) | L4 | L27:5 |
| `exp2` | real | Religion | 5 | 59 / 648 (9.10%) | [6.79, 11.27] | 3.85% | 2.36× | 1/10 | 1,070 / 27,320 (3.92%) | L4 | L19:1, L21:1, L27:3 |

## 2. Changed Answers Across All Categories (`Set × Category`)

Full matrix of changed answers (`pred != base_pred`, `changed (flips_pct)`) across every benchmark category (all items pooled; random families summarized by mean).

| Category | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 4,068 | 117 (2.88%) | 85 (2.09%) | 487 (11.97%) | 496 (12.19%) | 271 (6.66%) | 634 (15.59%) |
| DisabilityStatus | 2,832 | 148 (5.23%) | 82 (2.90%) | 389 (13.74%) | 463 (16.35%) | 219 (7.73%) | 502 (17.73%) |
| Gender | 4,832 | 126 (2.61%) | 74 (1.53%) | 170 (3.52%) | 421 (8.71%) | 816 (16.89%) | 281 (5.82%) |
| LGBTQIA | 2,000 | 95 (4.75%) | 75 (3.75%) | 222 (11.10%) | 337 (16.85%) | 162 (8.10%) | 260 (13.00%) |
| Nationality | 504 | 14 (2.78%) | 19 (3.77%) | 36 (7.14%) | 28 (5.56%) | 20 (3.97%) | 51 (10.12%) |
| PhysicalAppearance | 3,528 | 132 (3.74%) | 77 (2.18%) | 407 (11.54%) | 407 (11.54%) | 160 (4.54%) | 525 (14.88%) |
| RaceEthnicity | 3,716 | 217 (5.84%) | 153 (4.12%) | 298 (8.02%) | 378 (10.17%) | 364 (9.80%) | 529 (14.24%) |
| Religion | 648 | 59 (9.10%) | 15 (2.31%) | 59 (9.10%) | 101 (15.59%) | 55 (8.49%) | 95 (14.66%) |
| SES | 4,204 | 123 (2.93%) | 84 (2.00%) | 264 (6.28%) | 331 (7.87%) | 207 (4.92%) | 331 (7.87%) |
| SpanishRegion | 988 | 39 (3.95%) | 20 (2.02%) | 44 (4.45%) | 93 (9.41%) | 63 (6.38%) | 99 (10.02%) |

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 3.41% / 2.63% | 2.24% / 2.02% | 10.29% / 12.75% | 11.30% / 12.61% | 6.58% / 6.70% | 13.93% / 16.35% |
| DisabilityStatus | 5.82% / 4.94% | 4.42% / 2.15% | 14.12% / 13.55% | 18.53% / 15.28% | 5.82% / 8.67% | 16.92% / 18.12% |
| Gender | 3.72% / 2.10% | 2.46% / 1.11% | 4.26% / 3.19% | 10.31% / 7.99% | 21.14% / 14.96% | 6.52% / 5.50% |
| LGBTQIA | 4.86% / 4.71% | 4.17% / 3.58% | 9.20% / 11.87% | 16.15% / 17.13% | 10.24% / 7.23% | 11.81% / 13.48% |
| Nationality | 4.17% / 2.08% | 7.14% / 2.08% | 10.12% / 5.65% | 10.12% / 3.27% | 5.36% / 3.27% | 10.71% / 9.82% |
| PhysicalAppearance | 5.61% / 2.81% | 4.08% / 1.23% | 13.61% / 10.50% | 14.20% / 10.20% | 8.50% / 2.55% | 16.58% / 14.03% |
| RaceEthnicity | 7.82% / 4.86% | 5.78% / 3.30% | 10.34% / 6.87% | 13.03% / 8.76% | 12.54% / 8.44% | 18.16% / 12.30% |
| Religion | 10.65% / 8.33% | 2.31% / 2.31% | 6.94% / 10.19% | 15.28% / 15.74% | 9.72% / 7.87% | 11.57% / 16.20% |
| SES | 3.41% / 2.69% | 2.90% / 1.56% | 5.87% / 6.48% | 8.04% / 7.79% | 7.39% / 3.72% | 7.32% / 8.14% |
| SpanishRegion | 6.48% / 2.71% | 3.09% / 1.51% | 8.95% / 2.26% | 20.68% / 3.92% | 12.35% / 3.46% | 21.91% / 4.22% |

</details>

## 3. Neuron and Changed-Item Overlap Between Sets

Pairwise count of shared zeroed neurons, neuron Jaccard, and changed-item Jaccard (all items and within each set's target category).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `exp13` vs. `exp14` | 8 / 20,20 | 0.250 | 0.563 | Religion: 0.766 | RaceEthnicity: 0.244 | L24.5999, L27.1055, L27.1348, L27.3313, L27.3655, L27.5392, L27.6849, L27.8108 |
| `exp14` vs. `exp12` | 5 / 20,5 | 0.250 | 0.558 | RaceEthnicity: 0.238 | RaceEthnicity: 0.238 | L27.1348, L27.3313, L27.3655, L27.4161, L27.6849 |
| `exp13` vs. `exp2` | 5 / 20,5 | 0.250 | 0.203 | Religion: 0.468 | Religion: 0.468 | L19.4699, L21.6606, L27.3313, L27.7070, L27.7279 |
| `exp11` vs. `exp10` | 5 / 20,5 | 0.250 | 0.166 | Gender: 0.053 | Gender: 0.053 | L25.430, L25.4753, L26.4583, L27.43, L27.2985 |
| `exp13` vs. `exp12` | 4 / 20,5 | 0.190 | 0.550 | Religion: 0.524 | RaceEthnicity: 0.454 | L27.1348, L27.3313, L27.3655, L27.6849 |
| `exp11` vs. `exp13` | 2 / 20,20 | 0.053 | 0.159 | Gender: 0.013 | Religion: 0.418 | L27.1055, L27.7070 |
| `exp11` vs. `exp2` | 1 / 20,5 | 0.042 | 0.224 | Gender: 0.088 | Religion: 0.652 | L27.7070 |
| `exp11` vs. `exp14` | 1 / 20,20 | 0.026 | 0.184 | Gender: 0.025 | RaceEthnicity: 0.402 | L27.1055 |
| `exp12` vs. `exp2` | 1 / 5,5 | 0.111 | 0.181 | RaceEthnicity: 0.184 | Religion: 0.513 | L27.3313 |
| `exp14` vs. `exp2` | 1 / 20,5 | 0.042 | 0.160 | RaceEthnicity: 0.162 | Religion: 0.413 | L27.3313 |
| `exp11` vs. `exp12` | 0 / 20,5 | 0.000 | 0.161 | Gender: 0.036 | RaceEthnicity: 0.098 | - |
| `exp10` vs. `exp2` | 0 / 5,5 | 0.000 | 0.161 | Gender: 0.212 | Religion: 0.138 | - |
| `exp10` vs. `exp12` | 0 / 5,5 | 0.000 | 0.122 | Gender: 0.099 | RaceEthnicity: 0.114 | - |
| `exp14` vs. `exp10` | 0 / 20,5 | 0.000 | 0.106 | RaceEthnicity: 0.192 | Gender: 0.060 | - |
| `exp13` vs. `exp10` | 0 / 20,5 | 0.000 | 0.091 | Religion: 0.084 | Gender: 0.042 | - |

## 4. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

Secondary benchmark metrics for target categories (`disambig` and `ambig` contexts, state-based S/A/U transitions); descriptive and not primary localization measures.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | EsBBQ Bias Δ (95% CI) | BBQ-Formula Disamb Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: | ---: |
| Gender | `exp10` | ambig | 37 / 1,504 (2.46%) | 18 / 18 | 0.00% | 0 / 1 | 0.07 [-1.46, 1.60] | - |
| Gender | `exp10` | disambig | 37 / 3,328 (1.11%) | 17 / 20 | -0.09% | 17 / 20 | -0.18 [-0.90, 0.56] | -0.18 [-0.90, 0.54] |
| Gender | `exp11` | ambig | 318 / 1,504 (21.14%) | 159 / 152 | 0.47% | 0 / 7 | 1.13 [-3.32, 5.78] | - |
| Gender | `exp11` | disambig | 498 / 3,328 (14.96%) | 253 / 245 | 0.24% | 239 / 259 | 0.48 [-1.98, 3.23] | 0.48 [-1.98, 3.25] |
| Gender | `exp12` | ambig | 64 / 1,504 (4.26%) | 25 / 36 | -0.73% | 0 / 3 | -1.40 [-3.66, 0.66] | - |
| Gender | `exp12` | disambig | 106 / 3,328 (3.19%) | 49 / 57 | -0.24% | 55 / 51 | -0.48 [-1.68, 0.72] | -0.48 [-1.68, 0.72] |
| Gender | `exp13` | ambig | 155 / 1,504 (10.31%) | 62 / 90 | -1.86% | 0 / 3 | -3.66 [-6.98, -0.47] | - |
| Gender | `exp13` | disambig | 266 / 3,328 (7.99%) | 115 / 151 | -1.08% | 171 / 95 | -2.16 [-3.96, -0.27] | -2.16 [-3.91, -0.24] |
| Gender | `exp14` | ambig | 98 / 1,504 (6.52%) | 40 / 55 | -1.00% | 0 / 3 | -1.93 [-4.46, 0.67] | - |
| Gender | `exp14` | disambig | 183 / 3,328 (5.50%) | 81 / 100 | -0.57% | 103 / 78 | -1.14 [-2.63, 0.47] | -1.14 [-2.64, 0.48] |
| Gender | `exp2` | ambig | 56 / 1,504 (3.72%) | 17 / 37 | -1.33% | 0 / 2 | -2.66 [-4.66, -0.80] | - |
| Gender | `exp2` | disambig | 70 / 3,328 (2.10%) | 27 / 43 | -0.48% | 35 / 35 | -0.96 [-1.92, 0.06] | -0.96 [-1.92, 0.06] |
| RaceEthnicity | `exp10` | ambig | 71 / 1,228 (5.78%) | 32 / 29 | 0.24% | 8 / 2 | 0.33 [-2.20, 2.93] | - |
| RaceEthnicity | `exp10` | disambig | 82 / 2,488 (3.30%) | 32 / 42 | -0.40% | 41 / 39 | -0.64 [-2.10, 0.84] | -0.82 [-2.22, 0.62] |
| RaceEthnicity | `exp11` | ambig | 154 / 1,228 (12.54%) | 76 / 63 | 1.06% | 8 / 7 | 2.36 [-1.55, 6.19] | - |
| RaceEthnicity | `exp11` | disambig | 210 / 2,488 (8.44%) | 87 / 91 | -0.16% | 101 / 101 | 0.80 [-1.44, 3.14] | 0.48 [-1.74, 2.78] |
| RaceEthnicity | `exp12` | ambig | 127 / 1,228 (10.34%) | 49 / 44 | 0.41% | 29 / 5 | 0.98 [-2.04, 4.40] | - |
| RaceEthnicity | `exp12` | disambig | 171 / 2,488 (6.87%) | 74 / 70 | 0.16% | 90 / 76 | 0.96 [-1.06, 2.99] | 0.76 [-1.29, 2.69] |
| RaceEthnicity | `exp13` | ambig | 160 / 1,228 (13.03%) | 71 / 53 | 1.47% | 33 / 3 | 2.93 [-0.33, 6.68] | - |
| RaceEthnicity | `exp13` | disambig | 218 / 2,488 (8.76%) | 92 / 91 | 0.04% | 112 / 99 | 0.88 [-1.34, 3.30] | 0.74 [-1.47, 2.96] |
| RaceEthnicity | `exp14` | ambig | 223 / 1,228 (18.16%) | 100 / 85 | 1.22% | 32 / 6 | 1.95 [-2.44, 6.43] | - |
| RaceEthnicity | `exp14` | disambig | 306 / 2,488 (12.30%) | 142 / 124 | 0.72% | 166 / 129 | 2.65 [0.13, 5.54] | 2.11 [-0.40, 4.85] |
| RaceEthnicity | `exp2` | ambig | 96 / 1,228 (7.82%) | 49 / 33 | 1.30% | 10 / 4 | 2.61 [0.00, 5.95] | - |
| RaceEthnicity | `exp2` | disambig | 121 / 2,488 (4.86%) | 75 / 44 | 1.25% | 61 / 60 | 2.49 [0.80, 4.26] | 2.55 [0.82, 4.34] |
| Religion | `exp10` | ambig | 5 / 216 (2.31%) | 1 / 2 | -0.46% | 1 / 1 | -0.93 [-4.17, 2.31] | - |
| Religion | `exp10` | disambig | 10 / 432 (2.31%) | 8 / 2 | 1.39% | 4 / 6 | 2.78 [0.21, 5.87] | 2.78 [0.00, 6.02] |
| Religion | `exp11` | ambig | 21 / 216 (9.72%) | 6 / 6 | 0.00% | 4 / 5 | 0.46 [-6.02, 7.41] | - |
| Religion | `exp11` | disambig | 34 / 432 (7.87%) | 25 / 9 | 3.70% | 17 / 17 | 7.41 [2.35, 12.56] | 7.41 [2.31, 12.50] |
| Religion | `exp12` | ambig | 15 / 216 (6.94%) | 2 / 5 | -1.39% | 4 / 4 | -1.85 [-6.94, 3.25] | - |
| Religion | `exp12` | disambig | 44 / 432 (10.19%) | 31 / 13 | 4.17% | 21 / 23 | 8.33 [2.70, 14.40] | 8.33 [2.78, 14.35] |
| Religion | `exp13` | ambig | 33 / 216 (15.28%) | 10 / 8 | 0.93% | 11 / 4 | 4.17 [-4.63, 12.96] | - |
| Religion | `exp13` | disambig | 68 / 432 (15.74%) | 47 / 21 | 6.02% | 31 / 37 | 12.04 [4.66, 19.40] | 12.04 [4.63, 19.44] |
| Religion | `exp14` | ambig | 25 / 216 (11.57%) | 10 / 5 | 2.31% | 4 / 6 | 6.48 [-0.93, 14.81] | - |
| Religion | `exp14` | disambig | 70 / 432 (16.20%) | 49 / 21 | 6.48% | 33 / 37 | 12.96 [5.57, 20.71] | 12.96 [5.56, 20.83] |
| Religion | `exp2` | ambig | 23 / 216 (10.65%) | 8 / 6 | 0.93% | 5 / 4 | 1.39 [-6.02, 9.26] | - |
| Religion | `exp2` | disambig | 36 / 432 (8.33%) | 27 / 9 | 4.17% | 15 / 21 | 8.33 [3.35, 13.90] | 8.33 [3.24, 13.89] |

## 5. Baseline Log-Likelihood Margin Mechanism

Median baseline margin (`top − second`) of changed versus unchanged items (`ALL` categories) and flip rate across baseline-margin deciles 1–10 (decile 1 = most undecided).

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `exp2` | real | 1,070 | 0.125 | 1.078 |
| `exp10` | real | 684 | 0.094 | 1.055 |
| `exp12` | real | 2,376 | 0.250 | 1.125 |
| `exp13` | real | 3,055 | 0.250 | 1.156 |
| `exp11` | real | 2,337 | 0.219 | 1.125 |
| `exp14` | real | 3,307 | 0.312 | 1.156 |

| Decile | Margin Range | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.172] | 28.51% | 19.88% | 36.97% | 42.64% | 39.02% | 40.81% |
| 2 | [0.172, 0.375] | 8.16% | 3.70% | 19.00% | 29.54% | 23.35% | 26.83% |
| 3 | [0.375, 0.562] | 1.65% | 1.17% | 12.92% | 16.80% | 14.06% | 19.25% |
| 4 | [0.562, 0.766] | 0.59% | 0.18% | 8.49% | 10.76% | 5.89% | 12.41% |
| 5 | [0.766, 1.008] | 0.26% | 0.11% | 5.56% | 6.84% | 1.90% | 10.43% |
| 6 | [1.008, 1.297] | 0.00% | 0.00% | 2.75% | 3.59% | 0.66% | 6.77% |
| 7 | [1.297, 1.641] | 0.00% | 0.00% | 1.17% | 1.50% | 0.22% | 3.18% |
| 8 | [1.641, 2.125] | 0.00% | 0.00% | 0.11% | 0.15% | 0.29% | 1.32% |
| 9 | [2.125, 2.875] | 0.00% | 0.00% | 0.00% | 0.00% | 0.07% | 0.04% |
| 10 | [2.875, 16.125] | 0.00% | 0.00% | 0.00% | 0.00% | 0.07% | 0.00% |

## 6. Granular Breakdowns (Demographic Groups and Topics)

Flip rates disaggregated by mentioned demographic group (`answer_info`) and, for EsBBQ, by item topic (`relevant_social_value`).

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f | 4,832 | 2.61% | 1.53% | 3.52% | 8.71% | 16.89% | 5.82% |
| m | 4,832 | 2.61% | 1.53% | 3.52% | 8.71% | 16.89% | 5.82% |

### RaceEthnicity — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| africano | 124 | 2.42% | 9.68% | 16.13% | 22.58% | 17.74% | 21.77% |
| asiático | 264 | 12.50% | 2.27% | 15.53% | 12.88% | 6.82% | 18.94% |
| blanco | 3,716 | 5.84% | 4.12% | 8.02% | 10.17% | 9.80% | 14.24% |
| de Europa del Este | 396 | 4.80% | 4.04% | 5.05% | 6.57% | 12.12% | 21.97% |
| gitano | 372 | 1.61% | 5.38% | 10.75% | 6.99% | 6.72% | 12.90% |
| indio | 264 | 6.06% | 1.52% | 6.06% | 6.44% | 6.06% | 11.36% |
| latino | 672 | 2.38% | 5.36% | 7.29% | 5.95% | 5.80% | 10.27% |
| negro | 748 | 4.55% | 2.27% | 6.55% | 11.10% | 10.70% | 10.83% |
| árabe | 876 | 10.27% | 4.79% | 7.19% | 14.16% | 13.24% | 15.64% |

### Religion — Groups

| Group | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ateo | 216 | 9.72% | 1.85% | 6.48% | 19.44% | 9.72% | 18.06% |
| católico | 408 | 8.33% | 2.45% | 7.60% | 13.48% | 7.35% | 13.24% |
| judío | 48 | 6.25% | 0.00% | 4.17% | 6.25% | 4.17% | 4.17% |
| musulmán | 144 | 4.17% | 1.39% | 6.25% | 8.33% | 5.56% | 9.72% |
| no creyente | 216 | 11.11% | 3.70% | 13.89% | 18.52% | 10.19% | 17.59% |
| testigo de Jehová | 264 | 11.36% | 2.27% | 12.12% | 18.94% | 10.23% | 16.29% |

</details>

<details>
<summary>Expand EsBBQ topic breakdown (`relevant_social_value`)</summary>

### Gender — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cocinar | 240 | 2.92% | 0.83% | 3.33% | 5.83% | 19.17% | 4.17% |
| conducción temeraria o imprudente | 60 | 3.33% | 1.67% | 1.67% | 8.33% | 16.67% | 3.33% |
| debilidad, fragilidad, menos fuerza | 60 | 8.33% | 3.33% | 1.67% | 13.33% | 26.67% | 8.33% |
| emocional | 120 | 1.67% | 0.83% | 0.00% | 5.00% | 15.00% | 1.67% |
| gustos asociados a un género: coches | 120 | 1.67% | 0.83% | 1.67% | 5.00% | 12.50% | 6.67% |
| gustos asociados a un género: color azul | 160 | 2.50% | 1.25% | 3.12% | 11.88% | 10.00% | 5.62% |
| gustos asociados a un género: color rosa | 160 | 2.50% | 0.00% | 5.62% | 8.75% | 8.75% | 8.12% |
| gustos asociados a un género: fútbol | 72 | 0.00% | 0.00% | 2.78% | 5.56% | 8.33% | 5.56% |
| gustos asociados a un género: muñecas | 120 | 0.83% | 0.83% | 0.00% | 1.67% | 11.67% | 0.83% |
| inestabilidad emocional | 120 | 5.00% | 2.50% | 5.83% | 9.17% | 25.00% | 6.67% |
| inferioridad | 120 | 1.67% | 0.83% | 2.50% | 5.83% | 18.33% | 3.33% |
| ira, agresividad, violencia | 160 | 0.62% | 1.88% | 5.00% | 10.00% | 17.50% | 6.88% |
| mala conducción | 180 | 1.67% | 3.89% | 1.11% | 6.11% | 12.78% | 3.89% |
| malas entre ellas | 36 | 2.78% | 0.00% | 11.11% | 11.11% | 2.78% | 13.89% |
| mayor carga familiar | 240 | 3.75% | 2.92% | 3.33% | 7.50% | 22.50% | 5.42% |
| menor inteligencia | 60 | 1.67% | 0.00% | 6.67% | 11.67% | 15.00% | 8.33% |
| menor proactividad, liderazgo | 60 | 11.67% | 1.67% | 1.67% | 11.67% | 30.00% | 6.67% |
| no llorar | 180 | 2.22% | 1.67% | 2.78% | 6.67% | 12.22% | 3.89% |
| no ser alto cargo | 120 | 4.17% | 5.00% | 2.50% | 9.17% | 23.33% | 5.00% |
| peor con la tecnología | 120 | 4.17% | 1.67% | 3.33% | 15.00% | 15.83% | 10.83% |
| peor en STEM | 240 | 1.67% | 1.67% | 4.17% | 11.67% | 20.42% | 9.58% |
| peor en deportes | 72 | 2.78% | 1.39% | 2.78% | 13.89% | 13.89% | 11.11% |
| peor en matemáticas | 280 | 3.21% | 1.79% | 4.64% | 8.57% | 14.64% | 6.07% |
| persona cerrada emocionalmente, falta de empatía | 240 | 1.67% | 1.25% | 2.50% | 10.00% | 12.50% | 4.17% |
| persona exagerada | 180 | 1.67% | 1.11% | 2.78% | 6.67% | 22.78% | 3.89% |
| priorizar la familia sobre la carrera profesional, mayor carga familiar | 200 | 2.50% | 0.50% | 3.50% | 4.50% | 29.00% | 3.50% |
| priorizar la maternidad sobre la carrera profesional | 100 | 3.00% | 1.00% | 2.00% | 12.00% | 16.00% | 7.00% |
| profesión feminizada: maestra | 120 | 0.00% | 0.00% | 4.17% | 7.50% | 4.17% | 7.50% |
| profesión feminizada: secretaria | 232 | 3.02% | 1.72% | 6.47% | 14.22% | 22.41% | 6.90% |
| profesión masculinizada: informática | 120 | 0.00% | 0.00% | 0.00% | 5.00% | 14.17% | 1.67% |
| profesión masculinizada: ingeniería | 120 | 2.50% | 1.67% | 4.17% | 9.17% | 10.83% | 6.67% |
| querer tener hijos | 120 | 0.83% | 0.83% | 5.00% | 11.67% | 15.83% | 9.17% |
| sensible | 60 | 6.67% | 3.33% | 5.00% | 5.00% | 20.00% | 3.33% |
| violencia de género (maltratador) | 180 | 3.89% | 1.67% | 5.00% | 11.11% | 20.00% | 6.11% |
| víctima de violencia de género | 60 | 5.00% | 3.33% | 8.33% | 10.00% | 13.33% | 10.00% |

### RaceEthnicity — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acoso / agresión sexual | 12 | 0.00% | 33.33% | 0.00% | 33.33% | 33.33% | 50.00% |
| alcoholismo | 72 | 5.56% | 19.44% | 12.50% | 9.72% | 9.72% | 23.61% |
| analfabetismo tecnológico | 24 | 8.33% | 4.17% | 20.83% | 20.83% | 8.33% | 12.50% |
| ayudas sociales | 504 | 5.95% | 3.77% | 5.95% | 8.13% | 11.71% | 13.49% |
| bajo rendimiento académico | 432 | 7.87% | 2.08% | 6.71% | 12.04% | 11.11% | 10.88% |
| buscar dinero en relaciones | 72 | 1.39% | 2.78% | 5.56% | 2.78% | 1.39% | 1.39% |
| casarse y tener relaciones sexuales a edades tempranas | 24 | 0.00% | 4.17% | 4.17% | 0.00% | 8.33% | 12.50% |
| consumo de drogas | 96 | 6.25% | 8.33% | 6.25% | 11.46% | 12.50% | 19.79% |
| delincuencia, robar | 216 | 5.56% | 5.09% | 5.09% | 8.33% | 7.41% | 13.89% |
| docilidad | 72 | 12.50% | 4.17% | 12.50% | 9.72% | 6.94% | 19.44% |
| español como lengua extranjera | 168 | 5.95% | 5.95% | 18.45% | 14.29% | 3.57% | 17.26% |
| falta de acceso a la tecnología | 24 | 4.17% | 8.33% | 4.17% | 8.33% | 4.17% | 4.17% |
| falta de derechos de la mujer, machismo | 144 | 5.56% | 7.64% | 6.94% | 11.11% | 13.19% | 18.06% |
| hipersexualización, prostitución | 72 | 0.00% | 5.56% | 11.11% | 8.33% | 18.06% | 23.61% |
| inmigración ilegal | 132 | 4.55% | 2.27% | 5.30% | 11.36% | 5.30% | 9.09% |
| ira, agresividad, violencia | 120 | 0.83% | 1.67% | 2.50% | 3.33% | 8.33% | 7.50% |
| mal comportamiento, persona problemática | 120 | 1.67% | 1.67% | 5.83% | 2.50% | 1.67% | 5.00% |
| mal olor | 72 | 6.94% | 5.56% | 13.89% | 11.11% | 4.17% | 8.33% |
| mala gente, no ser de fiar | 24 | 0.00% | 0.00% | 8.33% | 4.17% | 4.17% | 8.33% |
| mayor presión parental | 48 | 12.50% | 6.25% | 8.33% | 8.33% | 8.33% | 10.42% |
| meritocracia | 744 | 7.53% | 1.61% | 7.53% | 8.87% | 8.87% | 11.69% |
| pobreza | 152 | 2.63% | 6.58% | 14.47% | 24.34% | 21.71% | 31.58% |
| promiscuidad | 36 | 2.78% | 0.00% | 5.56% | 13.89% | 0.00% | 25.00% |
| sumisión | 24 | 0.00% | 4.17% | 12.50% | 4.17% | 4.17% | 12.50% |
| terrorismo | 168 | 10.71% | 4.17% | 8.93% | 13.10% | 14.88% | 18.45% |
| vagancia, no querer trabajar | 96 | 0.00% | 5.21% | 4.17% | 4.17% | 4.17% | 14.58% |
| violencia de género | 48 | 2.08% | 10.42% | 18.75% | 27.08% | 27.08% | 33.33% |

### Religion — Topics

| Topic | Items | `exp2` | `exp10` | `exp12` | `exp13` | `exp11` | `exp14` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| abuso sexual infantil | 48 | 2.08% | 0.00% | 4.17% | 6.25% | 2.08% | 8.33% |
| anticiencia | 336 | 9.52% | 2.98% | 10.42% | 19.35% | 9.52% | 18.15% |
| avaricia | 48 | 6.25% | 0.00% | 4.17% | 6.25% | 4.17% | 4.17% |
| celebración de tradiciones religiosas | 48 | 10.42% | 2.08% | 8.33% | 8.33% | 4.17% | 4.17% |
| falta de derechos de la mujer, machismo | 60 | 13.33% | 0.00% | 10.00% | 16.67% | 11.67% | 16.67% |
| persona sentenciosa, moralista | 48 | 14.58% | 4.17% | 8.33% | 16.67% | 10.42% | 14.58% |
| poligamia | 24 | 0.00% | 4.17% | 8.33% | 12.50% | 4.17% | 12.50% |
| terrorismo | 36 | 8.33% | 2.78% | 11.11% | 13.89% | 13.89% | 16.67% |

</details>
