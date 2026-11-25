# E3 Unit Dossier — `llama-3.2-3b_en` (Llama-3.2-3B, EN, BBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **BBQ** (58,492 items, 11 categories) · Source: `results/e3/llama-3.2-3b_en/analysis/20261124_a100_items_3b_random`.

## 1. Set Summary and Target-Category Localization

**Why this check:** summarizes how strongly each selected (and ablation) neuron set concentrates answer changes (`pred != base_pred`) in its target category versus the mean of the other categories, together with its layer profile and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | ---: | :--- | :--- |
| `real_gender20` | real | Gender_identity | 20 | 504 / 5,672 (8.89%) | [8.16, 9.61] | 4.17% | 2.13× | 2,912 / 58,492 (4.98%) | A100-SXM4-80GB | L8:1, L14:1, L16:2, L19:1, L21:1, L22:2, L24:2, L25:1, L26:3, L27:6 |
| `real_race20` | real | Race_ethnicity | 20 | 857 / 6,880 (12.46%) | [11.72, 13.24] | 8.03% | 1.55× | 5,857 / 58,492 (10.01%) | A100-SXM4-80GB | L23:1, L24:1, L27:18 |
| `real_religion20` | real | Religion | 20 | 133 / 1,200 (11.08%) | [9.50, 12.84] | 9.61% | 1.15× | 6,261 / 58,492 (10.70%) | A100-SXM4-80GB | L17:1, L19:1, L21:1, L23:1, L24:1, L26:1, L27:14 |
| `real_gender5` | real | Gender_identity | 5 | 75 / 5,672 (1.32%) | [1.02, 1.60] | 1.46% | 0.90× | 887 / 58,492 (1.52%) | A100-SXM4-80GB | L14:1, L24:2, L26:2 |
| `real_race5` | real | Race_ethnicity | 5 | 329 / 6,880 (4.78%) | [4.26, 5.29] | 4.26% | 1.12× | 2,417 / 58,492 (4.13%) | A100-SXM4-80GB | L27:5 |
| `real_religion5` | real | Religion | 5 | 64 / 1,200 (5.33%) | [4.17, 6.58] | 7.81% | 0.68× | 5,075 / 58,492 (8.68%) | A100-SXM4-80GB | L19:1, L27:4 |

*Legend — **Kind**: `real` (selected top-K) or `ablation` (exploratory subset); **Target Flips**: changed answers (`pred != base_pred`) in target category; **95% CI (%)**: paired item-bootstrap interval (`B = 1,000`); **Other Mean**: mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **Layer Profile**: `L<layer>:<count>` of zeroed neurons.*

## 2. Selected vs. Layer-Matched Random Neurons

**Why this check:** tests whether the BiasScore-selected neurons change substantially more target-category answers than uniform random neuron sets drawn from the exact same MLP layers.

| Comparison | Target Cat. | Selected Flips | Random Flips (min–max, mean) | Real/Rand | Rank | GPUs |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| Gender·20 (seeds 0-4, A100-only) | Gender_identity | 504 / 5,672 (8.89%) | 70–94 (1.23–1.66%, mean 1.46%) | 6.10× | 1 of 6 | A100-SXM4-80GB / A100-SXM4-80GB |
| Religion·20 (seeds 0-9, A100-only) | Religion | 133 / 1,200 (11.08%) | 11–16 (0.92–1.33%, mean 1.17%) | 9.43× | 1 of 11 | A100-SXM4-80GB / A100-SXM4-40GB; A100-SXM4-80GB |
| Religion·20 (L4 seeds 0-1 + A100 seeds 2-9, sensitivity) | Religion | 133 / 1,200 (11.08%) | 13–20 (1.08–1.67%, mean 1.29%) | 8.58× | 1 of 11 | A100-SXM4-80GB / A100-SXM4-40GB; L4 |
| Gender·20 (L4 seed0 + A100 seeds 1-4, sensitivity) | Gender_identity | 504 / 5,672 (8.89%) | 77–94 (1.36–1.66%, mean 1.53%) | 5.81× | 1 of 6 | A100-SXM4-80GB / A100-SXM4-80GB; L4 |

*Legend — **Selected Flips**: target-category changed answers (`changed / n_items`, `%`, `pred != base_pred`); **Random Flips (min–max, mean)**: range across the layer-matched random seeds reported as `min–max changed items (min–max%, mean %)` over the same `n_items` target denominator; **Real/Rand**: `Selected % / Random mean %`; **Rank**: position of selected set among selected + random (`1` = highest).*

## 3. Changed Answers Across All Categories (`Set × Category`)

**Why this table:** exposes the full `set × category` flip matrix (`pred != base_pred`, all items pooled; random families summarized by mean) so off-target spillover across individual categories can be inspected directly.

| Category | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `rand_gender20` (mean) | `rand_religion20` (mean) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 3,680 | 209 (5.68%) | 52 (1.41%) | 288 (7.83%) | 160 (4.35%) | 343 (9.32%) | 283 (7.69%) | 1.99% [1.68–2.72] | 1.66% [1.28–2.12] |
| Disability_status | 1,556 | 45 (2.89%) | 6 (0.39%) | 63 (4.05%) | 21 (1.35%) | 93 (5.98%) | 72 (4.63%) | 0.53% [0.26–0.77] | 0.50% [0.26–0.77] |
| Gender_identity | 5,672 | 504 (8.89%) | 75 (1.32%) | 440 (7.76%) | 169 (2.98%) | 538 (9.49%) | 545 (9.61%) | 1.46% [1.23–1.66] | 1.42% [1.08–1.89] |
| Nationality | 3,080 | 104 (3.38%) | 46 (1.49%) | 340 (11.04%) | 198 (6.43%) | 379 (12.31%) | 239 (7.76%) | 1.55% [1.43–1.79] | 1.52% [1.33–2.05] |
| Physical_appearance | 1,576 | 95 (6.03%) | 33 (2.09%) | 109 (6.92%) | 81 (5.14%) | 155 (9.84%) | 124 (7.87%) | 1.97% [1.59–2.22] | 2.10% [1.46–4.19] |
| Race_ethnicity | 6,880 | 282 (4.10%) | 96 (1.40%) | 857 (12.46%) | 329 (4.78%) | 740 (10.76%) | 633 (9.20%) | 1.79% [1.45–2.05] | 1.65% [1.29–2.01] |
| Race_x_SES | 11,160 | 523 (4.69%) | 211 (1.89%) | 1,096 (9.82%) | 405 (3.63%) | 1,393 (12.48%) | 977 (8.75%) | 1.88% [1.66–2.19] | 1.76% [1.56–2.33] |
| Race_x_gender | 15,960 | 756 (4.74%) | 255 (1.60%) | 1,958 (12.27%) | 656 (4.11%) | 1,719 (10.77%) | 1,555 (9.74%) | 1.84% [1.52–2.21] | 1.66% [1.40–2.06] |
| Religion | 1,200 | 26 (2.17%) | 13 (1.08%) | 89 (7.42%) | 73 (6.08%) | 133 (11.08%) | 64 (5.33%) | 1.37% [1.17–1.67] | 1.17% [0.92–1.33] |
| SES | 6,864 | 342 (4.98%) | 82 (1.19%) | 575 (8.38%) | 287 (4.18%) | 729 (10.62%) | 540 (7.87%) | 1.40% [1.15–1.54] | 1.40% [1.06–1.70] |
| Sexual_orientation | 864 | 26 (3.01%) | 18 (2.08%) | 42 (4.86%) | 38 (4.40%) | 39 (4.51%) | 43 (4.98%) | 1.53% [1.04–1.97] | 1.64% [1.04–2.31] |

*Legend — Cells show `changed (flips_pct)` (`pred != base_pred`, pooled `ambig + disambig`); random-family columns show `mean% [min–max]` across seeds.*

<details>
<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>

Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.

| Category | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Age | 6.79% / 4.57% | 1.74% / 1.09% | 8.37% / 7.28% | 4.57% / 4.13% | 10.38% / 8.26% | 9.13% / 6.25% |
| Disability_status | 3.86% / 1.93% | 0.77% / 0.00% | 5.27% / 2.83% | 1.54% / 1.16% | 8.35% / 3.60% | 6.56% / 2.70% |
| Gender_identity | 11.14% / 6.63% | 1.48% / 1.16% | 9.52% / 5.99% | 3.67% / 2.29% | 11.67% / 7.30% | 11.64% / 7.58% |
| Nationality | 4.03% / 2.73% | 2.08% / 0.91% | 14.94% / 7.14% | 9.42% / 3.44% | 15.97% / 8.64% | 9.61% / 5.91% |
| Physical_appearance | 6.35% / 5.71% | 1.78% / 2.41% | 7.23% / 6.60% | 4.82% / 5.46% | 9.90% / 9.77% | 8.63% / 7.11% |
| Race_ethnicity | 4.62% / 3.58% | 1.72% / 1.08% | 14.91% / 10.00% | 5.64% / 3.92% | 12.73% / 8.78% | 10.76% / 7.65% |
| Race_x_SES | 4.21% / 3.03% | 1.86% / 1.15% | 8.49% / 7.81% | 3.10% / 2.60% | 11.59% / 9.41% | 7.87% / 6.61% |
| Race_x_gender | 4.30% / 2.28% | 1.47% / 0.65% | 11.37% / 6.82% | 4.12% / 2.09% | 10.09% / 5.50% | 8.71% / 5.04% |
| Religion | 2.50% / 1.83% | 1.17% / 1.00% | 9.00% / 5.83% | 6.67% / 5.50% | 12.67% / 9.50% | 6.83% / 3.83% |
| SES | 6.00% / 3.96% | 1.40% / 0.99% | 10.05% / 6.70% | 5.13% / 3.23% | 12.65% / 8.60% | 9.56% / 6.18% |
| Sexual_orientation | 2.55% / 3.47% | 2.08% / 2.08% | 5.32% / 4.40% | 3.94% / 4.86% | 4.86% / 4.17% | 5.09% / 4.86% |

*Legend — Cells show `ambig% / disambig%` (S/A/U state changes from `direction.csv`).*

</details>

## 4. Neuron and Changed-Item Overlap Between Sets

**Why this check:** tests whether distinct target categories share the same high-BiasScore neurons and whether different interventions flip the same underlying items (pairwise shared neurons, neuron Jaccard, and changed-item Jaccard).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `real_religion20` vs. `real_religion5` | 5 / 20,5 | 0.250 | 0.540 | Religion: 0.239 | Religion: 0.239 | L19.4699, L27.672, L27.1348, L27.5392, L27.7597 |
| `real_race20` vs. `real_religion20` | 5 / 20,20 | 0.143 | 0.386 | Race_ethnicity: 0.324 | Religion: 0.451 | L27.672, L27.1348, L27.5392, L27.7597, L27.7673 |
| `real_race20` vs. `real_race5` | 5 / 20,5 | 0.250 | 0.201 | Race_ethnicity: 0.204 | Race_ethnicity: 0.204 | L27.389, L27.3286, L27.3313, L27.3692, L27.3722 |
| `real_gender20` vs. `real_gender5` | 5 / 20,5 | 0.250 | 0.125 | Gender_identity: 0.055 | Gender_identity: 0.055 | L14.2374, L24.953, L24.4032, L26.1688, L26.5442 |
| `real_race20` vs. `real_religion5` | 4 / 20,5 | 0.190 | 0.502 | Race_ethnicity: 0.483 | Religion: 0.297 | L27.672, L27.1348, L27.5392, L27.7597 |
| `real_gender20` vs. `real_religion20` | 2 / 20,20 | 0.053 | 0.257 | Gender_identity: 0.453 | Religion: 0.112 | L27.672, L27.1055 |
| `real_gender20` vs. `real_religion5` | 1 / 20,5 | 0.042 | 0.315 | Gender_identity: 0.482 | Religion: 0.216 | L27.672 |
| `real_gender20` vs. `real_race20` | 1 / 20,20 | 0.026 | 0.235 | Gender_identity: 0.274 | Race_ethnicity: 0.158 | L27.672 |
| `real_gender5` vs. `real_race5` | 0 / 5,5 | 0.000 | 0.140 | Gender_identity: 0.104 | Race_ethnicity: 0.136 | - |
| `real_gender20` vs. `real_race5` | 0 / 20,5 | 0.000 | 0.134 | Gender_identity: 0.125 | Race_ethnicity: 0.127 | - |
| `real_race5` vs. `real_religion5` | 0 / 5,5 | 0.000 | 0.127 | Race_ethnicity: 0.136 | Religion: 0.046 | - |
| `real_religion20` vs. `real_race5` | 0 / 20,5 | 0.000 | 0.115 | Religion: 0.234 | Race_ethnicity: 0.115 | - |
| `real_gender5` vs. `real_religion5` | 0 / 5,5 | 0.000 | 0.077 | Gender_identity: 0.047 | Religion: 0.085 | - |
| `real_race20` vs. `real_gender5` | 0 / 20,5 | 0.000 | 0.068 | Race_ethnicity: 0.057 | Gender_identity: 0.066 | - |
| `real_religion20` vs. `real_gender5` | 0 / 20,5 | 0.000 | 0.066 | Religion: 0.043 | Gender_identity: 0.053 | - |

*Legend — **Shared Neurons**: identical `(layer, index)` neurons (`shared / size_A, size_B`); **Neuron Jaccard**: `|A ∩ B| / |A ∪ B|` over zeroed neurons; **Items Jacc (All / Target A / Target B)**: Jaccard overlap of changed items across all items or restricted to each set's target category.*

## 5. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

**Why this table:** reports secondary directional metrics (BBQ/EsBBQ bias-score deltas, stereotype transitions `A→S`/`S→A`, and accuracy transitions) for descriptive completeness; these are secondary to answer-change localization.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | Disamb Bias Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: |
| Gender_identity | `real_gender20` | ambig | 316 / 2,836 (11.14%) | 151 / 165 | -0.49% | 0 / 0 | - |
| Gender_identity | `real_gender20` | disambig | 188 / 2,836 (6.63%) | 81 / 107 | -0.92% | 95 / 93 | -1.83 [-3.81, 0.07] |
| Gender_identity | `real_gender5` | ambig | 42 / 2,836 (1.48%) | 19 / 23 | -0.14% | 0 / 0 | - |
| Gender_identity | `real_gender5` | disambig | 33 / 2,836 (1.16%) | 20 / 13 | 0.25% | 17 / 16 | 0.49 [-0.28, 1.27] |
| Gender_identity | `real_race20` | ambig | 270 / 2,836 (9.52%) | 127 / 143 | -0.56% | 0 / 0 | - |
| Gender_identity | `real_race20` | disambig | 170 / 2,836 (5.99%) | 96 / 74 | 0.78% | 91 / 79 | 1.55 [-0.28, 3.31] |
| Gender_identity | `real_race5` | ambig | 104 / 2,836 (3.67%) | 53 / 51 | 0.07% | 0 / 0 | - |
| Gender_identity | `real_race5` | disambig | 65 / 2,836 (2.29%) | 29 / 36 | -0.25% | 38 / 27 | -0.49 [-1.62, 0.63] |
| Gender_identity | `real_religion20` | ambig | 331 / 2,836 (11.67%) | 154 / 177 | -0.81% | 0 / 0 | - |
| Gender_identity | `real_religion20` | disambig | 207 / 2,836 (7.30%) | 106 / 101 | 0.18% | 107 / 100 | 0.35 [-1.76, 2.26] |
| Gender_identity | `real_religion5` | ambig | 330 / 2,836 (11.64%) | 153 / 177 | -0.85% | 0 / 0 | - |
| Gender_identity | `real_religion5` | disambig | 215 / 2,836 (7.58%) | 107 / 108 | -0.04% | 114 / 101 | -0.07 [-2.12, 1.90] |
| Race_ethnicity | `real_gender20` | ambig | 159 / 3,440 (4.62%) | 76 / 83 | -0.20% | 0 / 0 | - |
| Race_ethnicity | `real_gender20` | disambig | 123 / 3,440 (3.58%) | 59 / 64 | -0.15% | 67 / 56 | -0.29 [-1.51, 1.05] |
| Race_ethnicity | `real_gender5` | ambig | 59 / 3,440 (1.72%) | 32 / 27 | 0.15% | 0 / 0 | - |
| Race_ethnicity | `real_gender5` | disambig | 37 / 3,440 (1.08%) | 16 / 21 | -0.15% | 21 / 16 | -0.29 [-0.93, 0.35] |
| Race_ethnicity | `real_race20` | ambig | 513 / 3,440 (14.91%) | 243 / 270 | -0.78% | 0 / 0 | - |
| Race_ethnicity | `real_race20` | disambig | 344 / 3,440 (10.00%) | 187 / 157 | 0.87% | 200 / 144 | 1.74 [-0.41, 3.78] |
| Race_ethnicity | `real_race5` | ambig | 194 / 3,440 (5.64%) | 100 / 94 | 0.17% | 0 / 0 | - |
| Race_ethnicity | `real_race5` | disambig | 135 / 3,440 (3.92%) | 75 / 60 | 0.44% | 77 / 58 | 0.87 [-0.35, 2.15] |
| Race_ethnicity | `real_religion20` | ambig | 438 / 3,440 (12.73%) | 229 / 209 | 0.58% | 0 / 0 | - |
| Race_ethnicity | `real_religion20` | disambig | 302 / 3,440 (8.78%) | 160 / 142 | 0.52% | 179 / 123 | 1.05 [-0.93, 3.02] |
| Race_ethnicity | `real_religion5` | ambig | 370 / 3,440 (10.76%) | 187 / 183 | 0.12% | 0 / 0 | - |
| Race_ethnicity | `real_religion5` | disambig | 263 / 3,440 (7.65%) | 144 / 119 | 0.73% | 150 / 113 | 1.45 [-0.41, 3.31] |
| Religion | `real_gender20` | ambig | 15 / 600 (2.50%) | 6 / 7 | -0.17% | 2 / 0 | - |
| Religion | `real_gender20` | disambig | 11 / 600 (1.83%) | 5 / 6 | -0.17% | 8 / 3 | -0.33 [-2.34, 1.67] |
| Religion | `real_gender5` | ambig | 7 / 600 (1.17%) | 4 / 3 | 0.17% | 0 / 0 | - |
| Religion | `real_gender5` | disambig | 6 / 600 (1.00%) | 2 / 4 | -0.33% | 3 / 3 | -0.67 [-2.34, 0.67] |
| Religion | `real_race20` | ambig | 54 / 600 (9.00%) | 24 / 20 | 0.67% | 10 / 0 | - |
| Religion | `real_race20` | disambig | 35 / 600 (5.83%) | 13 / 21 | -1.33% | 21 / 14 | -2.84 [-6.67, 1.00] |
| Religion | `real_race5` | ambig | 40 / 600 (6.67%) | 17 / 23 | -1.00% | 0 / 0 | - |
| Religion | `real_race5` | disambig | 33 / 600 (5.50%) | 14 / 19 | -0.83% | 21 / 12 | -1.67 [-5.33, 2.33] |
| Religion | `real_religion20` | ambig | 76 / 600 (12.67%) | 28 / 38 | -1.67% | 10 / 0 | - |
| Religion | `real_religion20` | disambig | 57 / 600 (9.50%) | 20 / 36 | -2.67% | 35 / 21 | -5.17 [-9.67, -0.15] |
| Religion | `real_religion5` | ambig | 41 / 600 (6.83%) | 19 / 12 | 1.17% | 10 / 0 | - |
| Religion | `real_religion5` | disambig | 23 / 600 (3.83%) | 9 / 13 | -0.67% | 13 / 10 | -1.51 [-4.51, 1.49] |

*Legend — **A→S / S→A**: flips from anti-stereotype to stereotyped person and vice versa; **Net→Stereo**: `(A→S − S→A) / n_items` (`%`); **Corr→Inc / Inc→Corr**: accuracy transitions; **Bias Δ (95% CI)**: change in benchmark bias score vs. baseline with paired item-bootstrap CI.*

## 6. Baseline Log-Likelihood Margin Mechanism

**Why this check:** verifies whether changed answers come selectively from baseline-undecided items (low `top − second` margin) rather than confident predictions, comparing median margins and flip rates across deciles 1–10.

| Set | Kind | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) |
| :--- | :---: | ---: | ---: | ---: |
| `real_gender20` | real | 2,912 | 0.125 | 1.047 |
| `real_race20` | real | 5,857 | 0.246 | 1.125 |
| `real_religion20` | real | 6,261 | 0.250 | 1.125 |
| `real_gender5` | real | 887 | 0.062 | 1.000 |
| `real_race5` | real | 2,417 | 0.117 | 1.031 |
| `real_religion5` | real | 5,075 | 0.188 | 1.109 |
| `rand_gender20 (mean n=5)` | random_family_mean | 994.6 | 0.061 | 1.000 |
| `rand_religion20 (mean n=10)` | random_family_mean | 929.1 | 0.061 | 1.000 |

| Decile | Margin Range | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` | `rand_gender20 (mean n=5)` | `rand_religion20 (mean n=10)` |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [0.000, 0.172] | 32.56% | 14.34% | 40.41% | 29.20% | 41.49% | 40.05% | 15.91% | 15.03% |
| 2 | [0.172, 0.367] | 12.34% | 0.74% | 26.64% | 8.55% | 28.26% | 24.48% | 0.94% | 0.79% |
| 3 | [0.367, 0.531] | 3.80% | 0.03% | 15.44% | 2.19% | 16.81% | 12.16% | 0.07% | 0.05% |
| 4 | [0.531, 0.750] | 0.89% | 0.00% | 9.45% | 1.16% | 10.26% | 5.80% | 0.04% | 0.01% |
| 5 | [0.750, 0.992] | 0.15% | 0.00% | 5.21% | 0.22% | 5.57% | 3.06% | 0.03% | 0.00% |
| 6 | [0.992, 1.262] | 0.03% | 0.05% | 2.09% | 0.00% | 3.27% | 1.09% | 0.00% | 0.00% |
| 7 | [1.262, 1.652] | 0.00% | 0.00% | 0.68% | 0.00% | 1.09% | 0.12% | 0.00% | 0.00% |
| 8 | [1.652, 2.266] | 0.00% | 0.00% | 0.19% | 0.00% | 0.27% | 0.00% | 0.00% | 0.00% |
| 9 | [2.266, 3.499] | 0.00% | 0.00% | 0.02% | 0.00% | 0.02% | 0.00% | 0.00% | 0.00% |
| 10 | [3.500, 14.250] | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) of changed vs. unchanged items; **Decile**: baseline-margin decile (`1` = smallest margin / most undecided, `10` = largest margin).*

## 7. Granular Breakdowns (Demographic Groups and Topics)

**Why this table:** checks whether answer changes concentrate in specific demographic subgroups (`answer_info`) or social topics (`relevant_social_value`, EsBBQ) within each target category.

<details>
<summary>Expand demographic group breakdown (target categories)</summary>

### Gender_identity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | 4,616 | 9.40% | 1.32% | 8.19% | 2.99% | 8.73% | 9.73% |
| M | 4,616 | 9.40% | 1.32% | 8.19% | 2.99% | 8.73% | 9.73% |
| boy | 40 | 5.00% | 0.00% | 2.50% | 2.50% | 5.00% | 5.00% |
| girl | 40 | 7.50% | 0.00% | 5.00% | 2.50% | 10.00% | 7.50% |
| man | 152 | 3.95% | 0.66% | 6.58% | 1.97% | 8.55% | 8.55% |
| nonTrans | 464 | 4.53% | 1.72% | 4.09% | 3.66% | 15.52% | 7.11% |
| nonTrans_F | 200 | 10.00% | 1.00% | 7.50% | 2.50% | 9.00% | 9.50% |
| nonTrans_M | 200 | 10.50% | 1.50% | 8.50% | 2.50% | 15.00% | 14.50% |
| trans | 464 | 4.53% | 1.72% | 4.09% | 3.66% | 15.52% | 7.11% |
| trans_F | 200 | 10.50% | 1.50% | 8.50% | 2.50% | 15.00% | 14.50% |
| trans_M | 200 | 10.00% | 1.00% | 7.50% | 2.50% | 9.00% | 9.50% |
| woman | 152 | 3.29% | 0.66% | 5.92% | 1.97% | 7.24% | 7.89% |

### Race_ethnicity — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| African | 200 | 5.00% | 2.50% | 14.50% | 5.50% | 8.00% | 8.50% |
| African American | 520 | 4.04% | 0.96% | 13.08% | 4.04% | 9.42% | 6.73% |
| Arab | 208 | 0.96% | 0.96% | 9.13% | 6.73% | 9.13% | 4.81% |
| Asian | 312 | 4.49% | 1.60% | 19.55% | 24.68% | 8.01% | 3.53% |
| Black | 520 | 4.23% | 1.73% | 14.04% | 6.73% | 12.12% | 10.58% |
| Caucasian | 128 | 8.59% | 0.78% | 7.81% | 0.78% | 10.16% | 10.16% |
| European | 184 | 2.72% | 2.17% | 17.39% | 9.24% | 9.78% | 6.52% |
| F-Arab | 1,056 | 4.83% | 2.46% | 14.58% | 3.69% | 12.03% | 12.97% |
| F-Asian | 904 | 6.64% | 1.11% | 11.73% | 3.76% | 10.51% | 9.29% |
| F-Black | 1,904 | 5.25% | 1.26% | 12.87% | 4.67% | 9.45% | 10.03% |
| F-Latino | 1,112 | 4.32% | 1.62% | 12.86% | 3.06% | 8.00% | 9.08% |
| F-White | 784 | 5.48% | 1.28% | 9.44% | 4.59% | 7.78% | 7.53% |
| Hispanic | 328 | 7.93% | 2.74% | 9.76% | 8.23% | 16.77% | 8.23% |
| Jewish | 168 | 2.98% | 0.60% | 7.14% | 5.36% | 10.71% | 8.93% |
| Latino | 304 | 3.62% | 1.32% | 8.88% | 5.92% | 7.24% | 3.62% |
| M-Arab | 824 | 1.94% | 1.21% | 11.77% | 2.79% | 18.57% | 10.44% |
| M-Asian | 848 | 1.65% | 0.59% | 13.44% | 2.59% | 9.79% | 9.91% |
| M-Black | 1,032 | 1.36% | 0.58% | 11.14% | 3.49% | 10.08% | 9.11% |
| M-Latino | 936 | 2.78% | 1.39% | 10.47% | 3.21% | 10.79% | 7.80% |
| M-White | 600 | 3.33% | 1.00% | 10.33% | 3.50% | 9.17% | 7.17% |
| Middle Eastern | 224 | 4.46% | 2.68% | 9.82% | 6.70% | 11.61% | 10.71% |
| Native American | 240 | 3.75% | 1.25% | 17.50% | 9.58% | 13.33% | 14.58% |
| Roma | 168 | 3.57% | 1.79% | 8.33% | 5.36% | 10.12% | 9.52% |
| South American | 136 | 9.56% | 1.47% | 26.47% | 7.35% | 26.47% | 18.38% |
| White | 120 | 5.83% | 4.17% | 24.17% | 5.83% | 19.17% | 6.67% |

### Religion — Groups

| Group | Items | `real_gender20` | `real_gender5` | `real_race20` | `real_race5` | `real_religion20` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atheist | 288 | 2.43% | 0.69% | 5.56% | 4.51% | 6.94% | 6.94% |
| Buddhist | 112 | 0.00% | 0.89% | 8.93% | 11.61% | 8.04% | 2.68% |
| Catholic | 264 | 1.14% | 0.76% | 6.82% | 5.30% | 6.82% | 3.03% |
| Christian | 312 | 2.24% | 0.96% | 9.29% | 2.88% | 13.14% | 5.77% |
| Hindu | 264 | 1.89% | 2.65% | 9.47% | 13.64% | 13.26% | 6.82% |
| Jewish | 280 | 2.14% | 0.36% | 8.93% | 6.79% | 11.07% | 4.64% |
| Mormon | 264 | 1.52% | 0.76% | 3.41% | 4.17% | 10.23% | 4.55% |
| Muslim | 328 | 3.05% | 2.13% | 7.32% | 5.18% | 17.07% | 6.71% |
| Protestant | 144 | 3.47% | 0.69% | 5.56% | 1.39% | 8.33% | 5.56% |
| Sikh | 144 | 3.47% | 0.00% | 9.72% | 8.33% | 11.81% | 4.17% |

</details>
