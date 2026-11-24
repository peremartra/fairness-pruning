# E3 Unit Dossier — `llama-3.2-1b_en` (Llama-3.2-1B, EN, BBQ)

[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **BBQ** (58,492 items, 11 categories) · Source: `results/e3/llama-3.2-1b_en/per_set (summary-only)`.

## 1. Set Summary and Target-Category Localization

Changed answers (`pred != base_pred`, `changed / n_items`, `%`) for each real and ablation set in its target category versus the mean of the other categories, with layer distribution and GPU.

| Set | Kind | Target Cat. | K | Target Flips | 95% CI (%) | Other Mean | Ratio | Rank | All Benchmark Flips | GPU | Layer Profile |
| :--- | :---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | :--- | :--- |
| `real_gender20` | real | Gender_identity | 20 | 1,150 / 5,672 (20.28%) | - | 6.67% | 3.04× | 1/11 | 5,428 / 58,492 (9.28%) | A100-SXM4-40GB | L5:1, L10:2, L11:1, L12:2, L13:2, L14:4, L15:8 |
| `real_race20` | real | Race_ethnicity | 20 | 797 / 6,880 (11.58%) | - | 7.51% | 1.54× | 4/11 | 5,255 / 58,492 (8.98%) | A100-SXM4-40GB | L9:1, L10:1, L12:2, L13:2, L14:1, L15:13 |
| `real_religion20` | real | Religion | 20 | 270 / 1,200 (22.50%) | - | 8.32% | 2.70× | 1/11 | 5,556 / 58,492 (9.50%) | A100-SXM4-40GB | L11:1, L13:1, L15:18 |
| `real_gender5` | real | Gender_identity | 5 | 108 / 5,672 (1.90%) | - | 1.93% | 0.99× | 8/11 | 1,157 / 58,492 (1.98%) | A100-SXM4-40GB | L10:1, L14:3, L15:1 |
| `real_race5` | real | Race_ethnicity | 5 | 538 / 6,880 (7.82%) | - | 6.12% | 1.28× | 4/11 | 3,675 / 58,492 (6.28%) | A100-SXM4-40GB | L14:1, L15:4 |
| `real_religion5` | real | Religion | 5 | 141 / 1,200 (11.75%) | - | 4.43% | 2.65× | 1/11 | 2,934 / 58,492 (5.02%) | A100-SXM4-40GB | L11:1, L15:4 |

## 2. Changed Answers Across All Categories (`Set × Category`)

Full matrix of changed answers (`pred != base_pred`, `changed (flips_pct)`) across every benchmark category (all items pooled; random families summarized by mean).

| Category | Items | `real_gender20` | `real_race20` | `real_religion20` | `real_gender5` | `real_race5` | `real_religion5` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age | 3,680 | 302 (8.21%) | 191 (5.19%) | 194 (5.27%) | 62 (1.68%) | 173 (4.70%) | 124 (3.37%) |
| Disability_status | 1,556 | 48 (3.08%) | 45 (2.89%) | 49 (3.15%) | 8 (0.51%) | 28 (1.80%) | 29 (1.86%) |
| Gender_identity | 5,672 | 1,150 (20.28%) | 431 (7.60%) | 587 (10.35%) | 108 (1.90%) | 193 (3.40%) | 292 (5.15%) |
| Nationality | 3,080 | 221 (7.18%) | 400 (12.99%) | 595 (19.32%) | 68 (2.21%) | 452 (14.68%) | 194 (6.30%) |
| Physical_appearance | 1,576 | 91 (5.77%) | 82 (5.20%) | 75 (4.76%) | 41 (2.60%) | 55 (3.49%) | 73 (4.63%) |
| Race_ethnicity | 6,880 | 597 (8.68%) | 797 (11.58%) | 812 (11.80%) | 146 (2.12%) | 538 (7.82%) | 448 (6.51%) |
| Race_x_SES | 11,160 | 661 (5.92%) | 808 (7.24%) | 906 (8.12%) | 224 (2.01%) | 524 (4.70%) | 455 (4.08%) |
| Race_x_gender | 15,960 | 1,853 (11.61%) | 1,944 (12.18%) | 1,752 (10.98%) | 375 (2.35%) | 1,258 (7.88%) | 952 (5.96%) |
| Religion | 1,200 | 68 (5.67%) | 152 (12.67%) | 270 (22.50%) | 24 (2.00%) | 148 (12.33%) | 141 (11.75%) |
| SES | 6,864 | 395 (5.75%) | 373 (5.43%) | 268 (3.90%) | 78 (1.14%) | 269 (3.92%) | 195 (2.84%) |
| Sexual_orientation | 864 | 42 (4.86%) | 32 (3.70%) | 48 (5.56%) | 23 (2.66%) | 37 (4.28%) | 31 (3.59%) |

## 3. Neuron and Changed-Item Overlap Between Sets

Pairwise count of shared zeroed neurons, neuron Jaccard, and changed-item Jaccard (all items and within each set's target category).

| Pair | Shared Neurons | Neuron Jaccard | Items Jacc (All) | Items Jacc (Target A) | Items Jacc (Target B) | Shared Neuron IDs |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `real_gender20` vs. `real_gender5` | 5 / 20,5 | 0.250 | - | Gender_identity: - | Gender_identity: - | L10.5982, L14.1583, L14.1693, L14.5426, L15.2314 |
| `real_race20` vs. `real_religion20` | 5 / 20,20 | 0.143 | - | Race_ethnicity: - | Religion: - | L15.840, L15.3343, L15.4758, L15.5936, L15.7677 |
| `real_race20` vs. `real_race5` | 5 / 20,5 | 0.250 | - | Race_ethnicity: - | Race_ethnicity: - | L14.5144, L15.3343, L15.3718, L15.4758, L15.6853 |
| `real_religion20` vs. `real_religion5` | 5 / 20,5 | 0.250 | - | Religion: - | Religion: - | L11.7762, L15.166, L15.1070, L15.1788, L15.5936 |
| `real_gender20` vs. `real_religion20` | 2 / 20,20 | 0.053 | - | Gender_identity: - | Religion: - | L15.1788, L15.1949 |
| `real_religion20` vs. `real_race5` | 2 / 20,5 | 0.087 | - | Religion: - | Race_ethnicity: - | L15.3343, L15.4758 |
| `real_gender20` vs. `real_religion5` | 1 / 20,5 | 0.042 | - | Gender_identity: - | Religion: - | L15.1788 |
| `real_race20` vs. `real_religion5` | 1 / 20,5 | 0.042 | - | Race_ethnicity: - | Religion: - | L15.5936 |
| `real_gender20` vs. `real_race20` | 0 / 20,20 | 0.000 | - | Gender_identity: - | Race_ethnicity: - | - |
| `real_gender20` vs. `real_race5` | 0 / 20,5 | 0.000 | - | Gender_identity: - | Race_ethnicity: - | - |
| `real_race20` vs. `real_gender5` | 0 / 20,5 | 0.000 | - | Race_ethnicity: - | Gender_identity: - | - |
| `real_religion20` vs. `real_gender5` | 0 / 20,5 | 0.000 | - | Religion: - | Gender_identity: - | - |
| `real_gender5` vs. `real_race5` | 0 / 5,5 | 0.000 | - | Gender_identity: - | Race_ethnicity: - | - |
| `real_gender5` vs. `real_religion5` | 0 / 5,5 | 0.000 | - | Gender_identity: - | Religion: - | - |
| `real_race5` vs. `real_religion5` | 0 / 5,5 | 0.000 | - | Race_ethnicity: - | Religion: - | - |

## 4. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)

Secondary benchmark metrics for target categories (`disambig` and `ambig` contexts, state-based S/A/U transitions); descriptive and not primary localization measures.

| Category | Set | Context | Flips | A→S / S→A | Net→Stereo | Corr→Inc / Inc→Corr | Disamb Bias Δ (95% CI) |
| :--- | :--- | :---: | ---: | :---: | ---: | :---: | ---: |
| Gender_identity | `real_gender20` | disambig | - | - / - | - | - / - | -5.85 |
| Gender_identity | `real_gender5` | disambig | - | - / - | - | - / - | 0.35 |
| Gender_identity | `real_race20` | disambig | - | - / - | - | - / - | -0.78 |
| Gender_identity | `real_race5` | disambig | - | - / - | - | - / - | 0.28 |
| Gender_identity | `real_religion20` | disambig | - | - / - | - | - / - | 0.71 |
| Gender_identity | `real_religion5` | disambig | - | - / - | - | - / - | 0.85 |
| Race_ethnicity | `real_gender20` | disambig | - | - / - | - | - / - | -0.41 |
| Race_ethnicity | `real_gender5` | disambig | - | - / - | - | - / - | -0.26 |
| Race_ethnicity | `real_race20` | disambig | - | - / - | - | - / - | -0.55 |
| Race_ethnicity | `real_race5` | disambig | - | - / - | - | - / - | -0.29 |
| Race_ethnicity | `real_religion20` | disambig | - | - / - | - | - / - | -0.58 |
| Race_ethnicity | `real_religion5` | disambig | - | - / - | - | - / - | -1.25 |
| Religion | `real_gender20` | disambig | - | - / - | - | - / - | -0.01 |
| Religion | `real_gender5` | disambig | - | - / - | - | - / - | 0.51 |
| Religion | `real_race20` | disambig | - | - / - | - | - / - | 0.67 |
| Religion | `real_race5` | disambig | - | - / - | - | - / - | 4.46 |
| Religion | `real_religion20` | disambig | - | - / - | - | - / - | 2.00 |
| Religion | `real_religion5` | disambig | - | - / - | - | - / - | 1.86 |
