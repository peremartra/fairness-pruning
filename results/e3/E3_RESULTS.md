# E3 Results — Neuron Zeroing Interventions Across Models and Languages

Consolidated from `89` verified Drive files (`inputs_manifest.json`, generated `2026-11-24T18:05:08+00:00`); primary measure is changed answers (`changed / n_items`, `flips_pct`) against the unpruned baseline, while BBQ/EsBBQ bias scores and accuracy transitions are secondary.

## Navigation

- **Level 1 — Cross-Unit Overview:** [1.1 Inventory](#11-experiment-inventory) · [1.2 Real vs. Random Controls](#12-primary-result-selected-vs-layer-matched-random-neurons) · [1.3 Localization (All Items)](#13-target-category-concentration-all-items-pooled) · [1.4 Localization (Disambiguated Only)](#14-target-category-concentration-disambiguated-items-only)
- **Level 2 — Unit Dossiers:** [2. Index of Units](#2-unit-dossiers-level-2)
- **Level 3 — Diagnostics & Controls:** [3.1 Single-Neuron Ablation (`L31·6950`)](#31-single-neuron-ablation-llama-31-8b-en-l316950) · [3.2 Baseline Margin Mechanism](#32-baseline-indecision-mechanism-margin-deciles) · [3.3 Hardware Noise Controls](#33-hardware-noise-controls-l4-vs-a100) · [3.4 Provenance & Tables](#34-provenance-and-source-tables)

---

## 1. Cross-Unit Overview (Level 1)

### 1.1 Experiment Inventory

**Why this table:** summarizes the full experimental scope across all eight E3 units (models, languages, benchmarks, set counts, tested K values, and GPU hardware) so every downstream table maps to its exact evaluation setup.

| Unit | Model | Lang | Bench | Items | Real | Abl. | Random | K | Per-item | Intervened GPUs |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: | :---: | :---: | :--- |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | Llama-3.1-8B | EN | BBQ | 58,492 (11 cats) | 6 | 3 | 15 | 5, 20 | Yes | A100-SXM4-40GB; A100-SXM4-80GB |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | Salamandra-7B | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 10 | 5, 20 | Yes | A100-SXM4-40GB |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | Qwen3-4B-Base | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 10 | 5, 20 | Yes | A100-SXM4-40GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Llama-3.2-3B | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 15 (+3 L4) | 5, 20 | Yes | A100-SXM4-40GB; A100-SXM4-80GB |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | Llama-3.2-1B | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 0 | 5, 20 | Summary only | A100-SXM4-40GB |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | Salamandra-7B | ES | EsBBQ | 27,320 (10 cats) | 9 | 0 | 5 | 1, 5, 10, 20, 40 | Yes | A100-SXM4-80GB |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | Llama-3.1-8B | ES | EsBBQ | 27,320 (10 cats) | 6 | 0 | 0 | 5, 20 | Yes | A100-SXM4-40GB; A100-SXM4-80GB |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | Llama-3.2-3B | ES | EsBBQ | 27,320 (10 cats) | 6 | 0 | 0 | 5, 20 | Yes | L4 |

*Legend — **Real**: selected top-K neuron sets; **Abl.**: exploratory ablation subsets derived from real sets (see 3.1); **Random**: layer-matched uniform random control sets (`+N L4` = extra L4 hardware-sensitivity reruns); **K**: number of zeroed neurons per set; **Per-item**: whether full item-level logs and bootstrap analysis are available (`Summary only` = aggregated per-category summary JSONs only).*

### 1.2 Primary Result: Selected vs. Layer-Matched Random Neurons

**Why this check:** tests the paper's core localization claim—whether zeroing BiasScore-selected neurons changes substantially more target-category answers than zeroing the same number of random neurons drawn uniformly from the exact same MLP layers (primary comparisons first; sensitivity rows marked).

| Unit | Comparison | Target Cat. | Selected Flips (Target) | Random Flips (Target, min–max) | Real/Rand | Rank | GPUs (Real / Rand) |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | Gender·20 (seeds 0-4) | Gender_identity | 1,361 / 5,672 (24.00%) | 41–76 (0.72–1.34% (mean 0.93%)) | 25.68× | 1 of 6 | A100-SXM4-80GB / A100-SXM4-40GB |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | Religion·20 (seeds 0-9) | Religion | 120 / 1,200 (10.00%) | 10–38 (0.83–3.17% (mean 1.37%)) | 7.32× | 1 of 11 | A100-SXM4-80GB / A100-SXM4-40GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Gender·20 (seeds 0-4, A100-only) | Gender_identity | 504 / 5,672 (8.89%) | 70–94 (1.23–1.66% (mean 1.46%)) | 6.10× | 1 of 6 | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Religion·20 (seeds 0-9, A100-only) | Religion | 133 / 1,200 (11.08%) | 11–16 (0.92–1.33% (mean 1.17%)) | 9.43× | 1 of 11 | A100-SXM4-80GB / A100-SXM4-40GB; A100-SXM4-80GB |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | Gender·20 (seeds 0-4) | Gender_identity | 397 / 5,672 (7.00%) | 28–64 (0.49–1.13% (mean 0.76%)) | 9.19× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | Religion·20 (seeds 0-4) | Religion | 79 / 1,200 (6.58%) | 8–16 (0.67–1.33% (mean 0.93%)) | 7.05× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | Gender·20 (seeds 0-4) | Gender_identity | 928 / 5,672 (16.36%) | 41–81 (0.72–1.43% (mean 0.93%)) | 17.58× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | Religion·20 (seeds 0-4) | Religion | 163 / 1,200 (13.58%) | 11–19 (0.92–1.58% (mean 1.23%)) | 11.01× | 1 of 6 | A100-SXM4-40GB / A100-SXM4-40GB |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | Gender·5 (seeds 0-4) | Gender | 983 / 4,832 (20.34%) | 12–86 (0.25–1.78% (mean 0.62%)) | 32.77× | 1 of 6 | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Gender·20 (L4 seed0 + A100 seeds 1-4, sensitivity) *(sens.)* | Gender_identity | 504 / 5,672 (8.89%) | 77–94 (1.36–1.66% (mean 1.53%)) | 5.81× | 1 of 6 | A100-SXM4-80GB / A100-SXM4-80GB; L4 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Religion·20 (L4 seeds 0-1 + A100 seeds 2-9, sensitivity) *(sens.)* | Religion | 133 / 1,200 (11.08%) | 13–20 (1.08–1.67% (mean 1.29%)) | 8.58× | 1 of 11 | A100-SXM4-80GB / A100-SXM4-40GB; L4 |

*Legend — **Selected Flips (Target)**: target-category items whose 3-way prediction (`person0`/`person1`/`unknown`) differs from baseline (`changed / n_items`, `%`); **Random Flips (Target, min–max)**: range across the layer-matched random seeds reported as `min–max changed items (min–max% (mean %))` over the same `n_items` target denominator (e.g., `41–76` = lowest and highest seed counts, `0.72–1.34%` = corresponding percentages, `mean 0.93%` = average across seeds); **Real/Rand**: `Selected % / Random mean %`; **Rank**: ordinal position of the selected set among selected + random (`1` = highest); `*(sens.)*`: sensitivity comparison.*

### 1.3 Target-Category Concentration (All Items Pooled)

**Why this check:** tests whether the answer changes caused by selected neurons concentrate in their target demographic category rather than moving all benchmark categories equally, comparing `Target Flips` against `Other Mean` across every real set (`K = 5`, `K = 20`, and EsBBQ grid sets).

| Unit | Set | Target Cat. | K | Last Layer | Target Flips | 95% CI (%) | Other Mean | Ratio | All Benchmark Flips |
| :--- | :--- | :--- | ---: | :---: | ---: | :---: | ---: | ---: | ---: |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender20` | Gender_identity | 20 | 15/20 | 1,361 / 5,672 (24.00%) | [22.88, 25.09] | 6.28% | 3.82× | 5,291 / 58,492 (9.05%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race20` | Race_ethnicity | 20 | 20/20 | 519 / 6,880 (7.54%) | [6.96, 8.17] | 5.43% | 1.39× | 3,399 / 58,492 (5.81%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion20` | Religion | 20 | 19/20 | 120 / 1,200 (10.00%) | [8.25, 11.67] | 6.50% | 1.54× | 4,027 / 58,492 (6.88%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender5` | Gender_identity | 5 | 4/5 | 1,323 / 5,672 (23.33%) | [22.23, 24.40] | 4.39% | 5.32× | 4,598 / 58,492 (7.86%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 415 / 6,880 (6.03%) | [5.49, 6.60] | 3.78% | 1.59× | 2,375 / 58,492 (4.06%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion5` | Religion | 5 | 5/5 | 118 / 1,200 (9.83%) | [8.17, 11.42] | 3.55% | 2.77× | 2,487 / 58,492 (4.25%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender20` | Gender_identity | 20 | 8/20 | 928 / 5,672 (16.36%) | [15.39, 17.28] | 4.31% | 3.79× | 3,860 / 58,492 (6.60%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race20` | Race_ethnicity | 20 | 18/20 | 519 / 6,880 (7.54%) | [6.95, 8.18] | 4.14% | 1.82× | 3,131 / 58,492 (5.35%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion20` | Religion | 20 | 17/20 | 163 / 1,200 (13.58%) | [11.75, 15.59] | 5.49% | 2.48× | 3,866 / 58,492 (6.61%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender5` | Gender_identity | 5 | 2/5 | 913 / 5,672 (16.10%) | [15.14, 17.00] | 2.95% | 5.45× | 2,953 / 58,492 (5.05%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 351 / 6,880 (5.10%) | [4.61, 5.58] | 2.84% | 1.79× | 2,255 / 58,492 (3.86%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion5` | Religion | 5 | 4/5 | 113 / 1,200 (9.42%) | [7.75, 11.08] | 3.41% | 2.76× | 2,471 / 58,492 (4.22%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender20` | Gender_identity | 20 | 18/20 | 397 / 5,672 (7.00%) | [6.38, 7.63] | 7.48% | 0.94× | 4,353 / 58,492 (7.44%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race20` | Race_ethnicity | 20 | 19/20 | 371 / 6,880 (5.39%) | [4.90, 5.94] | 6.33% | 0.85× | 3,695 / 58,492 (6.32%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion20` | Religion | 20 | 20/20 | 79 / 1,200 (6.58%) | [5.17, 8.00] | 7.98% | 0.83× | 4,746 / 58,492 (8.11%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender5` | Gender_identity | 5 | 5/5 | 245 / 5,672 (4.32%) | [3.83, 4.85] | 5.22% | 0.83× | 3,132 / 58,492 (5.35%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 199 / 6,880 (2.89%) | [2.51, 3.31] | 3.03% | 0.96× | 2,123 / 58,492 (3.63%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion5` | Religion | 5 | 5/5 | 49 / 1,200 (4.08%) | [3.08, 5.25] | 3.89% | 1.05× | 2,560 / 58,492 (4.38%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender20` | Gender_identity | 20 | 6/20 | 504 / 5,672 (8.89%) | [8.16, 9.61] | 4.17% | 2.13× | 2,912 / 58,492 (4.98%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race20` | Race_ethnicity | 20 | 18/20 | 857 / 6,880 (12.46%) | [11.72, 13.24] | 8.03% | 1.55× | 5,857 / 58,492 (10.01%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion20` | Religion | 20 | 14/20 | 133 / 1,200 (11.08%) | [9.50, 12.84] | 9.61% | 1.15× | 6,261 / 58,492 (10.70%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender5` | Gender_identity | 5 | 0/5 | 75 / 5,672 (1.32%) | [1.02, 1.60] | 1.46% | 0.90× | 887 / 58,492 (1.52%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 329 / 6,880 (4.78%) | [4.26, 5.29] | 4.26% | 1.12× | 2,417 / 58,492 (4.13%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion5` | Religion | 5 | 4/5 | 64 / 1,200 (5.33%) | [4.17, 6.58] | 7.81% | 0.68× | 5,075 / 58,492 (8.68%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_gender20` | Gender_identity | 20 | 8/20 | 1,150 / 5,672 (20.28%) | - | 6.67% | 3.04× | 5,428 / 58,492 (9.28%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_race20` | Race_ethnicity | 20 | 13/20 | 797 / 6,880 (11.58%) | - | 7.51% | 1.54× | 5,255 / 58,492 (8.98%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_religion20` | Religion | 20 | 18/20 | 270 / 1,200 (22.50%) | - | 8.32% | 2.70× | 5,556 / 58,492 (9.50%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_gender5` | Gender_identity | 5 | 1/5 | 108 / 5,672 (1.90%) | - | 1.93% | 0.99× | 1,157 / 58,492 (1.98%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_race5` | Race_ethnicity | 5 | 4/5 | 538 / 6,880 (7.82%) | - | 6.12% | 1.28× | 3,675 / 58,492 (6.28%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_religion5` | Religion | 5 | 4/5 | 141 / 1,200 (11.75%) | - | 4.43% | 2.65× | 2,934 / 58,492 (5.02%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp5` | Religion | 40 | 34/40 | 78 / 648 (12.04%) | [9.41, 14.51] | 7.75% | 1.55× | 1,857 / 27,320 (6.80%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp9` | Age | 20 | 20/20 | 214 / 4,068 (5.26%) | [4.67, 5.92] | 6.38% | 0.82× | 1,527 / 27,320 (5.59%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp11` | Gender | 20 | 17/20 | 947 / 4,832 (19.60%) | [18.56, 20.70] | 5.79% | 3.38× | 2,334 / 27,320 (8.54%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp8` | Age | 10 | 10/10 | 212 / 4,068 (5.21%) | [4.57, 5.88] | 4.47% | 1.17× | 1,071 / 27,320 (3.92%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp7` | Age | 5 | 5/5 | 207 / 4,068 (5.09%) | [4.40, 5.78] | 3.27% | 1.56× | 862 / 27,320 (3.16%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp10` | Gender | 5 | 5/5 | 983 / 4,832 (20.34%) | [19.31, 21.40] | 4.21% | 4.83× | 2,048 / 27,320 (7.50%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp12` | RaceEthnicity | 5 | 5/5 | 69 / 3,716 (1.86%) | [1.45, 2.29] | 1.89% | 0.98× | 484 / 27,320 (1.77%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp2` | Religion | 5 | 4/5 | 22 / 648 (3.40%) | [2.16, 4.94] | 2.74% | 1.24× | 775 / 27,320 (2.84%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp1` | Religion | 1 | 1/1 | 9 / 648 (1.39%) | [0.62, 2.47] | 1.72% | 0.81× | 460 / 27,320 (1.68%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp11` | Gender | 20 | 16/20 | 626 / 4,832 (12.96%) | [11.92, 13.91] | 5.31% | 2.44× | 1,853 / 27,320 (6.78%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp14` | RaceEthnicity | 20 | 20/20 | 335 / 3,716 (9.02%) | [8.02, 9.99] | 8.65% | 1.04× | 2,418 / 27,320 (8.85%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp13` | Religion | 20 | 17/20 | 43 / 648 (6.64%) | [4.63, 8.80] | 5.03% | 1.32× | 1,354 / 27,320 (4.96%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp10` | Gender | 5 | 4/5 | 113 / 4,832 (2.34%) | [1.95, 2.79] | 3.29% | 0.71× | 881 / 27,320 (3.22%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp12` | RaceEthnicity | 5 | 5/5 | 114 / 3,716 (3.07%) | [2.53, 3.63] | 2.84% | 1.08× | 812 / 27,320 (2.97%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp2` | Religion | 5 | 4/5 | 24 / 648 (3.70%) | [2.16, 5.25] | 2.42% | 1.53× | 644 / 27,320 (2.36%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp11` | Gender | 20 | 10/20 | 816 / 4,832 (16.89%) | [15.85, 17.98] | 6.73% | 2.51× | 2,337 / 27,320 (8.55%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp14` | RaceEthnicity | 20 | 12/20 | 529 / 3,716 (14.24%) | [13.13, 15.31] | 12.19% | 1.17× | 3,307 / 27,320 (12.10%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp13` | Religion | 20 | 13/20 | 101 / 648 (15.59%) | [12.50, 18.36] | 10.96% | 1.42× | 3,055 / 27,320 (11.18%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp10` | Gender | 5 | 2/5 | 74 / 4,832 (1.53%) | [1.20, 1.88] | 2.79% | 0.55× | 684 / 27,320 (2.50%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp12` | RaceEthnicity | 5 | 5/5 | 298 / 3,716 (8.02%) | [7.13, 8.83] | 8.76% | 0.92× | 2,376 / 27,320 (8.70%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp2` | Religion | 5 | 3/5 | 59 / 648 (9.10%) | [6.79, 11.27] | 3.85% | 2.36× | 1,070 / 27,320 (3.92%) |

*Legend — **Last Layer**: zeroed neurons located in the final MLP layer out of `K`; **Target Flips**: changed answers (`pred != base_pred`) in the target category; **95% CI (%)**: paired item-bootstrap percentile interval (`B = 1,000`); **Other Mean**: unweighted mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **All Benchmark Flips**: total changed answers across the full benchmark.*

### 1.4 Target-Category Concentration (Disambiguated Items Only)

**Why this check:** verifies that target-category concentration still holds when restricted to disambiguated items alone (`context = disambig`, where the prompt supplies a clear factual answer), confirming the effect is not confined to ambiguous items.

<details>
<summary>Expand disambiguated-only localization table (all real sets with per-item logs)</summary>

| Unit | Set | Target Cat. | K | Disambig Target (%) | Disambig Other Mean (%) | Ratio |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender20` | Gender_identity | 20 | 14.67% | 3.75% | 3.91× |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race20` | Race_ethnicity | 20 | 4.68% | 3.03% | 1.55× |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion20` | Religion | 20 | 7.00% | 3.58% | 1.96× |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender5` | Gender_identity | 5 | 14.21% | 2.84% | 5.00× |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race5` | Race_ethnicity | 5 | 3.69% | 2.04% | 1.81× |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion5` | Religion | 5 | 7.00% | 1.93% | 3.63× |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender20` | Gender_identity | 20 | 8.43% | 2.42% | 3.49× |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race20` | Race_ethnicity | 20 | 3.75% | 2.14% | 1.75× |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion20` | Religion | 20 | 9.50% | 2.83% | 3.36× |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender5` | Gender_identity | 5 | 8.04% | 1.58% | 5.10× |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race5` | Race_ethnicity | 5 | 2.70% | 1.39% | 1.94× |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion5` | Religion | 5 | 6.33% | 1.61% | 3.94× |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender20` | Gender_identity | 20 | 3.28% | 3.20% | 1.03× |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race20` | Race_ethnicity | 20 | 1.57% | 2.55% | 0.62× |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion20` | Religion | 20 | 2.33% | 3.37% | 0.69× |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender5` | Gender_identity | 5 | 1.38% | 2.19% | 0.63× |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race5` | Race_ethnicity | 5 | 0.67% | 1.46% | 0.46× |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion5` | Religion | 5 | 1.33% | 1.58% | 0.85× |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender20` | Gender_identity | 20 | 6.63% | 3.31% | 2.00× |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race20` | Race_ethnicity | 20 | 10.00% | 6.14% | 1.63× |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion20` | Religion | 20 | 9.50% | 7.40% | 1.28× |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender5` | Gender_identity | 5 | 1.16% | 1.14% | 1.02× |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race5` | Race_ethnicity | 5 | 3.92% | 3.48% | 1.13× |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion5` | Religion | 5 | 3.83% | 5.99% | 0.64× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp5` | Religion | 40 | 11.57% | 6.78% | 1.71× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp9` | Age | 20 | 5.40% | 5.56% | 0.97× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp11` | Gender | 20 | 15.66% | 5.16% | 3.04× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp8` | Age | 10 | 5.19% | 3.98% | 1.30× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp7` | Age | 5 | 5.30% | 2.81% | 1.89× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp10` | Gender | 5 | 16.47% | 3.95% | 4.17× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp12` | RaceEthnicity | 5 | 1.45% | 1.71% | 0.85× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp2` | Religion | 5 | 3.01% | 2.20% | 1.37× |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp1` | Religion | 1 | 1.39% | 1.50% | 0.93× |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp11` | Gender | 20 | 10.52% | 4.08% | 2.58× |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp14` | RaceEthnicity | 20 | 8.08% | 7.05% | 1.15× |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp13` | Religion | 20 | 7.41% | 4.16% | 1.78× |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp10` | Gender | 5 | 1.83% | 2.73% | 0.67× |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp12` | RaceEthnicity | 5 | 2.57% | 2.42% | 1.06× |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp2` | Religion | 5 | 3.94% | 1.96% | 2.01× |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp11` | Gender | 20 | 14.96% | 5.77% | 2.59× |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp14` | RaceEthnicity | 20 | 12.30% | 11.76% | 1.05× |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp13` | Religion | 20 | 15.74% | 9.66% | 1.63× |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp10` | Gender | 5 | 1.11% | 2.19% | 0.51× |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp12` | RaceEthnicity | 5 | 6.87% | 8.49% | 0.81× |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp2` | Religion | 5 | 8.33% | 3.28% | 2.54× |

*Legend — **Disambig Target / Other Mean**: changed-answer `%` restricted to disambiguated items (`direction.csv`, S/A/U state changes); **Ratio**: `Disambig Target / Disambig Other Mean`.*

</details>

---

## 2. Unit Dossiers (Level 2)

**Why this table:** indexes the eight homogeneous per-unit dossiers where full `set × category` matrices, neuron/item overlaps, secondary bias/accuracy metrics, margin deciles, and demographic group/topic breakdowns are stored.

| Dossier | Model | Lang | Bench | Sets | Contents |
| :--- | :--- | :---: | :---: | :--- | :--- |
| [`units/llama-3.1-8b_en.md`](units/llama-3.1-8b_en.md) | Llama-3.1-8B | EN | BBQ | 6 real / 15 rand / 3 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups) |
| [`units/salamandra-7b_en.md`](units/salamandra-7b_en.md) | Salamandra-7B | EN | BBQ | 6 real / 10 rand / 0 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups) |
| [`units/qwen3-4b-base_en.md`](units/qwen3-4b-base_en.md) | Qwen3-4B-Base | EN | BBQ | 6 real / 10 rand / 0 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups) |
| [`units/llama-3.2-3b_en.md`](units/llama-3.2-3b_en.md) | Llama-3.2-3B | EN | BBQ | 6 real / 15 rand / 0 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups) |
| [`units/llama-3.2-1b_en.md`](units/llama-3.2-1b_en.md) | Llama-3.2-1B | EN | BBQ | 6 real / 0 rand / 0 abl | Summary-only (`per_set/*_summary.json`: flips by category, neuron overlap, disambig bias delta) |
| [`units/salamandra-7b_es.md`](units/salamandra-7b_es.md) | Salamandra-7B | ES | EsBBQ | 9 real / 5 rand / 0 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups, topics) |
| [`units/llama-3.1-8b_es.md`](units/llama-3.1-8b_es.md) | Llama-3.1-8B | ES | EsBBQ | 6 real / 0 rand / 0 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups, topics) |
| [`units/llama-3.2-3b_es.md`](units/llama-3.2-3b_es.md) | Llama-3.2-3B | ES | EsBBQ | 6 real / 0 rand / 0 abl | Full per-item analysis (flips, overlap, secondary metrics, margins, groups, topics) |

*Legend — **Sets**: number of selected (`real`), layer-matched random (`rand`), and exploratory ablation (`abl`) sets in each unit dossier. The `3 abl` sets in `llama-3.1-8b_en` decompose `real_race5` and `real_religion5` around their single shared neuron `L31·6950`: `abl_shared1` (`L31·6950` alone, 1 neuron), `abl_race4` (`real_race5` without `L31·6950`, 4 neurons), and `abl_religion4` (`real_religion5` without `L31·6950`, 4 neurons); see [Section 3.1](#31-single-neuron-ablation-llama-31-8b-en-l316950).*

---

## 3. Diagnostics and Controls (Level 3)

### 3.1 Single-Neuron Ablation (`Llama-3.1-8B EN`, `L31·6950`)

**Why this check:** tests whether the shared effect of `real_race5` and `real_religion5` in `Llama-3.1-8B EN` is spread across all five neurons or concentrated in their single shared neuron (`abl_shared1` = `L31·6950`), comparing changed answers and item-level overlap (`In` = share of row flips in the original Top-5; `Reprod.` = share of original Top-5 flips reproduced) across Religion, Race, and Gender.

| Set | Neurons (L31) | Religion (All) | Race (All) | Gender (All) | In Race5 (Rel/Race/Gen) | Reprod. Race5 (Rel/Race/Gen) | In Rel5 (Rel/Race/Gen) | Reprod. Rel5 (Rel/Race/Gen) |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: |
| `real_race5` | L31.1495, L31.2100, L31.6466, L31.6532, L31.6950 | 122 / 1,200 (10.17%) | 415 / 6,880 (6.03%) | 105 / 5,672 (1.85%) | 100.0% / 100.0% / 100.0% | 100.0% / 100.0% / 100.0% | 69.7% / 80.5% / 41.0% | 72.0% / 78.4% / 41.0% |
| `real_religion5` | L31.33, L31.311, L31.2398, L31.2564, L31.6950 | 118 / 1,200 (9.83%) | 426 / 6,880 (6.19%) | 105 / 5,672 (1.85%) | 72.0% / 78.4% / 41.0% | 69.7% / 80.5% / 41.0% | 100.0% / 100.0% / 100.0% | 100.0% / 100.0% / 100.0% |
| `abl_shared1` | L31.6950 | 116 / 1,200 (9.67%) | 402 / 6,880 (5.84%) | 60 / 5,672 (1.06%) | 95.7% / 86.6% / 73.3% | 91.0% / 83.9% / 41.9% | 75.0% / 83.8% / 66.7% | 73.7% / 79.1% / 38.1% |
| `abl_race4` | L31.1495, L31.2100, L31.6466, L31.6532 | 19 / 1,200 (1.58%) | 155 / 6,880 (2.25%) | 89 / 5,672 (1.57%) | 63.2% / 65.2% / 68.5% | 9.8% / 24.3% / 58.1% | 57.9% / 50.3% / 30.3% | 9.3% / 18.3% / 25.7% |
| `abl_religion4` | L31.33, L31.311, L31.2398, L31.2564 | 74 / 1,200 (6.17%) | 175 / 6,880 (2.54%) | 81 / 5,672 (1.43%) | 23.0% / 38.9% / 33.3% | 13.9% / 16.4% / 25.7% | 54.1% / 61.1% / 74.1% | 33.9% / 25.1% / 57.1% |

*Legend — **(All)**: pooled changed answers (`changed / n_items`, `%`, `pred != base_pred`); **In Race5 / In Rel5 (Rel/Race/Gen)**: percentage of the row's changed answers that also changed in the original `real_race5` / `real_religion5` set (`shared / row_changed`, ordered `Religion / Race / Gender`); **Reprod. Race5 / Reprod. Rel5 (Rel/Race/Gen)**: percentage of the original `real_race5` / `real_religion5` changed answers reproduced by the row (`shared / original_changed`, ordered `Religion / Race / Gender`).*

**Why this check:** tests whether `L31·6950`'s disproportionate effect is merely due to abnormally large output weights (`down_proj` column norm) rather than its differential activation (`BiasScore`). Below are the `down_proj` L2 norms, BiasScores, and within-layer percentiles across all 14,336 neurons of Layer 31 for the nine neurons in `real_race5` and `real_religion5` (all nine have below-median weight norms, percentiles 0.34%–25.96%, ruling out a weight-magnitude artifact).

| Neuron | Ablation Subset | `down_proj` Norm | Norm Pct. | Religion Score (Pct.) | Race Score (Pct.) | Gender Score (Pct.) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `L31·6950` | `abl_shared1` | 0.5622 | 8.20% | 2.328 (99.99%) | 1.627 (99.99%) | 0.337 (99.71%) |
| `L31·1495` | `abl_race4` | 0.7147 | 25.96% | 0.021 (58.80%) | 1.486 (99.99%) | 0.013 (72.64%) |
| `L31·2100` | `abl_race4` | 0.5849 | 10.78% | 0.583 (99.30%) | 1.096 (99.97%) | 0.416 (99.85%) |
| `L31·6466` | `abl_race4` | 0.4213 | 0.70% | 0.477 (98.92%) | 1.172 (99.98%) | 0.199 (98.97%) |
| `L31·6532` | `abl_race4` | 0.5809 | 10.36% | 0.468 (98.88%) | 1.067 (99.97%) | 0.082 (97.31%) |
| `L31·33` | `abl_religion4` | 0.5161 | 4.02% | 1.990 (99.97%) | 0.136 (96.59%) | 0.452 (99.90%) |
| `L31·311` | `abl_religion4` | 0.4344 | 0.91% | 2.355 (99.99%) | 0.923 (99.90%) | 0.371 (99.78%) |
| `L31·2398` | `abl_religion4` | 0.3790 | 0.34% | 2.247 (99.98%) | 1.063 (99.95%) | 0.321 (99.65%) |
| `L31·2564` | `abl_religion4` | 0.6237 | 15.39% | 1.893 (99.97%) | 0.698 (99.79%) | 0.355 (99.75%) |

*Legend — **`down_proj` Norm**: L2 norm of the neuron's output column in Layer 31; **Norm Pct.**: percentile within Layer 31's 14,336 neurons; **Score (Pct.)**: differential-activation BiasScore and within-layer percentile for each category.*

### 3.2 Baseline Indecision Mechanism (Margin Deciles)

**Why this check:** tests whether neuron zeroing disrupts confident predictions at random or acts selectively at the decision boundary by flipping only items where the unpruned model was already undecided. Below are the median baseline log-likelihood margins (`top − second` across `person0`/`person1`/`unknown`) of changed versus unchanged items, and the flip rate in Decile 1 (most undecided 10%), Decile 5, and Decile 10 (most confident 10%, `0.00%` flips in 18 of 19 rows, max `0.07%`) for `K = 20` real sets (plus `Gender·5` in `Salamandra-7B ES`).

| Unit | Set | Changed Items (All) | Median Margin (Changed) | Median Margin (Unchanged) | Decile 1 Flips | Decile 5 Flips | Decile 10 Flips |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender20` | 5,291 | 0.250 | 1.250 | 37.28% | 5.80% | 0.00% |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race20` | 3,399 | 0.141 | 1.219 | 33.52% | 1.16% | 0.00% |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion20` | 4,027 | 0.172 | 1.234 | 36.39% | 1.86% | 0.00% |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender20` | 3,860 | 0.250 | 1.750 | 34.74% | 0.44% | 0.00% |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race20` | 3,131 | 0.188 | 1.719 | 32.91% | 0.26% | 0.00% |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion20` | 3,866 | 0.250 | 1.746 | 34.80% | 2.65% | 0.00% |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender20` | 4,353 | 0.344 | 2.742 | 41.01% | 0.48% | 0.00% |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race20` | 3,695 | 0.281 | 2.672 | 38.12% | 0.14% | 0.00% |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion20` | 4,746 | 0.375 | 2.750 | 41.93% | 0.82% | 0.00% |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender20` | 2,912 | 0.125 | 1.047 | 32.56% | 0.15% | 0.00% |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race20` | 5,857 | 0.246 | 1.125 | 40.41% | 5.21% | 0.00% |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion20` | 6,261 | 0.250 | 1.125 | 41.49% | 5.57% | 0.00% |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp5` | 1,857 | 0.156 | 1.188 | 38.03% | 2.60% | 0.00% |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp10` | 2,048 | 0.250 | 1.211 | 32.21% | 3.77% | 0.00% |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp11` | 2,334 | 0.250 | 1.234 | 37.15% | 3.44% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp10` | 881 | 0.094 | 1.000 | 24.93% | 0.29% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp13` | 1,354 | 0.125 | 1.004 | 32.65% | 0.51% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp11` | 1,853 | 0.125 | 1.031 | 38.10% | 1.06% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp14` | 2,418 | 0.172 | 1.062 | 40.15% | 2.01% | 0.00% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp10` | 684 | 0.094 | 1.055 | 19.88% | 0.11% | 0.00% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp13` | 3,055 | 0.250 | 1.156 | 42.64% | 6.84% | 0.00% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp11` | 2,337 | 0.219 | 1.125 | 39.02% | 1.90% | 0.07% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp14` | 3,307 | 0.312 | 1.156 | 40.81% | 10.43% | 0.00% |

*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) for changed vs. unchanged items; **Decile 1 / 5 / 10 Flips**: flip `%` in baseline-margin deciles (`1` = smallest margin / most undecided, `10` = largest margin / most confident).*

### 3.3 Hardware Noise Controls (L4 vs. A100)

**Why this check:** quantifies the pure cross-GPU hardware flip floor when running the exact same unpruned model (`Llama-3.1-8B ES`, K3 rerun across all 27,320 EsBBQ items) on A100 versus L4, showing why primary comparisons must stay on matching GPU hardware.

| EsBBQ Category | Ambiguous Flips (L4 vs. A100) | Disambiguated Flips (L4 vs. A100) | Total Flips (L4 vs. A100) |
| :--- | ---: | ---: | ---: |
| Age | 43 / 1,292 (3.33%) | 61 / 2,776 (2.20%) | 104 / 4,068 (2.56%) |
| DisabilityStatus | 22 / 928 (2.37%) | 30 / 1,904 (1.58%) | 52 / 2,832 (1.84%) |
| Gender | 24 / 1,504 (1.60%) | 51 / 3,328 (1.53%) | 75 / 4,832 (1.55%) |
| LGBTQIA | 15 / 576 (2.60%) | 38 / 1,424 (2.67%) | 53 / 2,000 (2.65%) |
| Nationality | 5 / 168 (2.98%) | 7 / 336 (2.08%) | 12 / 504 (2.38%) |
| PhysicalAppearance | 27 / 1,176 (2.30%) | 34 / 2,352 (1.45%) | 61 / 3,528 (1.73%) |
| RaceEthnicity | 27 / 1,228 (2.20%) | 27 / 2,488 (1.09%) | 54 / 3,716 (1.45%) |
| Religion | 3 / 216 (1.39%) | 10 / 432 (2.31%) | 13 / 648 (2.01%) |
| SES | 29 / 1,380 (2.10%) | 30 / 2,824 (1.06%) | 59 / 4,204 (1.40%) |
| SpanishRegion | 17 / 324 (5.25%) | 6 / 664 (0.90%) | 23 / 988 (2.33%) |
| **TOTAL** | 212 / 8,792 (2.41%) | 294 / 18,528 (1.59%) | 506 / 27,320 (1.85%) |

*Legend — **Flips (L4 vs. A100)**: unpruned baseline items whose 3-way prediction changes solely due to running on L4 versus A100 hardware (`changed / n_items`, `%`).*

**Why this check:** measures how much L4-vs-A100 hardware noise inflates a random control set's flip rate when evaluated on L4 against an A100 baseline (identical `rand_gender20_seed0` set in `Llama-3.2-3B EN` run on A100-SXM4-80GB vs. L4).

| BBQ Category | A100-80GB Flips | L4 Flips | L4 − A100 (pp) |
| :--- | ---: | ---: | ---: |
| Age | 71 / 3,680 (1.93%) | 77 / 3,680 (2.09%) | +0.16 pp |
| Disability_status | 5 / 1,556 (0.32%) | 9 / 1,556 (0.58%) | +0.26 pp |
| Gender_identity | 70 / 5,672 (1.23%) | 91 / 5,672 (1.60%) | +0.37 pp |
| Nationality | 46 / 3,080 (1.49%) | 61 / 3,080 (1.98%) | +0.49 pp |
| Physical_appearance | 25 / 1,576 (1.59%) | 35 / 1,576 (2.22%) | +0.63 pp |
| Race_ethnicity | 100 / 6,880 (1.45%) | 130 / 6,880 (1.89%) | +0.44 pp |
| Race_x_SES | 214 / 11,160 (1.92%) | 229 / 11,160 (2.05%) | +0.13 pp |
| Race_x_gender | 266 / 15,960 (1.67%) | 291 / 15,960 (1.82%) | +0.16 pp |
| Religion | 18 / 1,200 (1.50%) | 15 / 1,200 (1.25%) | -0.25 pp |
| SES | 98 / 6,864 (1.43%) | 119 / 6,864 (1.73%) | +0.31 pp |
| Sexual_orientation | 13 / 864 (1.50%) | 15 / 864 (1.74%) | +0.23 pp |

*Legend — **A100-80GB / L4 Flips**: changed answers against the A100-40GB baseline for the exact same random neuron set on each GPU; **L4 − A100 (pp)**: percentage-point hardware difference.*

### 3.4 Provenance and Source Tables

**Why this table:** lists every consolidated CSV table in `results/e3/tables/` from which this report is deterministically generated, each traced by SHA-256 to its source Drive file in `tables/inputs_manifest.json`.

| CSV Table | Rows | Columns | Size |
| :--- | ---: | ---: | ---: |
| [`tables/e3_ablation_l31_6950.csv`](tables/e3_ablation_l31_6950.csv) | 45 | 12 | 6,176 B |
| [`tables/e3_esbbq_topics.csv`](tables/e3_esbbq_topics.csv) | 1,596 | 8 | 131,821 B |
| [`tables/e3_flips_by_category.csv`](tables/e3_flips_by_category.csv) | 3,486 | 14 | 450,545 B |
| [`tables/e3_gpu_noise_3b.csv`](tables/e3_gpu_noise_3b.csv) | 11 | 10 | 1,860 B |
| [`tables/e3_gpu_noise_k3.csv`](tables/e3_gpu_noise_k3.csv) | 11 | 13 | 1,062 B |
| [`tables/e3_groups.csv`](tables/e3_groups.csv) | 1,644 | 8 | 127,866 B |
| [`tables/e3_inventory.csv`](tables/e3_inventory.csv) | 8 | 16 | 1,769 B |
| [`tables/e3_l31_weights.csv`](tables/e3_l31_weights.csv) | 9 | 23 | 2,961 B |
| [`tables/e3_localization_summary.csv`](tables/e3_localization_summary.csv) | 54 | 29 | 15,503 B |
| [`tables/e3_margin_deciles.csv`](tables/e3_margin_deciles.csv) | 570 | 8 | 41,357 B |
| [`tables/e3_margin_summary.csv`](tables/e3_margin_summary.csv) | 238 | 7 | 16,280 B |
| [`tables/e3_neuron_overlap.csv`](tables/e3_neuron_overlap.csv) | 162 | 16 | 27,091 B |
| [`tables/e3_provenance.csv`](tables/e3_provenance.csv) | 120 | 18 | 28,916 B |
| [`tables/e3_real_vs_random.csv`](tables/e3_real_vs_random.csv) | 11 | 29 | 4,555 B |
| [`tables/e3_secondary_metrics.csv`](tables/e3_secondary_metrics.csv) | 477 | 28 | 91,388 B |
