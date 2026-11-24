# E3 Results — Neuron Zeroing Interventions Across Models and Languages

Consolidated from `89` verified Drive files (`inputs_manifest.json`, generated `2026-11-24T10:16:45+00:00`); primary measure is changed answers (`changed / n_items`, `flips_pct`) against the unpruned baseline, while BBQ/EsBBQ bias scores and accuracy transitions are secondary.

## Navigation

- **Level 1 — Cross-Unit Overview:** [1.1 Inventory](#11-experiment-inventory) · [1.2 Real vs. Random Controls](#12-primary-result-selected-vs-layer-matched-random-neurons) · [1.3 Localization (All Items)](#13-target-category-concentration-all-items-pooled) · [1.4 Localization (Disambiguated Only)](#14-target-category-concentration-disambiguated-items-only)
- **Level 2 — Unit Dossiers:** [2. Index of Units](#2-unit-dossiers-level-2)
- **Level 3 — Diagnostics & Controls:** [3.1 Single-Neuron Ablation (`L31·6950`)](#31-single-neuron-ablation-llama-31-8b-en-l316950) · [3.2 Baseline Margin Mechanism](#32-baseline-indecision-mechanism-margin-deciles) · [3.3 Hardware Noise Controls](#33-hardware-noise-controls-l4-vs-a100) · [3.4 Provenance & Tables](#34-provenance-and-source-tables)

---

## 1. Cross-Unit Overview (Level 1)

### 1.1 Experiment Inventory

All eight E3 units covered in this report, with dataset size, number of intervened sets, tested K values, and GPU hardware.

| Unit | Model | Lang | Bench | Items | Real | Abl. | Random | K | Per-item | Intervened GPUs |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: | :---: | :---: | :--- |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | Llama-3.1-8B | EN | BBQ | 58,492 (11 cats) | 6 | 3 | 15 | 5, 20 | Yes | A100-SXM4-40GB; A100-SXM4-80GB |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | Salamandra-7B | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 10 | 5, 20 | Yes | A100-SXM4-40GB |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | Qwen3-4B-Base | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 10 | 5, 20 | Yes | A100-SXM4-40GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Llama-3.2-3B | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 15 (+1 L4) | 5, 20 | Yes | A100-SXM4-40GB; A100-SXM4-80GB; L4 |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | Llama-3.2-1B | EN | BBQ | 58,492 (11 cats) | 6 | 0 | 0 | 5, 20 | Summary only | A100-SXM4-40GB |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | Salamandra-7B | ES | EsBBQ | 27,320 (10 cats) | 9 | 0 | 5 | 1, 5, 10, 20, 40 | Yes | A100-SXM4-80GB |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | Llama-3.1-8B | ES | EsBBQ | 27,320 (10 cats) | 6 | 0 | 0 | 5, 20 | Yes | A100-SXM4-40GB; A100-SXM4-80GB |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | Llama-3.2-3B | ES | EsBBQ | 27,320 (10 cats) | 6 | 0 | 0 | 5, 20 | Yes | L4 |

### 1.2 Primary Result: Selected vs. Layer-Matched Random Neurons

Changed answers (`changed / n_items`, `%`) in the target category and across the remaining categories for selected neurons versus random neurons drawn uniformly from the exact same layers (primary comparisons first; sensitivity rows marked).

| Unit | Comparison | Target Cat. | Selected Flips (Target) | Random Flips (Target, min–max) | Real/Rand | Rank | Other Cats (Real vs. Rand) | Ratio (Real vs. Rand) | GPUs (Real / Rand) |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | :--- |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | Gender·20 (seeds 0-4) | Gender_identity | 1,361 / 5,672 (24.00%) | 41–76 (0.72–1.34% (mean 0.93%)) | 25.68× | 1 of 6 | 6.28% vs. 1.23% | 3.82× vs. 0.75× | A100-SXM4-80GB / A100-SXM4-40GB |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | Religion·20 (seeds 0-9) | Religion | 120 / 1,200 (10.00%) | 10–38 (0.83–3.17% (mean 1.37%)) | 7.32× | 1 of 11 | 6.50% vs. 1.41% | 1.54× vs. 0.97× | A100-SXM4-80GB / A100-SXM4-40GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Gender·20 (seeds 0-4, A100-only) | Gender_identity | 504 / 5,672 (8.89%) | 70–94 (1.23–1.66% (mean 1.46%)) | 6.10× | 1 of 6 | 4.17% vs. 1.58% | 2.13× vs. 0.92× | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Religion·20 (E2 seeds 0-9, all) | Religion | 133 / 1,200 (11.08%) | 13–20 (1.08–1.67% (mean 1.29%)) | 8.58× | 1 of 11 | 9.61% vs. 1.63% | 1.15× vs. 0.80× | A100-SXM4-80GB / A100-SXM4-40GB; L4 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | Gender·20 (seeds 0-4) | Gender_identity | 397 / 5,672 (7.00%) | 28–64 (0.49–1.13% (mean 0.76%)) | 9.19× | 1 of 6 | 7.48% vs. 0.83% | 0.94× vs. 0.91× | A100-SXM4-40GB / A100-SXM4-40GB |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | Religion·20 (seeds 0-4) | Religion | 79 / 1,200 (6.58%) | 8–16 (0.67–1.33% (mean 0.93%)) | 7.05× | 1 of 6 | 7.98% vs. 0.59% | 0.83× vs. 1.63× | A100-SXM4-40GB / A100-SXM4-40GB |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | Gender·20 (seeds 0-4) | Gender_identity | 928 / 5,672 (16.36%) | 41–81 (0.72–1.43% (mean 0.93%)) | 17.58× | 1 of 6 | 4.31% vs. 1.15% | 3.79× vs. 0.80× | A100-SXM4-40GB / A100-SXM4-40GB |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | Religion·20 (seeds 0-4) | Religion | 163 / 1,200 (13.58%) | 11–19 (0.92–1.58% (mean 1.23%)) | 11.01× | 1 of 6 | 5.49% vs. 1.18% | 2.48× vs. 1.07× | A100-SXM4-40GB / A100-SXM4-40GB |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | Gender·5 (seeds 0-4) | Gender | 983 / 4,832 (20.34%) | 12–86 (0.25–1.78% (mean 0.62%)) | 32.77× | 1 of 6 | 4.21% vs. 0.77% | 4.83× vs. 0.77× | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Gender·20 (L4 seed0 + A100 seeds 1-4, sensitivity) *(sens.)* | Gender_identity | 504 / 5,672 (8.89%) | 77–94 (1.36–1.66% (mean 1.53%)) | 5.81× | 1 of 6 | 4.17% vs. 1.63% | 2.13× vs. 0.94× | A100-SXM4-80GB / A100-SXM4-80GB; L4 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | Religion·20 (E2 seeds 2-9, A100-only) *(sens.)* | Religion | 133 / 1,200 (11.08%) | 13–16 (1.08–1.33% (mean 1.21%)) | 9.17× | 1 of 9 | 9.61% vs. 1.58% | 1.15× vs. 0.77× | A100-SXM4-80GB / A100-SXM4-40GB |

### 1.3 Target-Category Concentration (All Items Pooled)

Changed answers for every real set (`K = 5`, `K = 20`, and EsBBQ grid sets) in its target category versus the unweighted mean of the other categories, with 95% item-bootstrap CI, concentration ratio, rank, and total benchmark flips.

| Unit | Set | Target Cat. | K | Last Layer | Target Flips | 95% CI (%) | Other Mean | Ratio | Rank | All Benchmark Flips |
| :--- | :--- | :--- | ---: | :---: | ---: | :---: | ---: | ---: | :---: | ---: |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender20` | Gender_identity | 20 | 15/20 | 1,361 / 5,672 (24.00%) | [22.88, 25.09] | 6.28% | 3.82× | 1/11 | 5,291 / 58,492 (9.05%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race20` | Race_ethnicity | 20 | 20/20 | 519 / 6,880 (7.54%) | [6.96, 8.17] | 5.43% | 1.39× | 3/11 | 3,399 / 58,492 (5.81%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion20` | Religion | 20 | 19/20 | 120 / 1,200 (10.00%) | [8.25, 11.67] | 6.50% | 1.54× | 2/11 | 4,027 / 58,492 (6.88%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender5` | Gender_identity | 5 | 4/5 | 1,323 / 5,672 (23.33%) | [22.23, 24.40] | 4.39% | 5.32× | 1/11 | 4,598 / 58,492 (7.86%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 415 / 6,880 (6.03%) | [5.49, 6.60] | 3.78% | 1.59× | 3/11 | 2,375 / 58,492 (4.06%) |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion5` | Religion | 5 | 5/5 | 118 / 1,200 (9.83%) | [8.17, 11.42] | 3.55% | 2.77× | 1/11 | 2,487 / 58,492 (4.25%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender20` | Gender_identity | 20 | 8/20 | 928 / 5,672 (16.36%) | [15.39, 17.28] | 4.31% | 3.79× | 1/11 | 3,860 / 58,492 (6.60%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race20` | Race_ethnicity | 20 | 18/20 | 519 / 6,880 (7.54%) | [6.95, 8.18] | 4.14% | 1.82× | 1/11 | 3,131 / 58,492 (5.35%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion20` | Religion | 20 | 17/20 | 163 / 1,200 (13.58%) | [11.75, 15.59] | 5.49% | 2.48× | 1/11 | 3,866 / 58,492 (6.61%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender5` | Gender_identity | 5 | 2/5 | 913 / 5,672 (16.10%) | [15.14, 17.00] | 2.95% | 5.45× | 1/11 | 2,953 / 58,492 (5.05%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 351 / 6,880 (5.10%) | [4.61, 5.58] | 2.84% | 1.79× | 2/11 | 2,255 / 58,492 (3.86%) |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion5` | Religion | 5 | 4/5 | 113 / 1,200 (9.42%) | [7.75, 11.08] | 3.41% | 2.76× | 1/11 | 2,471 / 58,492 (4.22%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender20` | Gender_identity | 20 | 18/20 | 397 / 5,672 (7.00%) | [6.38, 7.63] | 7.48% | 0.94× | 8/11 | 4,353 / 58,492 (7.44%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race20` | Race_ethnicity | 20 | 19/20 | 371 / 6,880 (5.39%) | [4.90, 5.94] | 6.33% | 0.85× | 9/11 | 3,695 / 58,492 (6.32%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion20` | Religion | 20 | 20/20 | 79 / 1,200 (6.58%) | [5.17, 8.00] | 7.98% | 0.83× | 9/11 | 4,746 / 58,492 (8.11%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender5` | Gender_identity | 5 | 5/5 | 245 / 5,672 (4.32%) | [3.83, 4.85] | 5.22% | 0.83× | 8/11 | 3,132 / 58,492 (5.35%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 199 / 6,880 (2.89%) | [2.51, 3.31] | 3.03% | 0.96× | 5/11 | 2,123 / 58,492 (3.63%) |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion5` | Religion | 5 | 5/5 | 49 / 1,200 (4.08%) | [3.08, 5.25] | 3.89% | 1.05× | 5/11 | 2,560 / 58,492 (4.38%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender20` | Gender_identity | 20 | 6/20 | 504 / 5,672 (8.89%) | [8.16, 9.61] | 4.17% | 2.13× | 1/11 | 2,912 / 58,492 (4.98%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race20` | Race_ethnicity | 20 | 18/20 | 857 / 6,880 (12.46%) | [11.72, 13.24] | 8.03% | 1.55× | 1/11 | 5,857 / 58,492 (10.01%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion20` | Religion | 20 | 14/20 | 133 / 1,200 (11.08%) | [9.50, 12.84] | 9.61% | 1.15× | 3/11 | 6,261 / 58,492 (10.70%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender5` | Gender_identity | 5 | 0/5 | 75 / 5,672 (1.32%) | [1.02, 1.60] | 1.46% | 0.90× | 8/11 | 887 / 58,492 (1.52%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race5` | Race_ethnicity | 5 | 5/5 | 329 / 6,880 (4.78%) | [4.26, 5.29] | 4.26% | 1.12× | 4/11 | 2,417 / 58,492 (4.13%) |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion5` | Religion | 5 | 4/5 | 64 / 1,200 (5.33%) | [4.17, 6.58] | 7.81% | 0.68× | 9/11 | 5,075 / 58,492 (8.68%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_gender20` | Gender_identity | 20 | 8/20 | 1,150 / 5,672 (20.28%) | - | 6.67% | 3.04× | 1/11 | 5,428 / 58,492 (9.28%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_race20` | Race_ethnicity | 20 | 13/20 | 797 / 6,880 (11.58%) | - | 7.51% | 1.54× | 4/11 | 5,255 / 58,492 (8.98%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_religion20` | Religion | 20 | 18/20 | 270 / 1,200 (22.50%) | - | 8.32% | 2.70× | 1/11 | 5,556 / 58,492 (9.50%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_gender5` | Gender_identity | 5 | 1/5 | 108 / 5,672 (1.90%) | - | 1.93% | 0.99× | 8/11 | 1,157 / 58,492 (1.98%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_race5` | Race_ethnicity | 5 | 4/5 | 538 / 6,880 (7.82%) | - | 6.12% | 1.28× | 4/11 | 3,675 / 58,492 (6.28%) |
| [`llama-3.2-1b_en`](units/llama-3.2-1b_en.md) | `real_religion5` | Religion | 5 | 4/5 | 141 / 1,200 (11.75%) | - | 4.43% | 2.65× | 1/11 | 2,934 / 58,492 (5.02%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp5` | Religion | 40 | 34/40 | 78 / 648 (12.04%) | [9.41, 14.51] | 7.75% | 1.55× | 3/10 | 1,857 / 27,320 (6.80%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp9` | Age | 20 | 20/20 | 214 / 4,068 (5.26%) | [4.67, 5.92] | 6.38% | 0.82× | 8/10 | 1,527 / 27,320 (5.59%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp11` | Gender | 20 | 17/20 | 947 / 4,832 (19.60%) | [18.56, 20.70] | 5.79% | 3.38× | 1/10 | 2,334 / 27,320 (8.54%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp8` | Age | 10 | 10/10 | 212 / 4,068 (5.21%) | [4.57, 5.88] | 4.47% | 1.17× | 2/10 | 1,071 / 27,320 (3.92%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp7` | Age | 5 | 5/5 | 207 / 4,068 (5.09%) | [4.40, 5.78] | 3.27% | 1.56× | 2/10 | 862 / 27,320 (3.16%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp10` | Gender | 5 | 5/5 | 983 / 4,832 (20.34%) | [19.31, 21.40] | 4.21% | 4.83× | 1/10 | 2,048 / 27,320 (7.50%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp12` | RaceEthnicity | 5 | 5/5 | 69 / 3,716 (1.86%) | [1.45, 2.29] | 1.89% | 0.98× | 5/10 | 484 / 27,320 (1.77%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp2` | Religion | 5 | 4/5 | 22 / 648 (3.40%) | [2.16, 4.94] | 2.74% | 1.24× | 3/10 | 775 / 27,320 (2.84%) |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp1` | Religion | 1 | 1/1 | 9 / 648 (1.39%) | [0.62, 2.47] | 1.72% | 0.81× | 7/10 | 460 / 27,320 (1.68%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp11` | Gender | 20 | 16/20 | 635 / 4,832 (13.14%) | [12.17, 14.09] | 5.39% | 2.44× | 1/10 | 1,876 / 27,320 (6.87%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp14` | RaceEthnicity | 20 | 20/20 | 324 / 3,716 (8.72%) | [7.75, 9.63] | 8.58% | 1.02× | 6/10 | 2,419 / 27,320 (8.85%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp13` | Religion | 20 | 17/20 | 44 / 648 (6.79%) | [4.78, 8.80] | 5.01% | 1.36× | 2/10 | 1,368 / 27,320 (5.01%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp10` | Gender | 5 | 4/5 | 138 / 4,832 (2.86%) | [2.44, 3.35] | 3.27% | 0.87× | 6/10 | 914 / 27,320 (3.35%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp12` | RaceEthnicity | 5 | 5/5 | 110 / 3,716 (2.96%) | [2.42, 3.50] | 3.35% | 0.88× | 6/10 | 895 / 27,320 (3.28%) |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp2` | Religion | 5 | 4/5 | 23 / 648 (3.55%) | [2.16, 5.09] | 2.60% | 1.36× | 2/10 | 732 / 27,320 (2.68%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp11` | Gender | 20 | 10/20 | 816 / 4,832 (16.89%) | [15.85, 17.98] | 6.73% | 2.51× | 1/10 | 2,337 / 27,320 (8.55%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp14` | RaceEthnicity | 20 | 12/20 | 529 / 3,716 (14.24%) | [13.13, 15.31] | 12.19% | 1.17× | 5/10 | 3,307 / 27,320 (12.10%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp13` | Religion | 20 | 13/20 | 101 / 648 (15.59%) | [12.50, 18.36] | 10.96% | 1.42× | 3/10 | 3,055 / 27,320 (11.18%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp10` | Gender | 5 | 2/5 | 74 / 4,832 (1.53%) | [1.20, 1.88] | 2.79% | 0.55× | 10/10 | 684 / 27,320 (2.50%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp12` | RaceEthnicity | 5 | 5/5 | 298 / 3,716 (8.02%) | [7.13, 8.83] | 8.76% | 0.92× | 6/10 | 2,376 / 27,320 (8.70%) |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp2` | Religion | 5 | 3/5 | 59 / 648 (9.10%) | [6.79, 11.27] | 3.85% | 2.36× | 1/10 | 1,070 / 27,320 (3.92%) |

### 1.4 Target-Category Concentration (Disambiguated Items Only)

Same localization ratios and ranks restricted to disambiguated items (`context = disambig`, matching the EsBBQ summary view reported in `R1_exp.md`).

<details>
<summary>Expand disambiguated-only localization table (all real sets with per-item logs)</summary>

| Unit | Set | Target Cat. | K | Disambig Target (%) | Disambig Other Mean (%) | Ratio | Rank |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender20` | Gender_identity | 20 | 14.67% | 3.75% | 3.91× | 1/11 |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race20` | Race_ethnicity | 20 | 4.68% | 3.03% | 1.55× | 3/11 |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion20` | Religion | 20 | 7.00% | 3.58% | 1.96× | 1/11 |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_gender5` | Gender_identity | 5 | 14.21% | 2.84% | 5.00× | 1/11 |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_race5` | Race_ethnicity | 5 | 3.69% | 2.04% | 1.81× | 3/11 |
| [`llama-3.1-8b_en`](units/llama-3.1-8b_en.md) | `real_religion5` | Religion | 5 | 7.00% | 1.93% | 3.63× | 1/11 |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender20` | Gender_identity | 20 | 8.43% | 2.42% | 3.49× | 1/11 |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race20` | Race_ethnicity | 20 | 3.75% | 2.14% | 1.75× | 3/11 |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion20` | Religion | 20 | 9.50% | 2.83% | 3.36× | 1/11 |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_gender5` | Gender_identity | 5 | 8.04% | 1.58% | 5.10× | 1/11 |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_race5` | Race_ethnicity | 5 | 2.70% | 1.39% | 1.94× | 2/11 |
| [`salamandra-7b_en`](units/salamandra-7b_en.md) | `real_religion5` | Religion | 5 | 6.33% | 1.61% | 3.94× | 1/11 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender20` | Gender_identity | 20 | 3.28% | 3.20% | 1.03× | 5/11 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race20` | Race_ethnicity | 20 | 1.57% | 2.55% | 0.62× | 8/11 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion20` | Religion | 20 | 2.33% | 3.37% | 0.69× | 6/11 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_gender5` | Gender_identity | 5 | 1.38% | 2.19% | 0.63× | 10/11 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_race5` | Race_ethnicity | 5 | 0.67% | 1.46% | 0.46× | 8/11 |
| [`qwen3-4b-base_en`](units/qwen3-4b-base_en.md) | `real_religion5` | Religion | 5 | 1.33% | 1.58% | 0.85× | 5/11 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender20` | Gender_identity | 20 | 6.63% | 3.31% | 2.00× | 1/11 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race20` | Race_ethnicity | 20 | 10.00% | 6.14% | 1.63× | 1/11 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion20` | Religion | 20 | 9.50% | 7.40% | 1.28× | 2/11 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_gender5` | Gender_identity | 5 | 1.16% | 1.14% | 1.02× | 3/11 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_race5` | Race_ethnicity | 5 | 3.92% | 3.48% | 1.13× | 5/11 |
| [`llama-3.2-3b_en`](units/llama-3.2-3b_en.md) | `real_religion5` | Religion | 5 | 3.83% | 5.99% | 0.64× | 10/11 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp5` | Religion | 40 | 11.57% | 6.78% | 1.71× | 3/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp9` | Age | 20 | 5.40% | 5.56% | 0.97× | 5/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp11` | Gender | 20 | 15.66% | 5.16% | 3.04× | 1/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp8` | Age | 10 | 5.19% | 3.98% | 1.30× | 2/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp7` | Age | 5 | 5.30% | 2.81% | 1.89× | 2/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp10` | Gender | 5 | 16.47% | 3.95% | 4.17× | 1/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp12` | RaceEthnicity | 5 | 1.45% | 1.71% | 0.85× | 6/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp2` | Religion | 5 | 3.01% | 2.20% | 1.37× | 2/10 |
| [`salamandra-7b_es`](units/salamandra-7b_es.md) | `exp1` | Religion | 1 | 1.39% | 1.50% | 0.93× | 6/10 |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp11` | Gender | 20 | 10.79% | 4.20% | 2.57× | 1/10 |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp14` | RaceEthnicity | 20 | 7.76% | 6.93% | 1.12× | 5/10 |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp13` | Religion | 20 | 7.87% | 4.21% | 1.87× | 1/10 |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp10` | Gender | 5 | 2.40% | 2.77% | 0.87× | 6/10 |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp12` | RaceEthnicity | 5 | 2.37% | 2.83% | 0.84× | 6/10 |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp2` | Religion | 5 | 3.94% | 2.20% | 1.79× | 2/10 |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp11` | Gender | 20 | 14.96% | 5.77% | 2.59× | 1/10 |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp14` | RaceEthnicity | 20 | 12.30% | 11.76% | 1.05× | 6/10 |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp13` | Religion | 20 | 15.74% | 9.66% | 1.63× | 2/10 |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp10` | Gender | 5 | 1.11% | 2.19% | 0.51× | 10/10 |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp12` | RaceEthnicity | 5 | 6.87% | 8.49% | 0.81× | 6/10 |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp2` | Religion | 5 | 8.33% | 3.28% | 2.54× | 1/10 |

</details>

---

## 2. Unit Dossiers (Level 2)

Each unit dossier follows the same layout: unit summary, real-vs-random controls, full `set × category` flip matrix, neuron/item Jaccard overlap, secondary BBQ/EsBBQ bias and accuracy metrics, baseline margin mechanism, and demographic group/topic breakdowns.

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

---

## 3. Diagnostics and Controls (Level 3)

### 3.1 Single-Neuron Ablation (`Llama-3.1-8B EN`, `L31·6950`)

Exploratory decomposition of `real_race5` and `real_religion5` into their shared neuron (`abl_shared1` = `L31·6950`) and the remaining four neurons of each set (`abl_race4`, `abl_religion4`), showing changed answers across contexts and Jaccard overlap with the original Top-5 sets.

| Set | Neurons (L31) | Religion (All) | Religion (Amb/Dis) | Race (All) | Race (Amb/Dis) | Gender (All) | Jacc vs. Race5 (Rel/Race) | Jacc vs. Rel5 (Rel/Race) |
| :--- | :--- | ---: | :---: | ---: | :---: | ---: | :---: | :---: |
| `real_race5` | L31.1495, L31.2100, L31.6466, L31.6532, L31.6950 | 122 / 1,200 (10.17%) | 13.50% / 6.83% | 415 / 6,880 (6.03%) | 8.37% / 3.69% | 105 / 5,672 (1.85%) | 1.00 / 1.00 | 0.55 / 0.66 |
| `real_religion5` | L31.33, L31.311, L31.2398, L31.2564, L31.6950 | 118 / 1,200 (9.83%) | 12.67% / 7.00% | 426 / 6,880 (6.19%) | 8.87% / 3.52% | 105 / 5,672 (1.85%) | 0.55 / 0.66 | 1.00 / 1.00 |
| `abl_shared1` | L31.6950 | 116 / 1,200 (9.67%) | 12.83% / 6.50% | 402 / 6,880 (5.84%) | 8.34% / 3.34% | 60 / 5,672 (1.06%) | 0.87 / 0.74 | 0.59 / 0.69 |
| `abl_race4` | L31.1495, L31.2100, L31.6466, L31.6532 | 19 / 1,200 (1.58%) | 1.67% / 1.50% | 155 / 6,880 (2.25%) | 3.28% / 1.22% | 89 / 5,672 (1.57%) | 0.09 / 0.22 | 0.09 / 0.16 |
| `abl_religion4` | L31.33, L31.311, L31.2398, L31.2564 | 74 / 1,200 (6.17%) | 7.67% / 4.67% | 175 / 6,880 (2.54%) | 3.60% / 1.48% | 81 / 5,672 (1.43%) | 0.09 / 0.13 | 0.26 / 0.22 |

L2 norm of the `down_proj` column and within-layer percentiles in Layer 31 (14,336 neurons) for all nine neurons of `real_race5` and `real_religion5`.

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

### 3.2 Baseline Indecision Mechanism (Margin Deciles)

Median baseline log-likelihood margin (`top − second` across person0/person1/unknown) of changed versus unchanged items across all categories, alongside flip rate in decile 1 (most undecided) versus decile 10 (most confident) for `K = 20` real sets (and `Gender·5` in Salamandra-7B ES).

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
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp10` | 914 | 0.094 | 1.000 | 25.66% | 0.29% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp13` | 1,368 | 0.125 | 1.004 | 33.42% | 0.51% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp11` | 1,876 | 0.125 | 1.031 | 38.65% | 0.92% | 0.00% |
| [`llama-3.1-8b_es`](units/llama-3.1-8b_es.md) | `exp14` | 2,419 | 0.156 | 1.062 | 40.41% | 1.79% | 0.00% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp10` | 684 | 0.094 | 1.055 | 19.88% | 0.11% | 0.00% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp13` | 3,055 | 0.250 | 1.156 | 42.64% | 6.84% | 0.00% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp11` | 2,337 | 0.219 | 1.125 | 39.02% | 1.90% | 0.07% |
| [`llama-3.2-3b_es`](units/llama-3.2-3b_es.md) | `exp14` | 3,307 | 0.312 | 1.156 | 40.81% | 10.43% | 0.00% |

### 3.3 Hardware Noise Controls (L4 vs. A100)

K3 baseline rerun comparison for `Llama-3.1-8B ES` (unpruned model evaluated on A100 vs. L4 across all 27,320 EsBBQ items), establishing the cross-GPU hardware flip floor.

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

Sensitivity comparison of the identical random set `rand_gender20_seed0` in `Llama-3.2-3B EN` evaluated on A100-SXM4-80GB versus L4 against the A100-SXM4-40GB baseline.

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

### 3.4 Provenance and Source Tables

Every number above is generated deterministically from the consolidated CSV files in `results/e3/tables/`, each traced by SHA-256 to its source Drive file in `tables/inputs_manifest.json`.

| CSV Table | Rows | Columns | Size |
| :--- | ---: | ---: | ---: |
| [`tables/e3_ablation_l31_6950.csv`](tables/e3_ablation_l31_6950.csv) | 45 | 12 | 6,176 B |
| [`tables/e3_esbbq_topics.csv`](tables/e3_esbbq_topics.csv) | 1,596 | 8 | 132,156 B |
| [`tables/e3_flips_by_category.csv`](tables/e3_flips_by_category.csv) | 3,420 | 14 | 440,478 B |
| [`tables/e3_gpu_noise_3b.csv`](tables/e3_gpu_noise_3b.csv) | 11 | 10 | 1,860 B |
| [`tables/e3_gpu_noise_k3.csv`](tables/e3_gpu_noise_k3.csv) | 11 | 13 | 1,062 B |
| [`tables/e3_groups.csv`](tables/e3_groups.csv) | 1,644 | 8 | 127,868 B |
| [`tables/e3_inventory.csv`](tables/e3_inventory.csv) | 8 | 16 | 1,754 B |
| [`tables/e3_l31_weights.csv`](tables/e3_l31_weights.csv) | 9 | 23 | 2,961 B |
| [`tables/e3_localization_summary.csv`](tables/e3_localization_summary.csv) | 54 | 29 | 15,492 B |
| [`tables/e3_margin_deciles.csv`](tables/e3_margin_deciles.csv) | 570 | 8 | 41,376 B |
| [`tables/e3_margin_summary.csv`](tables/e3_margin_summary.csv) | 238 | 7 | 16,275 B |
| [`tables/e3_neuron_overlap.csv`](tables/e3_neuron_overlap.csv) | 162 | 16 | 27,080 B |
| [`tables/e3_provenance.csv`](tables/e3_provenance.csv) | 118 | 18 | 28,243 B |
| [`tables/e3_real_vs_random.csv`](tables/e3_real_vs_random.csv) | 11 | 29 | 4,516 B |
| [`tables/e3_secondary_metrics.csv`](tables/e3_secondary_metrics.csv) | 477 | 28 | 91,406 B |
