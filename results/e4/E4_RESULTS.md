# E4 Results — General-Capability Evaluation After Neuron Zeroing

Paired item-level bootstrap (`B = 10,000`, `seed = 0`, equivalence margin `±1.0 pp`, chance exclusion `< 35%`) across five English base models and five benchmarks (`inputs.json`, built `2026-11-24T11:36:48+00:00`; refreshed `9` newer Drive logs (2026-11-23 `llama_3.2_3b_en` A100 reruns)).

## Navigation

- **Level 1 — Cross-Model Overview:** [1.1 Inventory & Protocol](#11-experiment-inventory-and-protocol) · [1.2 Main Matrix (`K = 20`, Real vs. Baseline)](#12-main-matrix-k--20-real-sets-vs-baseline) · [1.3 Real vs. Random (`Gender·20`)](#13-real-vs-layer-matched-random-control-gender20-seed-0) · [1.4 `K = 5` vs. `K = 20` Comparison](#14-k--5-vs-k--20-comparison-real-sets-vs-baseline)
- **Level 2 — Unit Dossiers:** [2. Index of Units](#2-unit-dossiers-level-2)
- **Level 3 — Diagnostics & Provenance:** [3.1 WikiText Multi-Seed Random Controls](#31-wikitext-multi-seed-random-controls) · [3.2 MMLU Micro vs. Macro](#32-mmlu-micro-vs-macro-57-subjects) · [3.3 Hardware Provenance (A100 vs. L4) & Checks](#33-hardware-provenance-a100-vs-l4-and-extraction-checks) · [3.4 Source Tables](#34-source-tables)

---

## 1. Cross-Model Overview (Level 1)

### 1.1 Experiment Inventory and Protocol

All five E4 units, counts of real and random sets evaluated, extraction verification checks, and hardware used for baseline/real versus random evaluations.

| Unit | Model | Real Sets | Random (Full) | Random (WikiText-only) | K | MMLU Status | Checks | Baseline/Real GPUs | Random GPUs |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: | :---: | :--- | :--- |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | Llama-3.1-8B | 6 | 1 | 14 | 5, 20 | Included | 51 (0 problems) | A100-SXM4-80GB; unknown | A100-SXM4-40GB; L4; unknown |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | Qwen3-4B-Base | 6 | 1 | 9 | 5, 20 | Included | 46 (0 problems) | A100-SXM4-40GB; unknown | A100-SXM4-40GB; L4; unknown |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | Salamandra-7B | 6 | 1 | 9 | 5, 20 | Included | 46 (0 problems) | A100-SXM4-40GB; unknown | A100-SXM4-40GB; L4; unknown |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | Llama-3.2-3B | 6 | 1 | 4 | 5, 20 | Included | 41 (0 problems) | A100-SXM4-80GB | A100-SXM4-80GB |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | Llama-3.2-1B | 6 | 0 | 0 | 5, 20 | Excluded (<35%) | 32 (0 problems) | A100-SXM4-40GB; unknown | - |

### 1.2 Main Matrix: `K = 20` Real Sets vs. Baseline

Paired differences (`real − baseline`, 95% bootstrap CI, equivalence verdict at `±1 pp`) across all five benchmarks for every `K = 20` real set, with unpruned baseline scores shown on the first row of each model.

| Unit | Model | Set | WikiText (Δ%) | ARC-C (pp) | HellaSwag (pp) | HellaSwag-es (pp) | MMLU micro (pp) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | Llama-3.1-8B | `baseline` *(raw)* | 7.544 | 54.61% | 79.30% | 66.88% | 65.28% |
|  |  | `real_gender20` | +0.72% [+0.50, +0.94] | +0.51 [-0.26, +1.28] (inconcl) | +0.06 [-0.17, +0.30] (equiv) | -0.14 [-0.39, +0.11] (equiv) | +0.06 [-0.16, +0.28] (equiv) |
|  |  | `real_race20` | +0.92% [+0.73, +1.10] | +1.02 [+0.26, +1.88] (inconcl) | +0.03 [-0.18, +0.24] (equiv) | -0.13 [-0.38, +0.13] (equiv) | +0.04 [-0.29, +0.38] (equiv) |
|  |  | `real_religion20` | +0.65% [+0.45, +0.84] | +0.51 [-0.26, +1.28] (inconcl) | -0.02 [-0.23, +0.19] (equiv) | -0.12 [-0.37, +0.13] (equiv) | +0.22 [-0.07, +0.51] (equiv) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | Qwen3-4B-Base | `baseline` *(raw)* | 10.374 | 51.02% | 73.69% | 62.12% | 73.09% |
|  |  | `real_gender20` | +6.33% [+5.92, +6.74] | +0.85 [-0.34, +2.13] (inconcl) | -0.65 [-1.05, -0.26] (inconcl) | +0.02 [-0.36, +0.39] (equiv) | -0.37 [-0.64, -0.10] (equiv) |
|  |  | `real_race20` | +3.77% [+3.49, +4.06] | +1.19 [+0.00, +2.39] (inconcl) | -0.60 [-0.95, -0.26] (equiv) | -0.05 [-0.47, +0.36] (equiv) | -0.25 [-0.47, -0.03] (equiv) |
|  |  | `real_religion20` | +7.48% [+7.05, +7.94] | +0.17 [-1.11, +1.45] (inconcl) | -0.73 [-1.15, -0.31] (inconcl) | +0.02 [-0.37, +0.41] (equiv) | -0.32 [-0.56, -0.08] (equiv) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | Salamandra-7B | `baseline` *(raw)* | 6.220 | 48.21% | 76.47% | 67.87% | 47.03% |
|  |  | `real_gender20` | +0.41% [+0.25, +0.57] | -0.26 [-0.94, +0.43] (equiv) | -0.08 [-0.29, +0.13] (equiv) | -0.11 [-0.31, +0.10] (equiv) | -0.47 [-0.84, -0.11] (equiv) |
|  |  | `real_race20` | +0.87% [+0.66, +1.10] | -0.26 [-1.11, +0.51] (inconcl) | -0.14 [-0.35, +0.07] (equiv) | -0.22 [-0.47, +0.02] (equiv) | -1.69 [-2.26, -1.13] (WORSE) |
|  |  | `real_religion20` | +0.68% [+0.52, +0.84] | +0.00 [-0.60, +0.60] (equiv) | -0.10 [-0.31, +0.11] (equiv) | +0.10 [-0.15, +0.34] (equiv) | -0.66 [-1.00, -0.33] (equiv) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | Llama-3.2-3B | `baseline` *(raw)* | 9.531 | 46.33% | 74.04% | 58.95% | 56.53% |
|  |  | `real_gender20` | +0.77% [+0.52, +1.03] | +0.00 [-0.85, +0.77] (equiv) | +0.04 [-0.25, +0.33] (equiv) | -0.05 [-0.29, +0.18] (equiv) | -0.14 [-0.39, +0.11] (equiv) |
|  |  | `real_race20` | +2.58% [+2.35, +2.81] | -0.77 [-1.79, +0.26] (inconcl) | -0.25 [-0.54, +0.04] (equiv) | -0.19 [-0.48, +0.10] (equiv) | -0.95 [-1.41, -0.51] (inconcl) |
|  |  | `real_religion20` | +2.86% [+2.59, +3.13] | -0.51 [-1.62, +0.60] (inconcl) | -0.29 [-0.62, +0.02] (equiv) | -0.35 [-0.68, -0.02] (equiv) | -0.04 [-0.39, +0.30] (equiv) |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | Llama-3.2-1B | `baseline` *(raw)* | 11.985 | 37.03% | 64.30% | 47.23% | 31.12% |
|  |  | `real_gender20` | +1.30% [+1.07, +1.55] | -0.85 [-2.05, +0.34] (inconcl) | -0.25 [-0.56, +0.06] (equiv) | -0.17 [-0.47, +0.13] (equiv) | +1.20 [+0.71, +1.69] (excl (<35%)) |
|  |  | `real_race20` | +0.90% [+0.68, +1.12] | -0.43 [-1.45, +0.51] (inconcl) | -0.03 [-0.26, +0.21] (equiv) | +0.12 [-0.14, +0.37] (equiv) | +0.44 [+0.03, +0.85] (excl (<35%)) |
|  |  | `real_religion20` | +1.21% [+0.73, +1.95] | +0.00 [-0.94, +0.94] (equiv) | -0.38 [-0.64, -0.12] (equiv) | -0.26 [-0.53, +0.03] (equiv) | +1.03 [+0.52, +1.55] (excl (<35%)) |

### 1.3 Real vs. Layer-Matched Random Control (`Gender·20`, seed 0)

Direct comparison of `real_gender20` against `rand_gender20_seed0` (same layer profile, full 5-task battery) across the four models with a full random control, with explicit GPU hardware per side.

| Unit | Task | Real − Baseline | Random(seed0) − Baseline | Real − Random(seed0) | GPUs (Real / Rand) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | wikitext | +0.72% [+0.50, +0.94] | +0.07% [+0.00, +0.19] | +0.65% [+0.42, +0.87] | A100-SXM4-80GB / A100-SXM4-40GB |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | arc_challenge | +0.51 [-0.26, +1.28] (inconcl) | +0.26 [-0.26, +0.85] (equiv) | +0.26 [-0.51, +1.02] (inconcl) | A100-SXM4-80GB / L4 |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag | +0.06 [-0.17, +0.30] (equiv) | +0.05 [-0.08, +0.18] (equiv) | +0.01 [-0.23, +0.25] (equiv) | A100-SXM4-80GB / L4 |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag_es | -0.14 [-0.39, +0.11] (equiv) | -0.14 [-0.31, +0.03] (equiv) | +0.00 [-0.27, +0.27] (equiv) | A100-SXM4-80GB / L4 |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | mmlu | +0.06 [-0.16, +0.28] (equiv) | +0.26 [+0.04, +0.48] (equiv) | -0.20 [-0.46, +0.07] (equiv) | A100-SXM4-80GB / A100-SXM4-40GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | wikitext | +6.33% [+5.92, +6.74] | +0.19% [+0.06, +0.36] | +6.13% [+5.73, +6.52] | A100-SXM4-40GB / A100-SXM4-40GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | arc_challenge | +0.85 [-0.34, +2.13] (inconcl) | +0.51 [-0.17, +1.28] (inconcl) | +0.34 [-0.85, +1.54] (inconcl) | A100-SXM4-40GB / L4 |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag | -0.65 [-1.05, -0.26] (inconcl) | +0.11 [-0.07, +0.29] (equiv) | -0.76 [-1.16, -0.37] (inconcl) | A100-SXM4-40GB / L4 |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag_es | +0.02 [-0.36, +0.39] (equiv) | +0.11 [-0.09, +0.30] (equiv) | -0.09 [-0.47, +0.30] (equiv) | A100-SXM4-40GB / L4 |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | mmlu | -0.37 [-0.64, -0.10] (equiv) | +0.01 [-0.20, +0.21] (equiv) | -0.38 [-0.67, -0.09] (equiv) | A100-SXM4-40GB / L4 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | wikitext | +0.41% [+0.25, +0.57] | +0.01% [-0.00, +0.03] | +0.40% [+0.24, +0.57] | A100-SXM4-40GB / L4 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | arc_challenge | -0.26 [-0.94, +0.43] (equiv) | -0.17 [-0.77, +0.43] (equiv) | -0.09 [-0.94, +0.77] (equiv) | A100-SXM4-40GB / L4 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag | -0.08 [-0.29, +0.13] (equiv) | -0.08 [-0.23, +0.07] (equiv) | +0.00 [-0.20, +0.21] (equiv) | A100-SXM4-40GB / L4 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag_es | -0.11 [-0.31, +0.10] (equiv) | -0.01 [-0.18, +0.15] (equiv) | -0.10 [-0.32, +0.13] (equiv) | A100-SXM4-40GB / L4 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | mmlu | -0.47 [-0.84, -0.11] (equiv) | -0.09 [-0.36, +0.18] (equiv) | -0.38 [-0.73, -0.02] (equiv) | A100-SXM4-40GB / A100-SXM4-40GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | wikitext | +0.77% [+0.52, +1.03] | +0.04% [+0.00, +0.11] | +0.73% [+0.48, +0.99] | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | arc_challenge | +0.00 [-0.85, +0.77] (equiv) | +0.00 [-0.60, +0.60] (equiv) | +0.00 [-0.85, +0.85] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag | +0.04 [-0.25, +0.33] (equiv) | +0.10 [-0.03, +0.24] (equiv) | -0.06 [-0.34, +0.22] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag_es | -0.05 [-0.29, +0.18] (equiv) | -0.14 [-0.31, +0.03] (equiv) | +0.09 [-0.15, +0.32] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | mmlu | -0.14 [-0.39, +0.11] (equiv) | -0.08 [-0.27, +0.11] (equiv) | -0.06 [-0.31, +0.19] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |

### 1.4 `K = 5` vs. `K = 20` Comparison (Real Sets vs. Baseline)

Paired deltas against baseline for `K = 5` alongside `K = 20` across the four tasks evaluated at both K values (`MMLU` is evaluated at `K = 20` only).

<details>
<summary>Expand K = 5 vs. K = 20 table across all five models</summary>

| Unit | Category | WikiText (K=5 / K=20) | ARC-C (K=5 / K=20) | HellaSwag (K=5 / K=20) | HellaSwag-es (K=5 / K=20) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | Gender | +0.25% / +0.72% | -0.34 / +0.51 | -0.02 / +0.06 | +0.00 / -0.14 |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | Race | +0.13% / +0.92% | +0.34 / +1.02 | +0.06 / +0.03 | -0.01 / -0.13 |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | Religion | +0.21% / +0.65% | +0.43 / +0.51 | +0.00 / -0.02 | -0.04 / -0.12 |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | Gender | +3.72% / +6.33% | +0.60 / +0.85 | -0.37 / -0.65 | +0.11 / +0.02 |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | Race | +1.89% / +3.77% | +1.02 / +1.19 | -0.04 / -0.60 | -0.11 / -0.05 |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | Religion | +1.98% / +7.48% | +0.85 / +0.17 | -0.28 / -0.73 | +0.14 / +0.02 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | Gender | +0.07% / +0.41% | -0.26 / -0.26 | -0.09 / -0.08 | +0.04 / -0.11 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | Race | +0.39% / +0.87% | +0.09 / -0.26 | -0.12 / -0.14 | -0.02 / -0.22 |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | Religion | +0.08% / +0.68% | +0.17 / +0.00 | -0.09 / -0.10 | +0.05 / +0.10 |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | Gender | +0.52% / +0.77% | -0.26 / +0.00 | +0.05 / +0.04 | +0.00 / -0.05 |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | Race | +0.47% / +2.58% | -0.17 / -0.77 | -0.14 / -0.25 | -0.06 / -0.19 |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | Religion | +1.61% / +2.86% | -0.60 / -0.51 | -0.27 / -0.29 | -0.19 / -0.35 |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | Gender | +0.23% / +1.30% | -0.60 / -0.85 | -0.15 / -0.25 | -0.10 / -0.17 |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | Race | +0.28% / +0.90% | -0.77 / -0.43 | -0.17 / -0.03 | +0.07 / +0.12 |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | Religion | +0.38% / +1.21% | -0.34 / +0.00 | -0.19 / -0.38 | -0.18 / -0.26 |

</details>

---

## 2. Unit Dossiers (Level 2)

Each unit dossier follows the same layout: raw scores across all sets, full baseline deltas with 95% bootstrap CIs, real-vs-random contrasts across all seeds, MMLU micro-vs-macro reconciliation, and per-task GPU/batch provenance.

| Dossier | Model | Sets | Baseline/Real GPUs | Random GPUs |
| :--- | :--- | :--- | :--- | :--- |
| [`units/llama_3.1_8b_en.md`](units/llama_3.1_8b_en.md) | Llama-3.1-8B | 6 real / 1 full-rand / 14 wt-rand | A100-SXM4-80GB; unknown | A100-SXM4-40GB; L4; unknown |
| [`units/qwen3_4b_base_en.md`](units/qwen3_4b_base_en.md) | Qwen3-4B-Base | 6 real / 1 full-rand / 9 wt-rand | A100-SXM4-40GB; unknown | A100-SXM4-40GB; L4; unknown |
| [`units/salamandra_7b_en.md`](units/salamandra_7b_en.md) | Salamandra-7B | 6 real / 1 full-rand / 9 wt-rand | A100-SXM4-40GB; unknown | A100-SXM4-40GB; L4; unknown |
| [`units/llama_3.2_3b_en.md`](units/llama_3.2_3b_en.md) | Llama-3.2-3B | 6 real / 1 full-rand / 4 wt-rand | A100-SXM4-80GB | A100-SXM4-80GB |
| [`units/llama_3.2_1b_en.md`](units/llama_3.2_1b_en.md) | Llama-3.2-1B | 6 real / 0 full-rand / 0 wt-rand | A100-SXM4-40GB; unknown | - |

---

## 3. Diagnostics and Provenance (Level 3)

### 3.1 WikiText Multi-Seed Random Controls

Relative word-perplexity change (`%` vs. baseline) of `K = 20` real sets compared against all available layer-matched random seeds (note: `bf16` document log-likelihood quantization sets an interpretability floor near `~0.1%`).

| Unit | Real Set | Random Family | Base PPL | Real Δ% [95% CI] | Random Δ% (min–max, mean) | Real − Random Δ% (mean [min, max]) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `real_gender20` | `rand_gender20` (n=5) | 7.544 | +0.72% [+0.50, +0.94] | +0.00% to +0.15% (mean +0.08%) | +0.64% [+0.58, +0.72] |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `real_religion20` | `rand_religion20` (n=10) | 7.544 | +0.65% [+0.45, +0.84] | +0.01% to +0.24% (mean +0.12%) | +0.52% [+0.41, +0.63] |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `real_gender20` | `rand_gender20` (n=5) | 10.374 | +6.33% [+5.92, +6.74] | +0.06% to +0.43% (mean +0.18%) | +6.14% [+5.88, +6.27] |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `real_religion20` | `rand_religion20` (n=5) | 10.374 | +7.48% [+7.05, +7.94] | +0.02% to +0.11% (mean +0.05%) | +7.43% [+7.36, +7.46] |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `real_gender20` | `rand_gender20` (n=5) | 6.220 | +0.41% [+0.25, +0.57] | +0.01% to +0.22% (mean +0.07%) | +0.34% [+0.19, +0.40] |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `real_religion20` | `rand_religion20` (n=5) | 6.220 | +0.68% [+0.52, +0.84] | +0.01% to +0.08% (mean +0.05%) | +0.63% [+0.60, +0.67] |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | `real_gender20` | `rand_gender20` (n=5) | 9.531 | +0.77% [+0.52, +1.03] | +0.04% to +0.25% (mean +0.09%) | +0.68% [+0.52, +0.73] |

### 3.2 MMLU Micro vs. Macro (57 Subjects)

Item-weighted micro MMLU accuracy (used throughout E4 and reported by `lm-eval`) versus the unweighted 57-subject macro average stored in `_cap.json` (+0.35 to +2.64 pp higher; never mix in one table).

<details>
<summary>Expand MMLU micro vs. macro table (all evaluated units and sets)</summary>

| Unit | Set | Micro (14,042 items) | Macro (57 subjects) | Macro − Micro | `_cap.json` Value |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `baseline` | 65.28% | 66.28% | +0.99 pp | 66.26% |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `rand_gender20_seed0` | 65.54% | 66.69% | +1.15 pp | 66.67% |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `real_gender20` | 65.34% | 66.45% | +1.11 pp | 66.42% |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `real_race20` | 65.33% | 66.39% | +1.06 pp | 66.37% |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | `real_religion20` | 65.50% | 66.55% | +1.05 pp | 66.54% |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `baseline` | 73.09% | 75.59% | +2.50 pp | 75.52% |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `rand_gender20_seed0` | 73.10% | 75.56% | +2.47 pp | 75.49% |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `real_gender20` | 72.72% | 75.28% | +2.56 pp | 75.20% |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `real_race20` | 72.84% | 75.29% | +2.45 pp | 75.21% |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `real_religion20` | 72.77% | 75.41% | +2.64 pp | 75.33% |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `baseline` | 47.03% | 48.47% | +1.44 pp | 48.40% |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `rand_gender20_seed0` | 46.94% | 48.44% | +1.50 pp | 48.37% |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `real_gender20` | 46.56% | 47.97% | +1.41 pp | 47.91% |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `real_race20` | 45.34% | 46.39% | +1.06 pp | 46.35% |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | `real_religion20` | 46.38% | 47.82% | +1.44 pp | 47.76% |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | `baseline` | 56.53% | 57.90% | +1.37 pp | 57.86% |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | `rand_gender20_seed0` | 56.45% | 57.83% | +1.38 pp | 57.79% |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | `real_gender20` | 56.39% | 57.77% | +1.38 pp | 57.72% |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | `real_race20` | 55.58% | 56.75% | +1.18 pp | 56.72% |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | `real_religion20` | 56.49% | 57.81% | +1.32 pp | 57.77% |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | `baseline` | 31.12% | 31.60% | +0.48 pp | 31.59% |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | `real_gender20` | 32.32% | 32.91% | +0.59 pp | 32.90% |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | `real_race20` | 31.56% | 31.91% | +0.35 pp | 31.91% |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | `real_religion20` | 32.15% | 32.61% | +0.46 pp | 32.60% |

</details>

### 3.3 Hardware Provenance (A100 vs. L4) and Extraction Checks

Cells where the random control ran on a different GPU architecture (L4) than the baseline/real sets (A100), plus overall extraction integrity across all per-item logs.

| Unit | Task | Baseline GPU | Real Gender·20 GPU | Random Gender·20 (seed0) GPU | Hardware Match |
| :--- | :--- | :--- | :--- | :--- | :---: |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | wikitext | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-40GB | Matched (A100) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | arc_challenge | A100-SXM4-80GB | A100-SXM4-80GB | L4 | Mixed (A100 vs. L4) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag | A100-SXM4-80GB | A100-SXM4-80GB | L4 | Mixed (A100 vs. L4) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag_es | A100-SXM4-80GB | A100-SXM4-80GB | L4 | Mixed (A100 vs. L4) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | mmlu | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-40GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | wikitext | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-40GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | arc_challenge | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag_es | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | mmlu | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | wikitext | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | arc_challenge | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag_es | A100-SXM4-40GB | A100-SXM4-40GB | L4 | Mixed (A100 vs. L4) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | mmlu | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-40GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | wikitext | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | arc_challenge | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag_es | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | mmlu | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |

Extraction verification (`tables/checks.csv`): **216** unit × set × task tables checked, **0** problems detected, 100% item-key and `doc_hash` alignment with baseline.

### 3.4 Source Tables

All numbers in this report and its unit dossiers are rendered directly from the verified CSV tables under `results/e4/tables/` (and mirrored in `results/e4/e4_capabilities_table.csv`).

| CSV Table | Rows | Columns | Size |
| :--- | ---: | ---: | ---: |
| [`tables/checks.csv`](tables/checks.csv) | 216 | 11 | 23,314 B |
| [`tables/deltas.csv`](tables/deltas.csv) | 247 | 19 | 60,098 B |
| [`tables/e4_capabilities_table.csv`](tables/e4_capabilities_table.csv) | 247 | 10 | 37,372 B |
| [`tables/e4_inventory.csv`](tables/e4_inventory.csv) | 5 | 14 | 714 B |
| [`tables/e4_wikitext_random_summary.csv`](tables/e4_wikitext_random_summary.csv) | 7 | 15 | 1,950 B |
| [`tables/mmlu_micro_macro.csv`](tables/mmlu_micro_macro.csv) | 24 | 9 | 3,135 B |
| [`tables/provenance.csv`](tables/provenance.csv) | 216 | 8 | 28,479 B |
| [`tables/raw_values.csv`](tables/raw_values.csv) | 216 | 9 | 21,153 B |
