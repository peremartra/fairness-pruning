# E4 Results — General-Capability Evaluation After Neuron Zeroing

Paired item-level bootstrap (`B = 10,000`, `seed = 0`, equivalence margin `±1.0 pp`, chance exclusion `< 35%`) across five English base models and five benchmarks (`inputs.json`, built `2026-11-24T18:02:53+00:00`; refreshed `20` A100 rerun logs (`llama_3.2_3b_en`, `llama_3.1_8b_en`, `salamandra_7b_en`, `qwen3_4b_base_en`)).

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
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | Llama-3.1-8B | 6 | 1 | 14 | 5, 20 | Included | 51 (0 problems) | A100-SXM4-80GB; unknown | A100-SXM4-40GB; A100-SXM4-80GB; unknown |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | Qwen3-4B-Base | 6 | 1 | 9 | 5, 20 | Included | 46 (0 problems) | A100-SXM4-40GB; unknown | A100-SXM4-40GB; A100-SXM4-80GB; unknown |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | Salamandra-7B | 6 | 1 | 9 | 5, 20 | Included | 46 (0 problems) | A100-SXM4-40GB; unknown | A100-SXM4-40GB; A100-SXM4-80GB; unknown |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | Llama-3.2-3B | 6 | 1 | 4 | 5, 20 | Included | 41 (0 problems) | A100-SXM4-80GB | A100-SXM4-80GB |
| [`llama_3.2_1b_en`](units/llama_3.2_1b_en.md) | Llama-3.2-1B | 6 | 0 | 0 | 5, 20 | Excluded (<35%) | 32 (0 problems) | A100-SXM4-40GB; unknown | - |

*Legend — **Real Sets**: selected top-K sets (`Gender`/`Race`/`Religion` × `K = 5, 20`); **Random (Full)**: layer-matched random control evaluated on all 5 benchmarks (`rand_gender20_seed0`); **Random (WikiText-only)**: additional random seeds evaluated on WikiText only; **MMLU Status**: `Excluded (<35%)` when unpruned baseline accuracy is below the 35% near-chance threshold; **Checks**: per-item extraction integrity checks (`0 problems` = 100% agreement with `lm-eval` aggregates and baseline item hashes).*

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

*Legend — **`baseline` *(raw)***: unpruned model score (`word_perplexity` for WikiText, lower is better; `%` accuracy for the four QA benchmarks); **WikiText (Δ%)**: relative percentage change in word perplexity vs. baseline (`+` = higher/worse perplexity); **ARC-C / HellaSwag / HellaSwag-es / MMLU (pp)**: percentage-point change (`real − baseline`) with 95% paired item-level bootstrap CI (`B = 10,000`, `seed = 0`); **Verdicts (`±1.0 pp` equivalence margin)**: `equiv` = entire 95% CI within `[-1, +1] pp`, `inconcl` = CI crosses `±1 pp`, `WORSE` = entire CI below `-1 pp`, `excl (<35%)` = baseline near chance.*

### 1.3 Real vs. Layer-Matched Random Control (`Gender·20`, seed 0)

Direct comparison of `real_gender20` against `rand_gender20_seed0` (same layer profile, full 5-task battery) across the four models with a full random control, with explicit GPU hardware per side.

| Unit | Task | Real − Baseline | Random(seed0) − Baseline | Real − Random(seed0) | GPUs (Real / Rand) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | wikitext | +0.72% [+0.50, +0.94] | +0.07% [+0.00, +0.19] | +0.65% [+0.42, +0.87] | A100-SXM4-80GB / A100-SXM4-40GB |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | arc_challenge | +0.51 [-0.26, +1.28] (inconcl) | +0.26 [-0.17, +0.68] (equiv) | +0.26 [-0.51, +1.02] (inconcl) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag | +0.06 [-0.17, +0.30] (equiv) | +0.09 [+0.00, +0.19] (equiv) | -0.03 [-0.27, +0.21] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag_es | -0.14 [-0.39, +0.11] (equiv) | -0.12 [-0.27, +0.02] (equiv) | -0.02 [-0.28, +0.23] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | mmlu | +0.06 [-0.16, +0.28] (equiv) | +0.26 [+0.04, +0.48] (equiv) | -0.20 [-0.46, +0.07] (equiv) | A100-SXM4-80GB / A100-SXM4-40GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | wikitext | +6.33% [+5.92, +6.74] | +0.19% [+0.06, +0.36] | +6.13% [+5.73, +6.52] | A100-SXM4-40GB / A100-SXM4-40GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | arc_challenge | +0.85 [-0.34, +2.13] (inconcl) | +0.51 [-0.09, +1.19] (inconcl) | +0.34 [-0.85, +1.54] (inconcl) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag | -0.65 [-1.05, -0.26] (inconcl) | +0.03 [-0.13, +0.19] (equiv) | -0.68 [-1.09, -0.28] (inconcl) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag_es | +0.02 [-0.36, +0.39] (equiv) | +0.03 [-0.16, +0.22] (equiv) | -0.01 [-0.39, +0.36] (equiv) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | mmlu | -0.37 [-0.64, -0.10] (equiv) | +0.01 [-0.14, +0.16] (equiv) | -0.38 [-0.66, -0.11] (equiv) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | wikitext | +0.41% [+0.25, +0.57] | +0.01% [-0.00, +0.03] | +0.40% [+0.24, +0.57] | A100-SXM4-40GB / A100-SXM4-80GB |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | arc_challenge | -0.26 [-0.94, +0.43] (equiv) | +0.00 [-0.43, +0.43] (equiv) | -0.26 [-0.94, +0.43] (equiv) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag | -0.08 [-0.29, +0.13] (equiv) | +0.01 [-0.14, +0.16] (equiv) | -0.09 [-0.30, +0.12] (equiv) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag_es | -0.11 [-0.31, +0.10] (equiv) | +0.07 [-0.07, +0.22] (equiv) | -0.18 [-0.39, +0.03] (equiv) | A100-SXM4-40GB / A100-SXM4-80GB |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | mmlu | -0.47 [-0.84, -0.11] (equiv) | -0.09 [-0.36, +0.18] (equiv) | -0.38 [-0.73, -0.02] (equiv) | A100-SXM4-40GB / A100-SXM4-40GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | wikitext | +0.77% [+0.52, +1.03] | +0.04% [+0.00, +0.11] | +0.73% [+0.48, +0.99] | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | arc_challenge | +0.00 [-0.85, +0.77] (equiv) | +0.00 [-0.60, +0.60] (equiv) | +0.00 [-0.85, +0.85] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag | +0.04 [-0.25, +0.33] (equiv) | +0.10 [-0.03, +0.24] (equiv) | -0.06 [-0.34, +0.22] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag_es | -0.05 [-0.29, +0.18] (equiv) | -0.14 [-0.31, +0.03] (equiv) | +0.09 [-0.15, +0.32] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | mmlu | -0.14 [-0.39, +0.11] (equiv) | -0.08 [-0.27, +0.11] (equiv) | -0.06 [-0.31, +0.19] (equiv) | A100-SXM4-80GB / A100-SXM4-80GB |

*Legend — **Real − Baseline / Random(seed0) − Baseline / Real − Random(seed0)**: paired item-level differences with 95% bootstrap CI (`Δ%` relative perplexity for WikiText; percentage points `pp` for accuracy tasks) and `±1.0 pp` equivalence verdicts (`equiv`, `inconcl`, `WORSE`); **GPUs (Real / Rand)**: verified A100 hardware for each side.*

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

*Legend — Cells show `K=5 / K=20` paired deltas vs. baseline (`Δ%` relative word perplexity for WikiText; percentage points `pp` for ARC-C, HellaSwag, and HellaSwag-es; MMLU is evaluated at `K = 20` only).*

</details>

---

## 2. Unit Dossiers (Level 2)

Each unit dossier follows the same layout: raw scores across all sets, full baseline deltas with 95% bootstrap CIs, real-vs-random contrasts across all seeds, MMLU micro-vs-macro reconciliation, and per-task GPU/batch provenance.

| Dossier | Model | Sets | Baseline/Real GPUs | Random GPUs |
| :--- | :--- | :--- | :--- | :--- |
| [`units/llama_3.1_8b_en.md`](units/llama_3.1_8b_en.md) | Llama-3.1-8B | 6 real / 1 full-rand / 14 wt-rand | A100-SXM4-80GB; unknown | A100-SXM4-40GB; A100-SXM4-80GB; unknown |
| [`units/qwen3_4b_base_en.md`](units/qwen3_4b_base_en.md) | Qwen3-4B-Base | 6 real / 1 full-rand / 9 wt-rand | A100-SXM4-40GB; unknown | A100-SXM4-40GB; A100-SXM4-80GB; unknown |
| [`units/salamandra_7b_en.md`](units/salamandra_7b_en.md) | Salamandra-7B | 6 real / 1 full-rand / 9 wt-rand | A100-SXM4-40GB; unknown | A100-SXM4-40GB; A100-SXM4-80GB; unknown |
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

*Legend — **Base PPL**: unpruned WikiText word perplexity (`62` documents, lower is better); **Real Δ% [95% CI]**: relative perplexity change vs. baseline (`+` = higher perplexity); **Random Δ%**: min–max and mean relative change across all layer-matched random seeds; **Real − Random Δ%**: paired real-minus-random difference (mean `[min, max]` across seeds). Differences `< ~0.1%` lie below the `bf16` document log-likelihood quantization floor.*

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
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | `rand_gender20_seed0` | 73.10% | 75.57% | +2.47 pp | - |
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

*Legend — **Micro (14,042 items)**: item-weighted accuracy used throughout E4 and reported by `lm-eval`; **Macro (57 subjects)**: unweighted average across the 57 MMLU subjects; **`_cap.json` Value**: macro value stored in per-set checkpoint summaries (never mix with micro in one table).*

</details>

### 3.3 Hardware Provenance (A100 vs. L4) and Extraction Checks

Per-task GPU hardware across baseline, `real_gender20`, and `rand_gender20_seed0` for the four units with full random controls (all 20/20 cells matched on A100), plus overall extraction integrity across all per-item logs.

| Unit | Task | Baseline GPU | Real Gender·20 GPU | Random Gender·20 (seed0) GPU | Hardware Match |
| :--- | :--- | :--- | :--- | :--- | :---: |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | wikitext | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-40GB | Matched (A100) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | arc_challenge | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | hellaswag_es | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.1_8b_en`](units/llama_3.1_8b_en.md) | mmlu | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-40GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | wikitext | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-40GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | arc_challenge | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | hellaswag_es | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`qwen3_4b_base_en`](units/qwen3_4b_base_en.md) | mmlu | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | wikitext | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | arc_challenge | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | hellaswag_es | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-80GB | Matched (A100) |
| [`salamandra_7b_en`](units/salamandra_7b_en.md) | mmlu | A100-SXM4-40GB | A100-SXM4-40GB | A100-SXM4-40GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | wikitext | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | arc_challenge | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | hellaswag_es | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |
| [`llama_3.2_3b_en`](units/llama_3.2_3b_en.md) | mmlu | A100-SXM4-80GB | A100-SXM4-80GB | A100-SXM4-80GB | Matched (A100) |

*Legend — **Hardware Match**: `Matched (A100)` confirms baseline, real, and random runs all executed on A100 GPUs (zero L4-vs-A100 hardware mixing).*

Extraction verification (`tables/checks.csv`): **216** unit × set × task tables checked, **0** problems detected, 100% item-key and `doc_hash` alignment with baseline.

### 3.4 Source Tables

All numbers in this report and its unit dossiers are rendered directly from the verified CSV tables under `results/e4/tables/` (and mirrored in `results/e4/e4_capabilities_table.csv`).

| CSV Table | Rows | Columns | Size |
| :--- | ---: | ---: | ---: |
| [`tables/checks.csv`](tables/checks.csv) | 216 | 11 | 23,209 B |
| [`tables/deltas.csv`](tables/deltas.csv) | 247 | 19 | 60,104 B |
| [`tables/e4_capabilities_table.csv`](tables/e4_capabilities_table.csv) | 247 | 10 | 37,378 B |
| [`tables/e4_inventory.csv`](tables/e4_inventory.csv) | 5 | 14 | 750 B |
| [`tables/e4_wikitext_random_summary.csv`](tables/e4_wikitext_random_summary.csv) | 7 | 15 | 1,948 B |
| [`tables/mmlu_micro_macro.csv`](tables/mmlu_micro_macro.csv) | 24 | 9 | 3,109 B |
| [`tables/provenance.csv`](tables/provenance.csv) | 216 | 8 | 28,946 B |
| [`tables/raw_values.csv`](tables/raw_values.csv) | 216 | 9 | 21,155 B |
