# E4 Unit Dossier — `salamandra_7b_en` (Salamandra-7B, EN)

[← Back to E4 Overview](../E4_RESULTS.md) · Sets: **6** real, **1** full-battery random, **9** WikiText-only random · Checks: 46 (0 problems).

## 1. Raw Benchmark Scores

Unmodified benchmark values (`word_perplexity` for WikiText; `%` accuracy for ARC-Challenge, HellaSwag, HellaSwag-es, and micro MMLU) for every evaluated set.

| Set | WikiText (ppl) | ARC-C (%) | HellaSwag (%) | HellaSwag-es (%) | MMLU micro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 6.220 | 48.21% | 76.47% | 67.87% | 47.03% |
| `real_gender20` | 6.246 | 47.95% | 76.39% | 67.76% | 46.56% |
| `real_gender5` | 6.224 | 47.95% | 76.38% | 67.91% | - |
| `real_race20` | 6.275 | 47.95% | 76.33% | 67.64% | 45.34% |
| `real_race5` | 6.245 | 48.29% | 76.35% | 67.85% | - |
| `real_religion20` | 6.263 | 48.21% | 76.37% | 67.96% | 46.38% |
| `real_religion5` | 6.225 | 48.38% | 76.38% | 67.92% | - |
| `rand_gender20_seed0` | 6.221 | 48.04% | 76.39% | 67.86% | 46.94% |
| `rand_gender20_seed1` | 6.222 | - | - | - | - |
| `rand_gender20_seed2` | 6.223 | - | - | - | - |
| `rand_gender20_seed3` | 6.234 | - | - | - | - |
| `rand_gender20_seed4` | 6.223 | - | - | - | - |
| `rand_religion20_seed0` | 6.225 | - | - | - | - |
| `rand_religion20_seed1` | 6.223 | - | - | - | - |
| `rand_religion20_seed2` | 6.223 | - | - | - | - |
| `rand_religion20_seed3` | 6.225 | - | - | - | - |
| `rand_religion20_seed4` | 6.221 | - | - | - | - |

## 2. Paired Differences vs. Baseline (`B = 10,000`, `±1 pp` Margin)

Item-paired deltas (`set − baseline`) with 95% percentile bootstrap confidence intervals and equivalence verdicts for all real sets and full-battery random controls.

| Task | Set | Kind | Set Value | Baseline | Delta | 95% CI | Verdict |
| :--- | :--- | :---: | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` | real | 6.246 | 6.220 | +0.410% | [+0.254, +0.573] | - |
| wikitext | `real_gender5` | real | 6.224 | 6.220 | +0.066% | [-0.005, +0.160] | - |
| wikitext | `real_race20` | real | 6.275 | 6.220 | +0.873% | [+0.659, +1.096] | - |
| wikitext | `real_race5` | real | 6.245 | 6.220 | +0.394% | [+0.253, +0.550] | - |
| wikitext | `real_religion20` | real | 6.263 | 6.220 | +0.681% | [+0.520, +0.844] | - |
| wikitext | `real_religion5` | real | 6.225 | 6.220 | +0.085% | [+0.009, +0.187] | - |
| wikitext | `rand_gender20_seed0` | random | 6.221 | 6.220 | +0.007% | [-0.004, +0.025] | - |
| wikitext | `rand_gender20_seed1` | random | 6.222 | 6.220 | +0.032% | [-0.004, +0.081] | - |
| wikitext | `rand_gender20_seed2` | random | 6.223 | 6.220 | +0.039% | [-0.002, +0.109] | - |
| wikitext | `rand_gender20_seed3` | random | 6.234 | 6.220 | +0.219% | [+0.107, +0.355] | - |
| wikitext | `rand_gender20_seed4` | random | 6.223 | 6.220 | +0.036% | [-0.010, +0.107] | - |
| wikitext | `rand_religion20_seed0` | random | 6.225 | 6.220 | +0.082% | [+0.010, +0.177] | - |
| wikitext | `rand_religion20_seed1` | random | 6.223 | 6.220 | +0.040% | [-0.002, +0.111] | - |
| wikitext | `rand_religion20_seed2` | random | 6.223 | 6.220 | +0.049% | [+0.012, +0.100] | - |
| wikitext | `rand_religion20_seed3` | random | 6.225 | 6.220 | +0.078% | [+0.013, +0.159] | - |
| wikitext | `rand_religion20_seed4` | random | 6.221 | 6.220 | +0.012% | [-0.004, +0.045] | - |
| arc_challenge | `real_gender20` | real | 47.95% | 48.21% | -0.256 pp | [-0.939, +0.427] | equiv |
| arc_challenge | `real_gender5` | real | 47.95% | 48.21% | -0.256 pp | [-0.853, +0.341] | equiv |
| arc_challenge | `real_race20` | real | 47.95% | 48.21% | -0.256 pp | [-1.109, +0.512] | inconcl |
| arc_challenge | `real_race5` | real | 48.29% | 48.21% | +0.085 pp | [-0.512, +0.683] | equiv |
| arc_challenge | `real_religion20` | real | 48.21% | 48.21% | +0.000 pp | [-0.597, +0.597] | equiv |
| arc_challenge | `real_religion5` | real | 48.38% | 48.21% | +0.171 pp | [-0.256, +0.597] | equiv |
| arc_challenge | `rand_gender20_seed0` | random | 48.04% | 48.21% | -0.171 pp | [-0.768, +0.427] | equiv |
| hellaswag | `real_gender20` | real | 76.39% | 76.47% | -0.080 pp | [-0.289, +0.129] | equiv |
| hellaswag | `real_gender5` | real | 76.38% | 76.47% | -0.090 pp | [-0.239, +0.060] | equiv |
| hellaswag | `real_race20` | real | 76.33% | 76.47% | -0.139 pp | [-0.349, +0.070] | equiv |
| hellaswag | `real_race5` | real | 76.35% | 76.47% | -0.119 pp | [-0.319, +0.080] | equiv |
| hellaswag | `real_religion20` | real | 76.37% | 76.47% | -0.100 pp | [-0.309, +0.110] | equiv |
| hellaswag | `real_religion5` | real | 76.38% | 76.47% | -0.090 pp | [-0.249, +0.070] | equiv |
| hellaswag | `rand_gender20_seed0` | random | 76.39% | 76.47% | -0.080 pp | [-0.229, +0.070] | equiv |
| hellaswag_es | `real_gender20` | real | 67.76% | 67.87% | -0.107 pp | [-0.309, +0.096] | equiv |
| hellaswag_es | `real_gender5` | real | 67.91% | 67.87% | +0.043 pp | [-0.085, +0.171] | equiv |
| hellaswag_es | `real_race20` | real | 67.64% | 67.87% | -0.224 pp | [-0.469, +0.021] | equiv |
| hellaswag_es | `real_race5` | real | 67.85% | 67.87% | -0.021 pp | [-0.224, +0.171] | equiv |
| hellaswag_es | `real_religion20` | real | 67.96% | 67.87% | +0.096 pp | [-0.149, +0.341] | equiv |
| hellaswag_es | `real_religion5` | real | 67.92% | 67.87% | +0.053 pp | [-0.107, +0.213] | equiv |
| hellaswag_es | `rand_gender20_seed0` | random | 67.86% | 67.87% | -0.011 pp | [-0.181, +0.149] | equiv |
| mmlu | `real_gender20` | real | 46.56% | 47.03% | -0.470 pp | [-0.840, -0.107] | equiv |
| mmlu | `real_race20` | real | 45.34% | 47.03% | -1.695 pp | [-2.258, -1.132] | WORSE |
| mmlu | `real_religion20` | real | 46.38% | 47.03% | -0.655 pp | [-0.997, -0.328] | equiv |
| mmlu | `rand_gender20_seed0` | random | 46.94% | 47.03% | -0.093 pp | [-0.356, +0.178] | equiv |

## 3. Real vs. Layer-Matched Random Contrasts

Paired differences (`real − random`) across all tasks and random seeds available for this unit.

| Task | Comparison | Real Value | Random Value | Delta | 95% CI | Verdict |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` vs. `rand_gender20_seed0` | 6.246 | 6.221 | +0.402% | [+0.244, +0.569] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed1` | 6.246 | 6.222 | +0.377% | [+0.221, +0.542] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed2` | 6.246 | 6.223 | +0.371% | [+0.219, +0.532] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed3` | 6.246 | 6.234 | +0.191% | [+0.031, +0.355] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed4` | 6.246 | 6.223 | +0.373% | [+0.224, +0.533] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed0` | 6.263 | 6.225 | +0.598% | [+0.431, +0.772] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed1` | 6.263 | 6.223 | +0.640% | [+0.476, +0.809] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed2` | 6.263 | 6.223 | +0.631% | [+0.460, +0.802] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed3` | 6.263 | 6.225 | +0.602% | [+0.418, +0.790] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed4` | 6.263 | 6.221 | +0.669% | [+0.507, +0.832] | - |
| arc_challenge | `real_gender20` vs. `rand_gender20_seed0` | 47.95% | 48.04% | -0.085 pp | [-0.939, +0.768] | equiv |
| hellaswag | `real_gender20` vs. `rand_gender20_seed0` | 76.39% | 76.39% | +0.000 pp | [-0.199, +0.209] | equiv |
| hellaswag_es | `real_gender20` vs. `rand_gender20_seed0` | 67.76% | 67.86% | -0.096 pp | [-0.320, +0.128] | equiv |
| mmlu | `real_gender20` vs. `rand_gender20_seed0` | 46.56% | 46.94% | -0.377 pp | [-0.734, -0.021] | equiv |

## 4. MMLU Micro vs. Macro Reconciliation

Item-weighted micro accuracy (`14,042` items) compared with unweighted 57-subject macro accuracy and `_cap.json` values.

| Set | Micro (14,042 items) | Macro (57 subjects) | Macro − Micro | `_cap.json` Value |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 47.03% | 48.47% | +1.44 pp | 48.40% |
| `rand_gender20_seed0` | 46.94% | 48.44% | +1.50 pp | 48.37% |
| `real_gender20` | 46.56% | 47.97% | +1.41 pp | 47.91% |
| `real_race20` | 45.34% | 46.39% | +1.06 pp | 46.35% |
| `real_religion20` | 46.38% | 47.82% | +1.44 pp | 47.76% |

## 5. Hardware and Batch-Size Provenance

GPU hardware, batch size, and `lm-eval` version recorded per set and task from `tables/provenance.csv`.

<details>
<summary>Expand per-set provenance table</summary>

| Set | Task | GPU | Batch | lm-eval | Source Note |
| :--- | :--- | :--- | ---: | :---: | :--- |
| `baseline` | arc_challenge | A100-SXM4-40GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `baseline` | hellaswag | A100-SXM4-40GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `baseline` | hellaswag_es | A100-SXM4-40GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `baseline` | mmlu | A100-SXM4-40GB | 4 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `baseline` | wikitext | A100-SXM4-40GB | 1 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `rand_gender20_seed0` | arc_challenge | L4 | 16 | 0.4.9.2 | orchestrator prompt A (L4 plan); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | hellaswag | L4 | 16 | 0.4.9.2 | orchestrator prompt A (L4 plan); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | hellaswag_es | L4 | 16 | 0.4.9.2 | orchestrator prompt A (L4 plan); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | mmlu | A100-SXM4-40GB | 4 | 0.4.9.2 | Salamandra MMLU retry (agent report) |
| `rand_gender20_seed0` | wikitext | L4 | 1 | 0.4.9.2 | orchestrator prompt A (L4 plan); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed1` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_gender20_seed2` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_gender20_seed3` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_gender20_seed4` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed0` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed1` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed2` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed3` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed4` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `real_gender20` | arc_challenge | A100-SXM4-40GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `real_gender20` | hellaswag | A100-SXM4-40GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `real_gender20` | hellaswag_es | A100-SXM4-40GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `real_gender20` | mmlu | A100-SXM4-40GB | 4 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `real_gender20` | wikitext | A100-SXM4-40GB | 1 | 0.4.9.2 | per-set summary env.gpu and batch sizes (Drive), checked 2026-11-15 |
| `real_gender5` | arc_challenge | unknown | unknown | 0.4.9.2 | - |
| `real_gender5` | hellaswag | unknown | unknown | 0.4.9.2 | - |
| `real_gender5` | hellaswag_es | unknown | unknown | 0.4.9.2 | - |
| `real_gender5` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `real_race20` | arc_challenge | unknown | unknown | 0.4.9.2 | - |
| `real_race20` | hellaswag | unknown | unknown | 0.4.9.2 | - |
| `real_race20` | hellaswag_es | unknown | unknown | 0.4.9.2 | - |
| `real_race20` | mmlu | unknown | unknown | 0.4.9.2 | - |
| `real_race20` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `real_race5` | arc_challenge | unknown | unknown | 0.4.9.2 | - |
| `real_race5` | hellaswag | unknown | unknown | 0.4.9.2 | - |
| `real_race5` | hellaswag_es | unknown | unknown | 0.4.9.2 | - |
| `real_race5` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `real_religion20` | arc_challenge | unknown | unknown | 0.4.9.2 | - |
| `real_religion20` | hellaswag | unknown | unknown | 0.4.9.2 | - |
| `real_religion20` | hellaswag_es | unknown | unknown | 0.4.9.2 | - |
| `real_religion20` | mmlu | unknown | unknown | 0.4.9.2 | - |
| `real_religion20` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `real_religion5` | arc_challenge | unknown | unknown | 0.4.9.2 | - |
| `real_religion5` | hellaswag | unknown | unknown | 0.4.9.2 | - |
| `real_religion5` | hellaswag_es | unknown | unknown | 0.4.9.2 | - |
| `real_religion5` | wikitext | unknown | unknown | 0.4.9.2 | - |

</details>
