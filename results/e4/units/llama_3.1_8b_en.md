# E4 Unit Dossier — `llama_3.1_8b_en` (Llama-3.1-8B, EN)

[← Back to E4 Overview](../E4_RESULTS.md) · Sets: **6** real, **1** full-battery random, **14** WikiText-only random · Checks: 51 (0 problems).

## 1. Raw Benchmark Scores

Unmodified benchmark values (`word_perplexity` for WikiText; `%` accuracy for ARC-Challenge, HellaSwag, HellaSwag-es, and micro MMLU) for every evaluated set.

| Set | WikiText (ppl) | ARC-C (%) | HellaSwag (%) | HellaSwag-es (%) | MMLU micro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 7.544 | 54.61% | 79.30% | 66.88% | 65.28% |
| `real_gender20` | 7.599 | 55.12% | 79.36% | 66.74% | 65.34% |
| `real_gender5` | 7.563 | 54.27% | 79.28% | 66.88% | - |
| `real_race20` | 7.614 | 55.63% | 79.33% | 66.75% | 65.33% |
| `real_race5` | 7.554 | 54.95% | 79.36% | 66.87% | - |
| `real_religion20` | 7.593 | 55.12% | 79.28% | 66.76% | 65.50% |
| `real_religion5` | 7.560 | 55.03% | 79.30% | 66.83% | - |
| `rand_gender20_seed0` | 7.550 | 54.86% | 79.35% | 66.74% | 65.54% |
| `rand_gender20_seed1` | 7.555 | - | - | - | - |
| `rand_gender20_seed2` | 7.554 | - | - | - | - |
| `rand_gender20_seed3` | 7.544 | - | - | - | - |
| `rand_gender20_seed4` | 7.549 | - | - | - | - |
| `rand_religion20_seed0` | 7.562 | - | - | - | - |
| `rand_religion20_seed1` | 7.553 | - | - | - | - |
| `rand_religion20_seed2` | 7.558 | - | - | - | - |
| `rand_religion20_seed3` | 7.559 | - | - | - | - |
| `rand_religion20_seed4` | 7.545 | - | - | - | - |
| `rand_religion20_seed5` | 7.548 | - | - | - | - |
| `rand_religion20_seed6` | 7.549 | - | - | - | - |
| `rand_religion20_seed7` | 7.554 | - | - | - | - |
| `rand_religion20_seed8` | 7.554 | - | - | - | - |
| `rand_religion20_seed9` | 7.552 | - | - | - | - |

## 2. Paired Differences vs. Baseline (`B = 10,000`, `±1 pp` Margin)

Item-paired deltas (`set − baseline`) with 95% percentile bootstrap confidence intervals and equivalence verdicts for all real sets and full-battery random controls.

| Task | Set | Kind | Set Value | Baseline | Delta | 95% CI | Verdict |
| :--- | :--- | :---: | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` | real | 7.599 | 7.544 | +0.722% | [+0.497, +0.941] | - |
| wikitext | `real_gender5` | real | 7.563 | 7.544 | +0.249% | [+0.085, +0.431] | - |
| wikitext | `real_race20` | real | 7.614 | 7.544 | +0.924% | [+0.726, +1.105] | - |
| wikitext | `real_race5` | real | 7.554 | 7.544 | +0.133% | [+0.036, +0.265] | - |
| wikitext | `real_religion20` | real | 7.593 | 7.544 | +0.645% | [+0.449, +0.845] | - |
| wikitext | `real_religion5` | real | 7.560 | 7.544 | +0.214% | [+0.075, +0.378] | - |
| wikitext | `rand_gender20_seed0` | random | 7.550 | 7.544 | +0.073% | [+0.000, +0.194] | - |
| wikitext | `rand_gender20_seed1` | random | 7.555 | 7.544 | +0.146% | [+0.015, +0.299] | - |
| wikitext | `rand_gender20_seed2` | random | 7.554 | 7.544 | +0.129% | [-0.004, +0.285] | - |
| wikitext | `rand_gender20_seed3` | random | 7.544 | 7.544 | +0.000% | [-0.020, +0.020] | - |
| wikitext | `rand_gender20_seed4` | random | 7.549 | 7.544 | +0.066% | [+0.000, +0.183] | - |
| wikitext | `rand_religion20_seed0` | random | 7.562 | 7.544 | +0.236% | [+0.092, +0.404] | - |
| wikitext | `rand_religion20_seed1` | random | 7.553 | 7.544 | +0.116% | [+0.011, +0.248] | - |
| wikitext | `rand_religion20_seed2` | random | 7.558 | 7.544 | +0.179% | [+0.032, +0.353] | - |
| wikitext | `rand_religion20_seed3` | random | 7.559 | 7.544 | +0.192% | [-0.054, +0.559] | - |
| wikitext | `rand_religion20_seed4` | random | 7.545 | 7.544 | +0.013% | [-0.013, +0.042] | - |
| wikitext | `rand_religion20_seed5` | random | 7.548 | 7.544 | +0.056% | [-0.057, +0.197] | - |
| wikitext | `rand_religion20_seed6` | random | 7.549 | 7.544 | +0.070% | [-0.006, +0.192] | - |
| wikitext | `rand_religion20_seed7` | random | 7.554 | 7.544 | +0.133% | [+0.014, +0.286] | - |
| wikitext | `rand_religion20_seed8` | random | 7.554 | 7.544 | +0.134% | [+0.016, +0.287] | - |
| wikitext | `rand_religion20_seed9` | random | 7.552 | 7.544 | +0.099% | [+0.000, +0.227] | - |
| arc_challenge | `real_gender20` | real | 55.12% | 54.61% | +0.512 pp | [-0.256, +1.280] | inconcl |
| arc_challenge | `real_gender5` | real | 54.27% | 54.61% | -0.341 pp | [-0.853, +0.171] | equiv |
| arc_challenge | `real_race20` | real | 55.63% | 54.61% | +1.024 pp | [+0.256, +1.877] | inconcl |
| arc_challenge | `real_race5` | real | 54.95% | 54.61% | +0.341 pp | [-0.171, +0.853] | equiv |
| arc_challenge | `real_religion20` | real | 55.12% | 54.61% | +0.512 pp | [-0.256, +1.280] | inconcl |
| arc_challenge | `real_religion5` | real | 55.03% | 54.61% | +0.427 pp | [-0.171, +1.024] | inconcl |
| arc_challenge | `rand_gender20_seed0` | random | 54.86% | 54.61% | +0.256 pp | [-0.256, +0.853] | equiv |
| hellaswag | `real_gender20` | real | 79.36% | 79.30% | +0.060 pp | [-0.169, +0.299] | equiv |
| hellaswag | `real_gender5` | real | 79.28% | 79.30% | -0.020 pp | [-0.179, +0.139] | equiv |
| hellaswag | `real_race20` | real | 79.33% | 79.30% | +0.030 pp | [-0.179, +0.239] | equiv |
| hellaswag | `real_race5` | real | 79.36% | 79.30% | +0.060 pp | [-0.080, +0.199] | equiv |
| hellaswag | `real_religion20` | real | 79.28% | 79.30% | -0.020 pp | [-0.229, +0.189] | equiv |
| hellaswag | `real_religion5` | real | 79.30% | 79.30% | +0.000 pp | [-0.139, +0.139] | equiv |
| hellaswag | `rand_gender20_seed0` | random | 79.35% | 79.30% | +0.050 pp | [-0.080, +0.179] | equiv |
| hellaswag_es | `real_gender20` | real | 66.74% | 66.88% | -0.139 pp | [-0.395, +0.107] | equiv |
| hellaswag_es | `real_gender5` | real | 66.88% | 66.88% | +0.000 pp | [-0.192, +0.181] | equiv |
| hellaswag_es | `real_race20` | real | 66.75% | 66.88% | -0.128 pp | [-0.384, +0.128] | equiv |
| hellaswag_es | `real_race5` | real | 66.87% | 66.88% | -0.011 pp | [-0.181, +0.160] | equiv |
| hellaswag_es | `real_religion20` | real | 66.76% | 66.88% | -0.117 pp | [-0.373, +0.128] | equiv |
| hellaswag_es | `real_religion5` | real | 66.83% | 66.88% | -0.043 pp | [-0.224, +0.139] | equiv |
| hellaswag_es | `rand_gender20_seed0` | random | 66.74% | 66.88% | -0.139 pp | [-0.309, +0.032] | equiv |
| mmlu | `real_gender20` | real | 65.34% | 65.28% | +0.057 pp | [-0.164, +0.278] | equiv |
| mmlu | `real_race20` | real | 65.33% | 65.28% | +0.043 pp | [-0.292, +0.385] | equiv |
| mmlu | `real_religion20` | real | 65.50% | 65.28% | +0.221 pp | [-0.071, +0.506] | equiv |
| mmlu | `rand_gender20_seed0` | random | 65.54% | 65.28% | +0.256 pp | [+0.036, +0.477] | equiv |

## 3. Real vs. Layer-Matched Random Contrasts

Paired differences (`real − random`) across all tasks and random seeds available for this unit.

| Task | Comparison | Real Value | Random Value | Delta | 95% CI | Verdict |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` vs. `rand_gender20_seed0` | 7.599 | 7.550 | +0.649% | [+0.422, +0.873] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed1` | 7.599 | 7.555 | +0.576% | [+0.336, +0.819] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed2` | 7.599 | 7.554 | +0.592% | [+0.372, +0.826] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed3` | 7.599 | 7.544 | +0.722% | [+0.497, +0.943] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed4` | 7.599 | 7.549 | +0.656% | [+0.429, +0.885] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed0` | 7.593 | 7.562 | +0.409% | [+0.229, +0.608] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed1` | 7.593 | 7.553 | +0.528% | [+0.340, +0.734] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed2` | 7.593 | 7.558 | +0.465% | [+0.265, +0.684] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed3` | 7.593 | 7.559 | +0.452% | [+0.007, +0.797] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed4` | 7.593 | 7.545 | +0.632% | [+0.436, +0.829] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed5` | 7.593 | 7.548 | +0.588% | [+0.398, +0.787] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed6` | 7.593 | 7.549 | +0.575% | [+0.383, +0.775] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed7` | 7.593 | 7.554 | +0.512% | [+0.332, +0.713] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed8` | 7.593 | 7.554 | +0.510% | [+0.329, +0.712] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed9` | 7.593 | 7.552 | +0.545% | [+0.351, +0.754] | - |
| arc_challenge | `real_gender20` vs. `rand_gender20_seed0` | 55.12% | 54.86% | +0.256 pp | [-0.512, +1.024] | inconcl |
| hellaswag | `real_gender20` vs. `rand_gender20_seed0` | 79.36% | 79.35% | +0.010 pp | [-0.229, +0.249] | equiv |
| hellaswag_es | `real_gender20` vs. `rand_gender20_seed0` | 66.74% | 66.74% | +0.000 pp | [-0.267, +0.267] | equiv |
| mmlu | `real_gender20` vs. `rand_gender20_seed0` | 65.34% | 65.54% | -0.199 pp | [-0.456, +0.071] | equiv |

## 4. MMLU Micro vs. Macro Reconciliation

Item-weighted micro accuracy (`14,042` items) compared with unweighted 57-subject macro accuracy and `_cap.json` values.

| Set | Micro (14,042 items) | Macro (57 subjects) | Macro − Micro | `_cap.json` Value |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 65.28% | 66.28% | +0.99 pp | 66.26% |
| `rand_gender20_seed0` | 65.54% | 66.69% | +1.15 pp | 66.67% |
| `real_gender20` | 65.34% | 66.45% | +1.11 pp | 66.42% |
| `real_race20` | 65.33% | 66.39% | +1.06 pp | 66.37% |
| `real_religion20` | 65.50% | 66.55% | +1.05 pp | 66.54% |

## 5. Hardware and Batch-Size Provenance

GPU hardware, batch size, and `lm-eval` version recorded per set and task from `tables/provenance.csv`.

<details>
<summary>Expand per-set provenance table</summary>

| Set | Task | GPU | Batch | lm-eval | Source Note |
| :--- | :--- | :--- | ---: | :---: | :--- |
| `baseline` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `baseline` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `baseline` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `baseline` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | E4_handoff.md §3 (mmlu_batch_size recorded in the summary) |
| `baseline` | wikitext | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | arc_challenge | L4 | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | hellaswag | L4 | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | hellaswag_es | L4 | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | mmlu | A100-SXM4-40GB | 4 | 0.4.9.2 | E4_handoff.md §4 (repeated on A100 after OOM on L4) |
| `rand_gender20_seed0` | wikitext | A100-SXM4-40GB | 1 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed1` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_gender20_seed2` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_gender20_seed3` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_gender20_seed4` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed0` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed1` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed2` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed3` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed4` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed5` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed6` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed7` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed8` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `rand_religion20_seed9` | wikitext | unknown | unknown | 0.4.9.2 | - |
| `real_gender20` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `real_gender20` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `real_gender20` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `real_gender20` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | E4_handoff.md §3 (mmlu_batch_size recorded in the summary) |
| `real_gender20` | wikitext | A100-SXM4-80GB | 16 | 0.4.9.2 | E4_handoff.md §3 (env.gpu of the summary); batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
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
