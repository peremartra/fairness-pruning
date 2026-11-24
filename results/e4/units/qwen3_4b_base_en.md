# E4 Unit Dossier — `qwen3_4b_base_en` (Qwen3-4B-Base, EN)

[← Back to E4 Overview](../E4_RESULTS.md) · Sets: **6** real, **1** full-battery random, **9** WikiText-only random · Checks: 46 (0 problems).

## 1. Raw Benchmark Scores

Unmodified benchmark values (`word_perplexity` for WikiText; `%` accuracy for ARC-Challenge, HellaSwag, HellaSwag-es, and micro MMLU) for every evaluated set.

| Set | WikiText (ppl) | ARC-C (%) | HellaSwag (%) | HellaSwag-es (%) | MMLU micro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 10.374 | 51.02% | 73.69% | 62.12% | 73.09% |
| `real_gender20` | 11.031 | 51.88% | 73.04% | 62.14% | 72.72% |
| `real_gender5` | 10.760 | 51.62% | 73.32% | 62.23% | - |
| `real_race20` | 10.765 | 52.22% | 73.09% | 62.07% | 72.84% |
| `real_race5` | 10.570 | 52.05% | 73.65% | 62.01% | - |
| `real_religion20` | 11.150 | 51.19% | 72.96% | 62.14% | 72.77% |
| `real_religion5` | 10.579 | 51.88% | 73.41% | 62.26% | - |
| `rand_gender20_seed0` | 10.394 | 51.54% | 73.72% | 62.15% | 73.10% |
| `rand_gender20_seed1` | 10.380 | - | - | - | - |
| `rand_gender20_seed2` | 10.418 | - | - | - | - |
| `rand_gender20_seed3` | 10.382 | - | - | - | - |
| `rand_gender20_seed4` | 10.390 | - | - | - | - |
| `rand_religion20_seed0` | 10.375 | - | - | - | - |
| `rand_religion20_seed1` | 10.376 | - | - | - | - |
| `rand_religion20_seed2` | 10.376 | - | - | - | - |
| `rand_religion20_seed3` | 10.379 | - | - | - | - |
| `rand_religion20_seed4` | 10.385 | - | - | - | - |

## 2. Paired Differences vs. Baseline (`B = 10,000`, `±1 pp` Margin)

Item-paired deltas (`set − baseline`) with 95% percentile bootstrap confidence intervals and equivalence verdicts for all real sets and full-battery random controls.

| Task | Set | Kind | Set Value | Baseline | Delta | 95% CI | Verdict |
| :--- | :--- | :---: | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` | real | 11.031 | 10.374 | +6.333% | [+5.922, +6.737] | - |
| wikitext | `real_gender5` | real | 10.760 | 10.374 | +3.725% | [+3.482, +3.970] | - |
| wikitext | `real_race20` | real | 10.765 | 10.374 | +3.774% | [+3.494, +4.058] | - |
| wikitext | `real_race5` | real | 10.570 | 10.374 | +1.892% | [+1.693, +2.097] | - |
| wikitext | `real_religion20` | real | 11.150 | 10.374 | +7.480% | [+7.052, +7.943] | - |
| wikitext | `real_religion5` | real | 10.579 | 10.374 | +1.980% | [+1.788, +2.172] | - |
| wikitext | `rand_gender20_seed0` | random | 10.394 | 10.374 | +0.192% | [+0.059, +0.365] | - |
| wikitext | `rand_gender20_seed1` | random | 10.380 | 10.374 | +0.061% | [+0.001, +0.145] | - |
| wikitext | `rand_gender20_seed2` | random | 10.418 | 10.374 | +0.428% | [+0.245, +0.629] | - |
| wikitext | `rand_gender20_seed3` | random | 10.382 | 10.374 | +0.080% | [+0.000, +0.176] | - |
| wikitext | `rand_gender20_seed4` | random | 10.390 | 10.374 | +0.156% | [+0.040, +0.305] | - |
| wikitext | `rand_religion20_seed0` | random | 10.375 | 10.374 | +0.018% | [-0.023, +0.082] | - |
| wikitext | `rand_religion20_seed1` | random | 10.376 | 10.374 | +0.028% | [-0.004, +0.093] | - |
| wikitext | `rand_religion20_seed2` | random | 10.376 | 10.374 | +0.020% | [-0.019, +0.084] | - |
| wikitext | `rand_religion20_seed3` | random | 10.379 | 10.374 | +0.055% | [-0.002, +0.140] | - |
| wikitext | `rand_religion20_seed4` | random | 10.385 | 10.374 | +0.113% | [+0.007, +0.252] | - |
| arc_challenge | `real_gender20` | real | 51.88% | 51.02% | +0.853 pp | [-0.341, +2.133] | inconcl |
| arc_challenge | `real_gender5` | real | 51.62% | 51.02% | +0.597 pp | [-0.427, +1.623] | inconcl |
| arc_challenge | `real_race20` | real | 52.22% | 51.02% | +1.195 pp | [+0.000, +2.389] | inconcl |
| arc_challenge | `real_race5` | real | 52.05% | 51.02% | +1.024 pp | [+0.000, +2.048] | inconcl |
| arc_challenge | `real_religion20` | real | 51.19% | 51.02% | +0.171 pp | [-1.109, +1.451] | inconcl |
| arc_challenge | `real_religion5` | real | 51.88% | 51.02% | +0.853 pp | [-0.085, +1.792] | inconcl |
| arc_challenge | `rand_gender20_seed0` | random | 51.54% | 51.02% | +0.512 pp | [-0.085, +1.195] | inconcl |
| hellaswag | `real_gender20` | real | 73.04% | 73.69% | -0.647 pp | [-1.046, -0.259] | inconcl |
| hellaswag | `real_gender5` | real | 73.32% | 73.69% | -0.368 pp | [-0.677, -0.050] | equiv |
| hellaswag | `real_race20` | real | 73.09% | 73.69% | -0.597 pp | [-0.946, -0.259] | equiv |
| hellaswag | `real_race5` | real | 73.65% | 73.69% | -0.040 pp | [-0.279, +0.189] | equiv |
| hellaswag | `real_religion20` | real | 72.96% | 73.69% | -0.727 pp | [-1.145, -0.309] | inconcl |
| hellaswag | `real_religion5` | real | 73.41% | 73.69% | -0.279 pp | [-0.548, -0.010] | equiv |
| hellaswag | `rand_gender20_seed0` | random | 73.72% | 73.69% | +0.030 pp | [-0.129, +0.189] | equiv |
| hellaswag_es | `real_gender20` | real | 62.14% | 62.12% | +0.021 pp | [-0.363, +0.395] | equiv |
| hellaswag_es | `real_gender5` | real | 62.23% | 62.12% | +0.107 pp | [-0.299, +0.512] | equiv |
| hellaswag_es | `real_race20` | real | 62.07% | 62.12% | -0.053 pp | [-0.469, +0.363] | equiv |
| hellaswag_es | `real_race5` | real | 62.01% | 62.12% | -0.107 pp | [-0.448, +0.235] | equiv |
| hellaswag_es | `real_religion20` | real | 62.14% | 62.12% | +0.021 pp | [-0.373, +0.406] | equiv |
| hellaswag_es | `real_religion5` | real | 62.26% | 62.12% | +0.139 pp | [-0.213, +0.491] | equiv |
| hellaswag_es | `rand_gender20_seed0` | random | 62.15% | 62.12% | +0.032 pp | [-0.160, +0.224] | equiv |
| mmlu | `real_gender20` | real | 72.72% | 73.09% | -0.370 pp | [-0.641, -0.100] | equiv |
| mmlu | `real_race20` | real | 72.84% | 73.09% | -0.249 pp | [-0.470, -0.028] | equiv |
| mmlu | `real_religion20` | real | 72.77% | 73.09% | -0.320 pp | [-0.563, -0.078] | equiv |
| mmlu | `rand_gender20_seed0` | random | 73.10% | 73.09% | +0.014 pp | [-0.135, +0.164] | equiv |

## 3. Real vs. Layer-Matched Random Contrasts

Paired differences (`real − random`) across all tasks and random seeds available for this unit.

| Task | Comparison | Real Value | Random Value | Delta | 95% CI | Verdict |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` vs. `rand_gender20_seed0` | 11.031 | 10.394 | +6.129% | [+5.729, +6.519] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed1` | 11.031 | 10.380 | +6.268% | [+5.882, +6.645] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed2` | 11.031 | 10.418 | +5.880% | [+5.469, +6.292] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed3` | 11.031 | 10.382 | +6.249% | [+5.835, +6.643] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed4` | 11.031 | 10.390 | +6.168% | [+5.785, +6.547] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed0` | 11.150 | 10.375 | +7.461% | [+7.019, +7.936] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed1` | 11.150 | 10.376 | +7.450% | [+7.030, +7.888] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed2` | 11.150 | 10.376 | +7.459% | [+7.040, +7.897] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed3` | 11.150 | 10.379 | +7.421% | [+6.992, +7.869] | - |
| wikitext | `real_religion20` vs. `rand_religion20_seed4` | 11.150 | 10.385 | +7.359% | [+6.930, +7.808] | - |
| arc_challenge | `real_gender20` vs. `rand_gender20_seed0` | 51.88% | 51.54% | +0.341 pp | [-0.853, +1.536] | inconcl |
| hellaswag | `real_gender20` vs. `rand_gender20_seed0` | 73.04% | 73.72% | -0.677 pp | [-1.085, -0.279] | inconcl |
| hellaswag_es | `real_gender20` vs. `rand_gender20_seed0` | 62.14% | 62.15% | -0.011 pp | [-0.395, +0.363] | equiv |
| mmlu | `real_gender20` vs. `rand_gender20_seed0` | 72.72% | 73.10% | -0.385 pp | [-0.655, -0.107] | equiv |

## 4. MMLU Micro vs. Macro Reconciliation

Item-weighted micro accuracy (`14,042` items) compared with unweighted 57-subject macro accuracy and `_cap.json` values.

| Set | Micro (14,042 items) | Macro (57 subjects) | Macro − Micro | `_cap.json` Value |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 73.09% | 75.59% | +2.50 pp | 75.52% |
| `rand_gender20_seed0` | 73.10% | 75.57% | +2.47 pp | - |
| `real_gender20` | 72.72% | 75.28% | +2.56 pp | 75.20% |
| `real_race20` | 72.84% | 75.29% | +2.45 pp | 75.21% |
| `real_religion20` | 72.77% | 75.41% | +2.64 pp | 75.33% |

## 5. Hardware and Batch-Size Provenance

GPU hardware, batch size, and `lm-eval` version recorded per set and task from `tables/provenance.csv`.

<details>
<summary>Expand per-set provenance table</summary>

| Set | Task | GPU | Batch | lm-eval | Source Note |
| :--- | :--- | :--- | ---: | :---: | :--- |
| `baseline` | arc_challenge | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `baseline` | hellaswag | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `baseline` | hellaswag_es | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `baseline` | mmlu | A100-SXM4-40GB | 4 | 0.4.9.2 | E4_handoff.md §3 |
| `baseline` | wikitext | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `rand_gender20_seed0` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | A100 rerun (2026-11-24, checkpoints_e4_a100_rerun, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-12 L4 run |
| `rand_gender20_seed0` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | A100 rerun (2026-11-24, checkpoints_e4_a100_rerun, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-12 L4 run |
| `rand_gender20_seed0` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | A100 rerun (2026-11-24, checkpoints_e4_a100_rerun, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-12 L4 run |
| `rand_gender20_seed0` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | A100 rerun (2026-11-24, checkpoints_e4_a100_rerun, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-12 L4 run |
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
| `real_gender20` | arc_challenge | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `real_gender20` | hellaswag | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `real_gender20` | hellaswag_es | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
| `real_gender20` | mmlu | A100-SXM4-40GB | 4 | 0.4.9.2 | E4_handoff.md §3 |
| `real_gender20` | wikitext | A100-SXM4-40GB | 16 | 0.4.9.2 | E4_handoff.md §3; batch from per-set summary (batch_size 16, wikitext_batch_size 1), checked 2026-11-15 |
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
