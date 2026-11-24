# E4 Unit Dossier — `llama_3.2_3b_en` (Llama-3.2-3B, EN)

[← Back to E4 Overview](../E4_RESULTS.md) · Sets: **6** real, **1** full-battery random, **4** WikiText-only random · Checks: 41 (0 problems).

## 1. Raw Benchmark Scores

Unmodified benchmark values (`word_perplexity` for WikiText; `%` accuracy for ARC-Challenge, HellaSwag, HellaSwag-es, and micro MMLU) for every evaluated set.

| Set | WikiText (ppl) | ARC-C (%) | HellaSwag (%) | HellaSwag-es (%) | MMLU micro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 9.531 | 46.33% | 74.04% | 58.95% | 56.53% |
| `real_gender20` | 9.604 | 46.33% | 74.08% | 58.90% | 56.39% |
| `real_gender5` | 9.580 | 46.08% | 74.09% | 58.95% | - |
| `real_race20` | 9.776 | 45.56% | 73.79% | 58.76% | 55.58% |
| `real_race5` | 9.576 | 46.16% | 73.90% | 58.89% | - |
| `real_religion20` | 9.804 | 45.82% | 73.75% | 58.60% | 56.49% |
| `real_religion5` | 9.684 | 45.73% | 73.77% | 58.76% | - |
| `rand_gender20_seed0` | 9.535 | 46.33% | 74.14% | 58.81% | 56.45% |
| `rand_gender20_seed1` | 9.555 | - | - | - | - |
| `rand_gender20_seed2` | 9.536 | - | - | - | - |
| `rand_gender20_seed3` | 9.537 | - | - | - | - |
| `rand_gender20_seed4` | 9.535 | - | - | - | - |

## 2. Paired Differences vs. Baseline (`B = 10,000`, `±1 pp` Margin)

Item-paired deltas (`set − baseline`) with 95% percentile bootstrap confidence intervals and equivalence verdicts for all real sets and full-battery random controls.

| Task | Set | Kind | Set Value | Baseline | Delta | 95% CI | Verdict |
| :--- | :--- | :---: | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` | real | 9.604 | 9.531 | +0.772% | [+0.521, +1.033] | - |
| wikitext | `real_gender5` | real | 9.580 | 9.531 | +0.518% | [+0.315, +0.748] | - |
| wikitext | `real_race20` | real | 9.776 | 9.531 | +2.575% | [+2.352, +2.813] | - |
| wikitext | `real_race5` | real | 9.576 | 9.531 | +0.472% | [+0.285, +0.669] | - |
| wikitext | `real_religion20` | real | 9.804 | 9.531 | +2.862% | [+2.595, +3.125] | - |
| wikitext | `real_religion5` | real | 9.684 | 9.531 | +1.609% | [+1.459, +1.782] | - |
| wikitext | `rand_gender20_seed0` | random | 9.535 | 9.531 | +0.043% | [+0.000, +0.113] | - |
| wikitext | `rand_gender20_seed1` | random | 9.555 | 9.531 | +0.249% | [+0.050, +0.478] | - |
| wikitext | `rand_gender20_seed2` | random | 9.536 | 9.531 | +0.056% | [-0.014, +0.147] | - |
| wikitext | `rand_gender20_seed3` | random | 9.537 | 9.531 | +0.068% | [+0.000, +0.189] | - |
| wikitext | `rand_gender20_seed4` | random | 9.535 | 9.531 | +0.040% | [-0.013, +0.111] | - |
| arc_challenge | `real_gender20` | real | 46.33% | 46.33% | +0.000 pp | [-0.853, +0.768] | equiv |
| arc_challenge | `real_gender5` | real | 46.08% | 46.33% | -0.256 pp | [-0.768, +0.256] | equiv |
| arc_challenge | `real_race20` | real | 45.56% | 46.33% | -0.768 pp | [-1.792, +0.256] | inconcl |
| arc_challenge | `real_race5` | real | 46.16% | 46.33% | -0.171 pp | [-0.939, +0.597] | equiv |
| arc_challenge | `real_religion20` | real | 45.82% | 46.33% | -0.512 pp | [-1.621, +0.597] | inconcl |
| arc_challenge | `real_religion5` | real | 45.73% | 46.33% | -0.597 pp | [-1.706, +0.427] | inconcl |
| arc_challenge | `rand_gender20_seed0` | random | 46.33% | 46.33% | +0.000 pp | [-0.597, +0.597] | equiv |
| hellaswag | `real_gender20` | real | 74.08% | 74.04% | +0.040 pp | [-0.249, +0.329] | equiv |
| hellaswag | `real_gender5` | real | 74.09% | 74.04% | +0.050 pp | [-0.189, +0.289] | equiv |
| hellaswag | `real_race20` | real | 73.79% | 74.04% | -0.249 pp | [-0.538, +0.040] | equiv |
| hellaswag | `real_race5` | real | 73.90% | 74.04% | -0.139 pp | [-0.349, +0.070] | equiv |
| hellaswag | `real_religion20` | real | 73.75% | 74.04% | -0.289 pp | [-0.617, +0.020] | equiv |
| hellaswag | `real_religion5` | real | 73.77% | 74.04% | -0.269 pp | [-0.548, +0.010] | equiv |
| hellaswag | `rand_gender20_seed0` | random | 74.14% | 74.04% | +0.100 pp | [-0.030, +0.239] | equiv |
| hellaswag_es | `real_gender20` | real | 58.90% | 58.95% | -0.053 pp | [-0.288, +0.181] | equiv |
| hellaswag_es | `real_gender5` | real | 58.95% | 58.95% | +0.000 pp | [-0.181, +0.171] | equiv |
| hellaswag_es | `real_race20` | real | 58.76% | 58.95% | -0.192 pp | [-0.480, +0.096] | equiv |
| hellaswag_es | `real_race5` | real | 58.89% | 58.95% | -0.064 pp | [-0.288, +0.160] | equiv |
| hellaswag_es | `real_religion20` | real | 58.60% | 58.95% | -0.352 pp | [-0.683, -0.021] | equiv |
| hellaswag_es | `real_religion5` | real | 58.76% | 58.95% | -0.192 pp | [-0.459, +0.064] | equiv |
| hellaswag_es | `rand_gender20_seed0` | random | 58.81% | 58.95% | -0.139 pp | [-0.309, +0.032] | equiv |
| mmlu | `real_gender20` | real | 56.39% | 56.53% | -0.142 pp | [-0.392, +0.107] | equiv |
| mmlu | `real_race20` | real | 55.58% | 56.53% | -0.954 pp | [-1.410, -0.506] | inconcl |
| mmlu | `real_religion20` | real | 56.49% | 56.53% | -0.043 pp | [-0.392, +0.299] | equiv |
| mmlu | `rand_gender20_seed0` | random | 56.45% | 56.53% | -0.078 pp | [-0.271, +0.114] | equiv |

## 3. Real vs. Layer-Matched Random Contrasts

Paired differences (`real − random`) across all tasks and random seeds available for this unit.

| Task | Comparison | Real Value | Random Value | Delta | 95% CI | Verdict |
| :--- | :--- | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` vs. `rand_gender20_seed0` | 9.604 | 9.535 | +0.728% | [+0.479, +0.991] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed1` | 9.604 | 9.555 | +0.521% | [+0.190, +0.855] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed2` | 9.604 | 9.536 | +0.715% | [+0.468, +0.975] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed3` | 9.604 | 9.537 | +0.703% | [+0.458, +0.965] | - |
| wikitext | `real_gender20` vs. `rand_gender20_seed4` | 9.604 | 9.535 | +0.732% | [+0.495, +0.981] | - |
| arc_challenge | `real_gender20` vs. `rand_gender20_seed0` | 46.33% | 46.33% | +0.000 pp | [-0.853, +0.853] | equiv |
| hellaswag | `real_gender20` vs. `rand_gender20_seed0` | 74.08% | 74.14% | -0.060 pp | [-0.339, +0.219] | equiv |
| hellaswag_es | `real_gender20` vs. `rand_gender20_seed0` | 58.90% | 58.81% | +0.085 pp | [-0.149, +0.320] | equiv |
| mmlu | `real_gender20` vs. `rand_gender20_seed0` | 56.39% | 56.45% | -0.064 pp | [-0.306, +0.185] | equiv |

## 4. MMLU Micro vs. Macro Reconciliation

Item-weighted micro accuracy (`14,042` items) compared with unweighted 57-subject macro accuracy and `_cap.json` values.

| Set | Micro (14,042 items) | Macro (57 subjects) | Macro − Micro | `_cap.json` Value |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 56.53% | 57.90% | +1.37 pp | 57.86% |
| `rand_gender20_seed0` | 56.45% | 57.83% | +1.38 pp | 57.79% |
| `real_gender20` | 56.39% | 57.77% | +1.38 pp | 57.72% |
| `real_race20` | 55.58% | 56.75% | +1.18 pp | 56.72% |
| `real_religion20` | 56.49% | 57.81% | +1.32 pp | 57.77% |

## 5. Hardware and Batch-Size Provenance

GPU hardware, batch size, and `lm-eval` version recorded per set and task from `tables/provenance.csv`.

<details>
<summary>Expand per-set provenance table</summary>

| Set | Task | GPU | Batch | lm-eval | Source Note |
| :--- | :--- | :--- | ---: | :---: | :--- |
| `baseline` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu (lm_eval 0.4.9.2, saved 2026-11-10) + verified in cap logs (arc_challenge/wikitext); NOTE: env.gpu reflects capabilities session, NOT bbq (see bbq row) |
| `baseline` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu (lm_eval 0.4.9.2, saved 2026-11-10) + verified in cap logs (arc_challenge/wikitext); NOTE: env.gpu reflects capabilities session, NOT bbq (see bbq row) |
| `baseline` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu (lm_eval 0.4.9.2, saved 2026-11-10) + verified in cap logs (arc_challenge/wikitext); NOTE: env.gpu reflects capabilities session, NOT bbq (see bbq row) |
| `baseline` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | per-set summary env.gpu and mmlu_batch_size=4 (lm_eval 0.4.9.2, 2026-11-10) |
| `baseline` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log pretty_env_info |
| `rand_gender20_seed0` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | T2 rerun (2026-11-23, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-13 L4 run (preserved as *_L4.json) |
| `rand_gender20_seed0` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | T2 rerun (2026-11-23, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-13 L4 run (preserved as *_L4.json) |
| `rand_gender20_seed0` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | T2 rerun (2026-11-23, lm_eval 0.4.9.2, nvidia-smi: NVIDIA A100-SXM4-80GB); supersedes 2026-11-13 L4 run (preserved as *_L4.json) |
| `rand_gender20_seed0` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | T2 rerun (2026-11-23, lm_eval 0.4.9.2, mmlu_batch_size=4, NVIDIA A100-SXM4-80GB); L4 preserved as *_L4.json |
| `rand_gender20_seed0` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | T2 rerun (2026-11-23, lm_eval 0.4.9.2, wikitext_batch_size=1, NVIDIA A100-SXM4-80GB); L4 preserved as *_L4.json |
| `rand_gender20_seed1` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | T1 run (2026-11-23, lm_eval 0.4.9.2, wikitext_batch_size=1, NVIDIA A100-SXM4-80GB) |
| `rand_gender20_seed2` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | T1 run (2026-11-23, lm_eval 0.4.9.2, wikitext_batch_size=1, NVIDIA A100-SXM4-80GB) |
| `rand_gender20_seed3` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | T1 run (2026-11-23, lm_eval 0.4.9.2, wikitext_batch_size=1, NVIDIA A100-SXM4-80GB) |
| `rand_gender20_seed4` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | T1 run (2026-11-23, lm_eval 0.4.9.2, wikitext_batch_size=1, NVIDIA A100-SXM4-80GB) |
| `real_gender20` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in bbq/arc_challenge/wikitext logs |
| `real_gender20` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in bbq/arc_challenge/wikitext logs |
| `real_gender20` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in bbq/arc_challenge/wikitext logs |
| `real_gender20` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | per-set summary env.gpu and mmlu_batch_size=4 (lm_eval 0.4.9.2, 2026-11-10) |
| `real_gender20` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log |
| `real_gender5` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_gender5` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_gender5` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_gender5` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log |
| `real_race20` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in arc_challenge/wikitext logs |
| `real_race20` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in arc_challenge/wikitext logs |
| `real_race20` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in arc_challenge/wikitext logs |
| `real_race20` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | per-set summary env.gpu and mmlu_batch_size=4 (lm_eval 0.4.9.2, 2026-11-10) |
| `real_race20` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log |
| `real_race5` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_race5` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_race5` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_race5` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log |
| `real_religion20` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in bbq/arc_challenge/wikitext logs |
| `real_religion20` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in bbq/arc_challenge/wikitext logs |
| `real_religion20` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10); verified in bbq/arc_challenge/wikitext logs |
| `real_religion20` | mmlu | A100-SXM4-80GB | 4 | 0.4.9.2 | per-set summary env.gpu and mmlu_batch_size=4 (lm_eval 0.4.9.2, 2026-11-10) |
| `real_religion20` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log |
| `real_religion5` | arc_challenge | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_religion5` | hellaswag | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_religion5` | hellaswag_es | A100-SXM4-80GB | 16 | 0.4.9.2 | per-set summary env.gpu and batch_size=16 (lm_eval 0.4.9.2, 2026-11-10; no MMLU per MMLU_KS=[20]); verified in arc_challenge/wikitext logs |
| `real_religion5` | wikitext | A100-SXM4-80GB | 1 | 0.4.9.2 | per-set summary env.gpu and wikitext_batch_size=1 (lm_eval 0.4.9.2, 2026-11-10); verified in wikitext log |

</details>
