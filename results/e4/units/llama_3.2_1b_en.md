# E4 Unit Dossier — `llama_3.2_1b_en` (Llama-3.2-1B, EN)

[← Back to E4 Overview](../E4_RESULTS.md) · Sets: **6** real, **0** full-battery random, **0** WikiText-only random · Checks: 32 (0 problems).

## 1. Raw Benchmark Scores

Unmodified benchmark values (`word_perplexity` for WikiText; `%` accuracy for ARC-Challenge, HellaSwag, HellaSwag-es, and micro MMLU) for every evaluated set.

| Set | WikiText (ppl) | ARC-C (%) | HellaSwag (%) | HellaSwag-es (%) | MMLU micro (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 11.985 | 37.03% | 64.30% | 47.23% | 31.12% |
| `real_gender20` | 12.141 | 36.18% | 64.05% | 47.06% | 32.32% |
| `real_gender5` | 12.013 | 36.43% | 64.15% | 47.13% | - |
| `real_race20` | 12.093 | 36.60% | 64.27% | 47.34% | 31.56% |
| `real_race5` | 12.019 | 36.26% | 64.13% | 47.30% | - |
| `real_religion20` | 12.130 | 37.03% | 63.92% | 46.97% | 32.15% |
| `real_religion5` | 12.030 | 36.69% | 64.11% | 47.05% | - |

## 2. Paired Differences vs. Baseline (`B = 10,000`, `±1 pp` Margin)

Item-paired deltas (`set − baseline`) with 95% percentile bootstrap confidence intervals and equivalence verdicts for all real sets and full-battery random controls.

| Task | Set | Kind | Set Value | Baseline | Delta | 95% CI | Verdict |
| :--- | :--- | :---: | ---: | ---: | ---: | :---: | :---: |
| wikitext | `real_gender20` | real | 12.141 | 11.985 | +1.296% | [+1.071, +1.545] | - |
| wikitext | `real_gender5` | real | 12.013 | 11.985 | +0.229% | [+0.063, +0.478] | - |
| wikitext | `real_race20` | real | 12.093 | 11.985 | +0.899% | [+0.678, +1.120] | - |
| wikitext | `real_race5` | real | 12.019 | 11.985 | +0.279% | [+0.139, +0.446] | - |
| wikitext | `real_religion20` | real | 12.130 | 11.985 | +1.206% | [+0.735, +1.950] | - |
| wikitext | `real_religion5` | real | 12.030 | 11.985 | +0.377% | [-0.051, +1.145] | - |
| arc_challenge | `real_gender20` | real | 36.18% | 37.03% | -0.853 pp | [-2.048, +0.341] | inconcl |
| arc_challenge | `real_gender5` | real | 36.43% | 37.03% | -0.597 pp | [-1.195, +0.000] | inconcl |
| arc_challenge | `real_race20` | real | 36.60% | 37.03% | -0.427 pp | [-1.451, +0.512] | inconcl |
| arc_challenge | `real_race5` | real | 36.26% | 37.03% | -0.768 pp | [-1.621, +0.085] | inconcl |
| arc_challenge | `real_religion20` | real | 37.03% | 37.03% | +0.000 pp | [-0.939, +0.939] | equiv |
| arc_challenge | `real_religion5` | real | 36.69% | 37.03% | -0.341 pp | [-1.195, +0.427] | inconcl |
| hellaswag | `real_gender20` | real | 64.05% | 64.30% | -0.249 pp | [-0.558, +0.060] | equiv |
| hellaswag | `real_gender5` | real | 64.15% | 64.30% | -0.149 pp | [-0.368, +0.070] | equiv |
| hellaswag | `real_race20` | real | 64.27% | 64.30% | -0.030 pp | [-0.259, +0.209] | equiv |
| hellaswag | `real_race5` | real | 64.13% | 64.30% | -0.169 pp | [-0.378, +0.040] | equiv |
| hellaswag | `real_religion20` | real | 63.92% | 64.30% | -0.378 pp | [-0.637, -0.119] | equiv |
| hellaswag | `real_religion5` | real | 64.11% | 64.30% | -0.189 pp | [-0.408, +0.020] | equiv |
| hellaswag_es | `real_gender20` | real | 47.06% | 47.23% | -0.171 pp | [-0.469, +0.128] | equiv |
| hellaswag_es | `real_gender5` | real | 47.13% | 47.23% | -0.096 pp | [-0.267, +0.075] | equiv |
| hellaswag_es | `real_race20` | real | 47.34% | 47.23% | +0.117 pp | [-0.139, +0.373] | equiv |
| hellaswag_es | `real_race5` | real | 47.30% | 47.23% | +0.075 pp | [-0.149, +0.288] | equiv |
| hellaswag_es | `real_religion20` | real | 46.97% | 47.23% | -0.256 pp | [-0.534, +0.032] | equiv |
| hellaswag_es | `real_religion5` | real | 47.05% | 47.23% | -0.181 pp | [-0.448, +0.085] | equiv |
| mmlu | `real_gender20` | real | 32.32% | 31.12% | +1.196 pp | [+0.712, +1.695] | excl (<35%) |
| mmlu | `real_race20` | real | 31.56% | 31.12% | +0.442 pp | [+0.028, +0.847] | excl (<35%) |
| mmlu | `real_religion20` | real | 32.15% | 31.12% | +1.033 pp | [+0.520, +1.545] | excl (<35%) |

## 3. MMLU Micro vs. Macro Reconciliation

Item-weighted micro accuracy (`14,042` items) compared with unweighted 57-subject macro accuracy and `_cap.json` values.

| Set | Micro (14,042 items) | Macro (57 subjects) | Macro − Micro | `_cap.json` Value |
| :--- | ---: | ---: | ---: | ---: |
| `baseline` | 31.12% | 31.60% | +0.48 pp | 31.59% |
| `real_gender20` | 32.32% | 32.91% | +0.59 pp | 32.90% |
| `real_race20` | 31.56% | 31.91% | +0.35 pp | 31.91% |
| `real_religion20` | 32.15% | 32.61% | +0.46 pp | 32.60% |

## 4. Hardware and Batch-Size Provenance

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
