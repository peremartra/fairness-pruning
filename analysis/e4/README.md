# analysis/e4: capability cost of the E3 interventions

How much the Gender / Race / Religion interventions (K = 5 and 20) change general capabilities with respect to the unpruned
model and, where it exists, with respect to a random control of the same size: WikiText (0-shot word perplexity), MMLU (5-shot),
ARC-Challenge, HellaSwag and HellaSwag-es (0-shot, `acc_norm`). Experiment log: `R1_exp.md`, section E4; plan and run record:
`claude/E4_plan.md`, `claude/E4_equivalence_plan.md`. Everything here reads logs that already exist: no model is run, no GPU.
The scripts compute differences and intervals; the verdict column only applies the margin fixed beforehand (see Definitions).
This analysis replaces the earlier bootstrap of the two first models (`results/e4_bootstrap/run_20261113_103721`): same method, now
for five units and every set; `--compare-old` prints OK/WARN against it and never stops.

## Two steps

| Step | Script | Where | Reads | Writes |
|---|---|---|---|---|
| 1 | `extract_items_cap.py` | Colab, **CPU runtime** (the logs are on Drive) | per-item lm-eval logs of notebook 10 (MMLU about 200 MB each) and the `_cap.json` of each set | `<unit>/items/<set_id>__<task>.csv.gz` and `<unit>/items/sets.json` |
| 2 | `e4_cap_analysis.py` | Colab CPU or laptop | the `items/` folders | `analysis/<timestamp>_caps/` |

Step 1 is run once and is resumable (tables already extracted are skipped; `--overwrite` recomputes them). Step 2 never overwrites:
every run writes a new timestamped folder.

### Step 1 (Colab, CPU runtime)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q https://github.com/peremartra/fairness-pruning.git
!python fairness-pruning/analysis/e4/extract_items_cap.py --fair-pruning-dir /content/drive/MyDrive/fair_pruning
```

Options: `--units` (default: the five below), `--sets`, `--tasks`, `--logs-dir`, `--out-root`, `--overwrite`.

### Step 2 (Colab, CPU runtime)

```python
!python fairness-pruning/analysis/e4/e4_cap_analysis.py \
  --items-root /content/drive/MyDrive/fair_pruning/results/e4 \
  --compare-old /content/drive/MyDrive/fair_pruning/results/e4_bootstrap/run_20261113_103721/e4_bootstrap_deltas.csv
```

Options: `--units`, `--bootstrap` (default **10000**; E3 uses 1000 for BBQ), `--seed` (0), `--margin` (1.0 pp),
`--chance-excl` (0.35), `--provenance` (default `provenance.csv` next to the script), `--out-dir`.

## Inputs

| What | Where on Drive (`fair_pruning/`) |
|---|---|
| Per-item logs | `checkpoints_e3_cap/results/lm_evals/<unit>__<set_id>_<task>.json`; tasks `arc_challenge`, `hellaswag`, `hellaswag_es`, `mmlu`, `wikitext`. Files with another suffix (`_mmlu_bs1`, `_pre_E4`) are not read |
| Per-set capability values | `checkpoints_e3_cap/<unit>/<set_id>_cap.json` (cross-check only) |

Units: `llama_3.1_8b_en`, `qwen3_4b_base_en`, `salamandra_7b_en`, `llama_3.2_3b_en` (E3 baseline only; the E2 runs are not mixed), `llama_3.2_1b_en`.
Sets: `baseline`; `real_{gender,race,religion}{5,20}`; `rand_gender20_seed0` (not in Llama-3.2-1B). MMLU exists only for the baseline, the
K = 20 real sets and the random set. Random control with the full battery exists only for Gender·20, so Race·20 and Religion·20 are
compared with the baseline only.

## What is written (step 1)

| File | Content |
|---|---|
| `<unit>/items/<set_id>__<task>.csv.gz` | One row per item, sorted by (group, doc_id). ARC-C, HellaSwag, HellaSwag-es, MMLU: `group` (subject in MMLU), `doc_id`, `doc_hash`, `target`, `acc`, `acc_norm` (empty in MMLU). WikiText: `doc_id`, `doc_hash`, `loglik`, `n_words` (62 documents). |
| `<unit>/items/sets.json` | Per set (kind, category, K, seed) and task: source log, size, sha256 of log and table, items, aggregate recomputed vs lm-eval vs `_cap.json`, lm-eval version, log date, `problems`. |

Checks per file (a file with problems prints "CHECK" and the run continues): item count (ARC-C 1,172; HellaSwag 10,042; HellaSwag-es 9,374;
MMLU 14,042 in 57 subjects; WikiText 62), aggregate from the items equal to lm-eval's (accuracy 1e-5; WikiText 5e-4 relative), argmax of the
logged loglikelihoods equal to the logged `acc`, finite values, no `limit`, number of few-shot examples (MMLU 5, others 0). The `_cap.json`
value is compared except in MMLU, where it is a macro average (see Definitions).

## What is written (step 2)

| File | Content |
|---|---|
| `report.md` | Per unit, table of task × comparison with delta, 95 % interval and verdict. |
| `deltas.csv` | Every comparison: unit, task, comparison, sets, delta, interval, verdict, baseline value, bootstrap, seed, margin. Real sets vs baseline always; random vs baseline and real vs random where the random set exists. |
| `raw_values.csv` | Value of every set and task (accuracy in %, MMLU micro; WikiText word perplexity) and number of items. |
| `mmlu_micro_macro.csv` | Per set: micro from lm-eval, micro from the items, macro over the 57 subjects, value stored in the `_cap.json`, difference. |
| `e4_capabilities_table.csv` | Compact table (value of each side, delta, interval, verdict). **This is the file copied to `results/e4/` in the repository.** |
| `provenance.csv` | GPU and batch size per unit, set and task, from `analysis/e4/provenance.csv`; `unknown` where nothing is recorded. |
| `checks.csv` | Items, alignment with the baseline (keys and `doc_hash`), aggregate checks, extraction problems. |
| `compare_old_bootstrap.csv` | Only with `--compare-old`: old vs new point estimates (OK / WARN, tolerance 0.02). |
| `results_full.json`, `inputs.json` | Every number; sha256 of each table read, script sha256, repository commit, versions, arguments. |

## Definitions

- **Delta**: set A minus set B on the same items. Accuracy tasks in percentage points (pp). WikiText: relative change in % of word perplexity, `exp(-Σ loglik / Σ words)`, resampling documents.
- **Paired bootstrap**: items (documents in WikiText) are resampled with replacement; the same indices are used for every set of a unit and task; B = 10,000 replicates; 95 % percentile interval; RNG seed `[seed, unit index, task index]`.
- **Verdict** (accuracy tasks only; margin ±1 pp, fixed beforehand): `equivalent` = whole interval inside ±1; `worse` = upper end below −1; `better` = lower end above +1; otherwise `inconclusive`. Tasks with a baseline accuracy below 0.35 (chance + 10 pp, applied to the baseline only) are `excluded`; in the present data this concerns Llama-3.2-1B MMLU (about 32 %). WikiText has no verdict.
- **MMLU micro / macro**: micro is item-weighted over 14,042 items (what lm-eval returns and what is analysed here); macro is the mean over the 57 subjects (what the `_cap.json` stores; it differs from micro by up to about 1 pp). Do not mix them in one table.
- **Random control**: one random set, seed 0, same layers and K as the real set. With one seed the real-vs-random contrast has no between-seed variance.

## Limits

- Provenance: the GPU and batch size of each cell are not stored in the logs. Known cases are in `provenance.csv` (random-set tasks ran on L4, except WikiText and Llama MMLU on A100; Salamandra random MMLU on A100 at batch 4; baseline and real sets on A100). Any real-vs-random difference carries this hardware noise; the interval includes it.
- Intervals describe item sampling only, not variance across seeds or across hardware.
- ARC-Challenge has about 1.2 k items, so its interval rarely fits inside ±1 pp; "inconclusive" is expected there.
- Llama-3.2-3B uses only the E3 runs; its E2 baseline differs and is not used.
- This analysis supports statements about the cost in capabilities, not about localization.
