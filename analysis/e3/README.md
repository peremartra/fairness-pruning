# analysis/e3: per-item analysis of the E3 interventions

Per-item analysis of one E3 unit (model × language; first Llama-3.1-8B EN): which answers change when neurons are zeroed,
in which direction, in which groups, whether different sets change the same items, and whether the changed items are
those where the unpruned model was undecided. Experiment log: `R1_exp.md`, section E3. Descriptive only: no thresholds,
no verdicts. The scripts below are for BBQ (English units); the Spanish unit (EsBBQ) has its own pair of scripts, described in the last section.

## Two steps

| Step | Script | Where | Reads | Writes |
|---|---|---|---|---|
| 1 | `extract_items.py` | Colab, **CPU runtime** (the logs are on Drive) | per-set summaries of notebook 10 and the per-item BBQ logs (about 700 MB each) | `items/<set_id>.csv.gz` (58,492 rows, a few MB) and `items/sets.json` |
| 2 | `e3_item_analysis.py` | laptop | the `items/` folder | `analysis/<timestamp>_items/` |

Step 1 is run once per unit and is resumable (sets already extracted are skipped; `--overwrite` recomputes them).

### Step 1 (Colab, CPU runtime)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q https://github.com/peremartra/fairness-pruning.git
!pip install -q ijson
!python fairness-pruning/analysis/e3/extract_items.py --unit-dir /content/drive/MyDrive/fair_pruning/results/e3/llama-3.1-8b_en
```

Inputs found automatically: `results/e3/<unit>/per_set/*_summary.json` (set list, kind, neurons, baseline log path);
logs of the zeroed sets in `checkpoints_e3/results/lm_evals/<unit>__<set_id>_bbq.json`; the baseline log is the one
recorded in the baseline summary (`baseline_bbq_log`: the E1 log reused after gate S4). Each set prints the number of
items and the mismatches against lm-eval's own per-item `acc` (expected: 58,492 and 0; otherwise "CHECK").

### Step 2 (laptop)

```bash
pip install numpy pandas tabulate
python analysis/e3/e3_item_analysis.py --items-dir "<Drive>/fair_pruning/results/e3/llama-3.1-8b_en/items"
```

Options: `--out-dir`, `--bootstrap` (default 1000), `--seed` (default 0), `--target-categories` (default Religion
Race_ethnicity Gender_identity; they get their own sections in `report.md`; the CSV files cover all 11 categories).

## Ablation sets

Sets with `kind = "ablation"` (notebook 10, `PHASE = "ablation"`, Llama-3.1-8B EN: `abl_shared1`, `abl_race4`, `abl_religion4`) are extracted like the others (the per-set summary carries `derived_from`) and are listed after the real sets in every table, in the overlap matrices and in the bootstrap. Random sets stay summarised as min / mean / max. The header of `report.md` counts real, ablation and random sets. Step 2 also runs in Colab (CPU): `!python fairness-pruning/analysis/e3/e3_item_analysis.py --items-dir <Drive>/.../items` writes the report to `<unit>/analysis/` on Drive. Use a fresh clone of the repository: an older clone does not know the ablation sets and silently leaves them out (the line "N sets (baseline + M)" would show M without them).

## L31 weight check (`l31_weight_check.py`)

Not part of the two steps. Reads only `down_proj` of one layer (default L31 of Llama-3.1-8B; no forward pass, CPU is enough; the model is gated and the script downloads the safetensors shard that holds the tensor, about 5 GB) and the `*_bias_scores.json` of the three categories. For a list of neurons (default: the nine of the E3 ablation) it reports the L2 norm of the `down_proj` column and the percentile within the layer of that norm, of the BiasScore of each category, and of BiasScore × column norm (a proxy, not the effect). Output: `<out-dir>/<timestamp>_l31_weights/` (`report.md`, `neurons.csv`, `layer_summary.json`). Percentile = % of the layer's neurons with a strictly lower value.

```python
!python fairness-pruning/analysis/e3/l31_weight_check.py \
  --scores-dir /content/drive/MyDrive/fairness-pruning/results/llama-3.1-8b/fairness-pruning-pairs-en \
  --out-dir /content/drive/MyDrive/fair_pruning/results/e3/llama-3.1-8b_en/analysis
```

## Per-item table (step 1)

One row per BBQ item: `doc_id`, `category`, `context` (ambig / disambig), `polarity`, `question_index`, `example_id`,
`subcategory`, `stereotyped_groups`; the two non-"unknown" answers in document order (`text0`, `text1`), their groups
from `answer_info` (`group0`, `group1`) and whether each is in `biased_answers` (`biased0`, `biased1`); `target` (0, 1,
2 = unknown), `pred` (same coding), `correct`, `biased` (the predicted person is in `biased_answers`); `ll0`, `ll1`
(loglikelihood of each person) and `ll_unk` (maximum over the ten "unknown" phrasings); `acc_lm_eval` (lm-eval's own
per-item accuracy, for the check). Mapping as in `claude/BBQ_log_format.md` and notebooks 09/10.

## What is written (step 2)

| File | Content |
|---|---|
| `report.md` | For each target category: direction of the changes, changes by group, overlap of the changed items, baseline margin of changed vs unchanged items, bootstrap intervals; real sets one by one, random sets as min / mean / max. Then changes by decile of the baseline margin. |
| `direction.csv` | Set × category × context (ambig, disambig, all): items, changed answers (n, %), changed answers that go to the stereotyped person / the other person / unknown, S→A and A→S, net towards the stereotype (% of items), correct→incorrect and incorrect→correct. |
| `transitions.csv` | Full 3 × 3 transition counts (baseline state → set state; S = stereotyped person, A = other person, U = unknown) per set, category and context. |
| `groups.csv` | Set × category × group (from `answer_info`): items that mention the group and % of them whose answer changed. |
| `overlap_changed_items_all.csv`, `overlap_changed_items_<category>.csv` | Jaccard index between the sets of changed items of every pair of sets (all items, and per target category). |
| `margin_changed_vs_unchanged.csv`, `margin_deciles.csv` | Baseline margin (top minus second loglikelihood among person 0, person 1, unknown) of changed and unchanged items; % changed per decile of that margin. |
| `bootstrap.csv` | Set × category: change in disambiguated bias and % of changed answers with 95 % percentile intervals from an item bootstrap (the same resamples for every set). |
| `results_full.json`, `inputs.json` | Every number above; every file read with size and sha256, script sha256, repository commit, versions, arguments. |

## Definitions

- **Changed answer**: the prediction (person 0, person 1, unknown) differs from the unpruned model's on the same item.
- **S / A / U**: the answer is the stereotyped person (in `biased_answers`, which already accounts for question polarity), the other person, or "unknown".
- **Net towards the stereotype** = (A→S − S→A) / items of the category and context, in %. In disambiguated items its sign follows the change in disambiguated bias.
- **Bootstrap**: items of the category are resampled with replacement (paired: the same items for the baseline and the set). Intervals reflect item sampling only, not run-to-run variation (runs are deterministic).
- Loglikelihoods are stored by lm-eval in bfloat16 resolution, so small margins are discretised.

## EsBBQ scripts (Spanish unit, Salamandra-7B ES)

EsBBQ is scored differently from BBQ (11 options, `label` is the target, bias measures from `question_type` and `stereotyped_groups`; see `claude/EsBBQ_log_format.md`), so it has its own scripts. The BBQ scripts are not changed.

| Step | Script | Where | Reads | Writes |
|---|---|---|---|---|
| 1 | `extract_items_esbbq.py` | Colab, CPU runtime, no extra packages | the neuron manifest of notebook 12, the baseline logs (`checkpoints_mbbq/results/lm_evals/bsc_lt_salamandra_7b_esbbq_<category>.json`) and the logs of the 8 experiments (`checkpoints_esbbq_zeroed_7b/results/lm_evals/salamandra_7b__<exp_id>_esbbq_<category>.json`), one file per category and set | `items/<set_id>.csv.gz` (27,320 rows each) and `items/sets.json`, in `fair_pruning/results/esbbq_zeroed_7b/` |
| 2 | `e3_item_analysis_esbbq.py` | Colab CPU or laptop | the `items/` folder | `analysis/<timestamp>_items/` next to it |

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q https://github.com/peremartra/fairness-pruning.git
!python fairness-pruning/analysis/e3/extract_items_esbbq.py --results-dir /content/drive/MyDrive/fair_pruning/results/esbbq_zeroed_7b
!pip install -q tabulate
!python fairness-pruning/analysis/e3/e3_item_analysis_esbbq.py --items-dir /content/drive/MyDrive/fair_pruning/results/esbbq_zeroed_7b/items
```

Use a fresh clone of the repository: an older clone silently runs older scripts. Step 1 is resumable (sets already extracted are skipped; `--overwrite` recomputes them, needed whenever the columns of the table change). Options of step 1: `--manifest`, `--logs-dir`, `--baseline-dir`, `--baseline-prefix`, `--out`, `--sets`, `--tasks` (subset, for tests). Options of step 2: `--out-dir`, `--bootstrap` (default 1000), `--seed` (default 0), `--target-categories` (default Religion Age Gender), `--allow-partial` (tests only).

**Checks of step 1** (stored in `sets.json` under `checks` and `problems`; a set with problems prints "CHECK"): items per category; prediction equal to lm-eval's per-item `acc`; the per-item metric tuples recomputed equal the ones lm-eval stored; the aggregates recomputed equal the `results` of each log (tolerance 1e-9); `label` is 2 in ambiguous items and 0 or 1 in disambiguated ones; "unknown" is `ans2`; no non-finite loglikelihoods. Ambiguous items where the pro-stereotype rule marks both persons the same way are kept, flagged (`rule_amb`) and counted. Step 2 refuses tables that do not cover the ten tasks.

**Per-item table.** Same columns as the BBQ table where they exist, with these differences: `item_id` (= task:`doc_id`; `doc_id` restarts in every category) replaces `doc_id` as the key; `question_type` (pro-stereo / anti-stereo / n/a), `relevant_social_value` (topic of the item) and `rule_amb` are added; `biased0`, `biased1` and `biased` mean "pro-stereotype person" according to `esbbq_utils` (not `biased_answers`); `ll_unk` is the maximum over the nine "unknown" phrasings.

**Report.** Same five questions and the same layout as the BBQ report, with these differences:
- Section 0 (new): % of changed answers of every category for every set. This unit has no random sets, so the categories that are not the target of a set are the only internal reference.
- Section 2b (new): % of changed answers by `relevant_social_value` (also `topics.csv`). Section 2 (by group) is printed only when the category has more than two groups; Age and Gender have two in every item, so there it would repeat the category total.
- Section 5 has three bias measures, each with its 95 % interval: `disamb_bias_delta` (the formula of the BBQ script, for comparison across units), `esbbq_disamb_bias_delta` (accuracy on pro-stereo minus accuracy on anti-stereo questions, as lm-eval reports it, in points) and `esbbq_amb_bias_delta` ((incorrect pro-stereo − incorrect anti-stereo) / ambiguous items, in points), plus the % of changed answers.
- No random or ablation sets: every set is read against the baseline and, in section 0, against the other categories. The report says so. The sets of one category come from the same ranking of neurons (Top-1 ⊂ Top-5 ⊂ Top-40), so a high Jaccard between them is expected.

**Reading limits.** Descriptive only. The bootstrap resamples items, not neurons or models. Small categories (Religion has 648 items in total) have few items per cell, so intervals are wide.
