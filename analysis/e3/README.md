# analysis/e3: per-item analysis of the E3 interventions

Per-item analysis of one E3 unit (model × language; first Llama-3.1-8B EN): which answers change when neurons are zeroed,
in which direction, in which groups, whether different sets change the same items, and whether the changed items are
those where the unpruned model was undecided. Experiment log: `R1_exp.md`, section E3. Descriptive only: no thresholds,
no verdicts.

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
