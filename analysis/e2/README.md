# analysis/e2: random and cross controls (Llama-3.2-3B)

Tables and figure for experiment E2 (does zeroing the 20 neurons selected for Religion change the Religion bias more than
zeroing 20 random neurons of the same layers?). Experiment log: `R1_exp.md`, section E2.

## Inputs (Drive, `fair_pruning/`)

| File | What it is |
|---|---|
| `results/e2/per_set/*_summary.json` | One small JSON per set, written by `notebooks/09_E2_Controls_Llama3B.ipynb` as soon as the set is evaluated: `baseline`, `real_religion20`, `real_race20` and `rand_religion_seed0` to `seed9` (13 files). BBQ by category with the percentage of changed answers against the baseline, raw capabilities, neurons zeroed, environment, sha256 of the manifest. |
| `results/e2/per_set/baseline_bbq_by_category.json` | BBQ of the unpruned model by category. It is not in `baseline_summary.json` (that field is `null`), so it is produced once by `make_baseline_table.py`. |
| `checkpoints_bbq_rerun/results/lm_evals/meta_llama_llama_3.2_3b_bbq.json` | Per-item lm-eval log of the unpruned model (about 700 MB). Read only by `make_baseline_table.py`. |
| `results/e2_bbq_by_category_*.csv` (optional) | Consolidated CSV of notebook 09, used as a cross-check of the summaries. |

## How to run

```bash
# 1. once: BBQ table of the unpruned model (about 700 MB log; `pip install ijson` avoids loading it into memory)
python analysis/e2/make_baseline_table.py \
    --log "<Drive>/fair_pruning/checkpoints_bbq_rerun/results/lm_evals/meta_llama_llama_3.2_3b_bbq.json" \
    --out "<Drive>/fair_pruning/results/e2/per_set/baseline_bbq_by_category.json"

# 2. analysis
python analysis/e2/e2_analysis.py --input-dir "<Drive>/fair_pruning/results/e2/per_set"
```

`<Drive>` is the Drive folder as seen from your machine (Google Drive for desktop, or a copy of the files).
Options of `e2_analysis.py`: `--out-dir`, `--baseline-bbq`, `--csv`, `--l4-seeds` (default `0 1`), `--no-figures`.

## What is written

Folder `<input-dir>/../analysis/<timestamp>/` (on Drive: `fair_pruning/results/e2/analysis/<timestamp>/`):

| File | Content |
|---|---|
| `report.md` | All the tables below in one page. |
| `table_main.md` | The table of the paper: unpruned / 20 selected neurons / 20 random neurons (range over the 10 sets); Religion disambiguated bias, its change in pp, and percentage of changed answers. |
| `table_main_extended.md` | The same with Race-20 (cross control), disambiguated accuracy, random mean and range, and WikiText. |
| `inventory.csv`, `checks.csv`, `overlap_matrix.csv` | Sets found, neurons per layer, environment, GPU, consistency checks and overlap of neurons between all sets. |
| `bbq_long.csv`, `bbq_baseline.csv` | Set x category, all BBQ metrics and their change against the baseline (signed, and in magnitude). |
| `random_stats_<category>.csv`, `real_vs_random_<category>.csv` | For Religion and Race_ethnicity: distribution of the random sets (n, mean, SD, median, min, max, of the value and of the change) and position of each real set. Also `..._without_L4_seeds.csv`. |
| `collateral_by_category.csv`, `pooled.csv` | Effect of every set on every category; pooled values weighted by number of items. |
| `capabilities.csv`, `capabilities_random_stats*.csv` | WikiText for the 13 sets, MMLU / ARC-Challenge / HellaSwag / HellaSwag-es for baseline and real sets. |
| `e2_religion_strip.png` / `.pdf` | One dot per random set and one line per real set: change in Religion disambiguated bias, Religion answers changed, change in WikiText perplexity. Seeds whose BBQ ran on an L4 are drawn open. |
| `results_full.json` | Every number above. |
| `inputs.json` | Every file read, with size and sha256; script sha256, repository commit, python / numpy / pandas versions, arguments. |

## Definitions

- **Change** = value of the set minus value of the unpruned model (pp for bias and accuracy). `abs_*` rows use |value| (magnitude), so a bias that crosses zero shows the change in magnitude.
- **Changed answers** (`flips_pct`) = percentage of items whose prediction (person 0, person 1, unknown) differs from the unpruned model, per category; the baseline is 0 by construction.
- **Rank** (`rank_abs_delta_of_N`) = 1 + number of random sets whose absolute change is larger than that of the real set (1 = largest change among real set + random sets). When all values are tied (for example ambiguous accuracy, which is the same in every random set) the rank is 1 and `p_empirical` is 1.0: read the values, not the rank.
- **`n_random_abs_delta_ge_real`** = number of random sets whose absolute change is at least that of the real set; **`p_empirical`** = (that number + 1) / (n + 1), so with 10 random sets the smallest possible value is 1/11.
- **`z_signed_delta`** = (change of the real set - mean change of the random sets) / sample SD (n - 1) of the random sets. It is reported as raw descriptive information; with 10 sets it is a rough number.
- **Sensitivity without L4 seeds**: seeds 0 and 1 evaluated BBQ on an NVIDIA L4 and the rest on an A100 (see `R1_exp.md`, E2/S5); the `..._without_L4_seeds` tables repeat the analysis with seeds 2 to 9 only.
- **GPU**: the `gpu` field inside a summary is the Colab session that wrote the file, not necessarily the one that ran the evaluation. The inventory shows both columns: `summary_session_gpu` (from the data) and `bbq_gpu_from_R1_exp` (from the experiment log; edit `bbq_gpu_note` in `e2_analysis.py` if it changes).

## Checks (`checks.csv`)

13 sets present; same manifest sha256, torch, transformers, lm-eval and optipfair versions in all summaries; `eval_limit` null (no smoke tests); 11 categories per set with BBQ; random sets with the same neurons per layer as Religion-20 and no overlap with it; maximum overlap between two random sets; overlap Religion-20 / Race-20; agreement of the summaries with the consolidated CSV. A failed check is printed on screen and the script keeps going.

## Reference run (2026-11-04)

Input: the 13 summaries of 2026-11-03; output on Drive `fair_pruning/results/e2/analysis/20261104_092942/`; 13 of 13 checks passed (summaries equal to the consolidated CSV to 5e-11). Main table of that run:

| | Religion bias | Change | Changed answers |
|---|---|---|---|
| Unpruned | 3.84 | - | - |
| 20 selected neurons | -1.33 | -5.17 | 11.1 % |
| 20 random neurons (10 sets) | 2.84 to 4.17 | -1.00 to +0.33 | 1.1 to 1.7 % |

## What these scripts do not do

No confidence intervals, no correct/incorrect breakdown of the changed answers and no KL between option distributions: they need the per-item logs of the 12 sets and belong to a separate script. Only Religion-matched random sets exist, so Race-20 has no random counterpart (the Race_ethnicity tables use the Religion random sets as reference).
