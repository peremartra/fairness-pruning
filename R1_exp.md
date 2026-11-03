# R1 — Experiments tracker and run guide (P0: Fairness Pruning)

Last updated: 2026-11-02

Purpose: one place to see the status of each experiment (E0–E6) and to consult before launching the next one. Numbers are kept to the minimum needed to read the status; the full tables live in the linked files.

How to maintain it: when something changes, (1) update the row in the Status table, (2) edit only the affected experiment section, (3) add a dated line to the Changelog at the bottom.

Status values: **Done** · **In progress** · **Pending** · **Not planned**

## Status

| Exp | What | Status | Models | Headline | Section |
|---|---|---|---|---|---|
| E0 | Last-layer checks on stored scores | Done | Llama-3.2-1B, 3B, Salamandra-2B | Last-layer share in top-1 %: 34–71 % raw, 12–31 % after per-layer normalisation | [E0](#e0--last-layer-checks) |
| E1 | New models and baselines | Done | Llama-3.1-8B, Salamandra-7B, Qwen3-4B-Base | Disambiguated accuracy above 60 % in BBQ and EsBBQ for the three; all OptiPFair-compatible; A/B/C base/instruct pair study: Llama-3.2-3B and Llama-3.1-8B pairs done; Salamandra-7B base does not follow the format | [E1](#e1--new-models-and-baselines) |
| E2 | Random and cross controls (pilot) | In progress | Llama-3.2-3B | Design closed 2026-11-02; S1, S2/S4 and S3 (real sets, full BBQ + capabilities) done; Religion·Top-20 lowers Religion bias more than Race·Top-20 does, but both change 4–12 % of the answers in every category; two identical baseline runs are bit-for-bit equal, so those changes are an effect of the intervention and not run-to-run noise; S5 (10 random sets, seeds 0–9) done 2026-11-03; next: analysis (S6) | [E2](#e2--random-and-cross-controls-pilot-in-llama-32-3b) |
| E3 | Intervention and controls in new models | Pending | Llama-3.1-8B, Salamandra-7B | Blocked by E1 and E2 | [E3](#e3--intervention-and-controls-in-the-new-models) |
| E4 | Capabilities | Pending | All with interventions | Missing configurations listed | [E4](#e4--capabilities) |
| E5 | Bootstrap statistics | Pending | Llama-3.2-3B + new models | Needs per-item logs (partly available) | [E5](#e5--statistics-bootstrap) |
| E6 | DPO pilot on selected neurons | Not planned for P0 | — | Outside P0 | [E6](#e6--dpo-pilot-on-selected-neurons) |

## Common protocol (applies to every run)

1. **Environment.** Colab with an A100 (40 GB or more) is preferred; an L4 works but is several times slower. The setup cell installs `requirements.txt` (lm-eval 0.4.9.2, torch 2.10.0, transformers 4.57.3) and prints versions and GPU. Copy them into the evaluation log at the top of the notebook.
2. **Evaluation settings.** 0-shot, full dataset (`limit=None`), bfloat16, `batch_size=16` fixed (not `"auto"`), `log_samples=True`. Do not mix settings inside one comparison. Reference throughput: full BBQ for a 7–8B model takes about 56 min on an A100-40GB (about 207 requests/s); full EsBBQ for Salamandra-7B took about 28 min on an A100-80GB.
3. **Checkpoints.** Drive: `fair_pruning/<checkpoints_dir>/<size>/<model>.json`. `run_robust_evaluation` skips tasks already completed in a checkpoint, so **to repeat a run use a new checkpoint directory** (for example `checkpoints_bbq_rerun`).
4. **Per-item logs.** Raw lm-eval JSON with per-item predictions: `<checkpoints_dir>/results/lm_evals/<model>_<task>.json` (about 700 MB for full BBQ; Drive only, not in the repository).
5. **Per-category accuracy.** lm-eval's `bbq` task reports per-category bias scores but not per-category accuracy; compute it from the per-item logs.
6. **How to report bias.** Ambiguous accuracy, disambiguated accuracy and S (ambiguous bias score / (1 − ambiguous accuracy)). BBQ aggregates are those of lm-eval (pooled over items); EsBBQ aggregates are item-weighted. The disambiguated bias score has a different definition in BBQ and EsBBQ and must not be compared across languages. Details: [results/bias-benchmarks-base/README.md](results/bias-benchmarks-base/README.md).
7. **After each run.** Copy the consolidated CSVs to `results/`, fill the evaluation log in the notebook, update this file.
8. **Design before code.** New notebooks and scripts are discussed before they are written.

## E0 — Last-layer checks

- **Status:** Done (2026-10-27). Repeat when new models are added.
- **Goal:** check how much of the last-layer concentration of top-scoring neurons is due to the scale of the scores.
- **Design:** read-only analysis of the stored BiasScores (`down_proj` input), 3 models × EN/ES × 5 categories = 30 cases. Scale proxy: per-layer median of the score (an assumption, because only |a1 − a2| is stored).
- **Results:**
  - The paper's criterion reproduces in 30/30 cases (last layer has the highest mean score).
  - Share of the last layer in the global top-1 %: 34–71 % raw, 12–31 % after per-layer normalisation (uniform: 4–6 %). It remains the most over-represented layer.
  - Overlap across categories in the last layer (top-40): mean Jaccard 0.19–0.31 (random: 0.002–0.004).
- **Interpretation:** raw magnitude is a reasonable criterion for deciding what to intervene on (larger activation difference means more potential influence). What is **not** shown is that these neurons are specific to the demographic attribute. That needs random last-layer magnitude-matched controls and non-demographic control pairs (second iteration).
- **Files:** [docs/E0_last_layer_checks.md](docs/E0_last_layer_checks.md) · [docs/E0_last_layer_check.py](docs/E0_last_layer_check.py) · [docs/E0_last_layer_overlap.py](docs/E0_last_layer_overlap.py) · data in [results/neuron_analysis/](results/neuron_analysis/)
- **Open items:**
  - [ ] Recompute the percentages with the new models before citing them.

## E1 — New models and baselines

- **Status:** Done (closed by Pere, 2026-11-02; the verifications left in the open items are not blocking). Baselines of Llama-3.1-8B and Salamandra-7B finished on 2026-10-28 and Qwen3-4B-Base on 2026-10-29; data files copied to the repository (commit pending); the figures are left for the writing phase. A/B/C base/instruct pair study done on 2026-10-30 for Llama-3.2-3B and Llama-3.1-8B (plus Salamandra-7B base only); see open items.
- **Goal:** add Llama-3.1-8B and Salamandra-7B, and check that their baselines are clearly above chance. An instruct model (Llama-3.2-3B-Instruct) was evaluated; whether to include one is open again (see open items). Per-item logs are reused later for bootstrap (E5). Gemma was discarded; Qwen3-4B-Base was added as a third family (standard dense GLU-MLP, architecture `qwen3`; the hybrid-attention Qwen3.5 would need changes to the extraction code and is left for later).
- **Design (as run):** BBQ (English) and EsBBQ (Spanish), 0-shot, full dataset, bfloat16, `batch_size=16`, `log_samples=True`. Llama-3.2-3B-Instruct was also evaluated, in English only, with a different task (`bbq_instruct_mc`: options A/B/C, chat template); its numbers are not comparable with the base-model ones and it is a candidate for an appendix (see the pair study below). The position-bias diagnostic of the plan (is the ~50 % disambiguated accuracy of Llama-3.2-1B and Salamandra-2B in EsBBQ position bias or chance?) is dropped (2026-10-29); it will be run only if a reviewer asks. It would need the per-item logs of those two models, which exist on Drive.
- **Results** (item-weighted; reference for "above chance": about 50 % in disambiguated items is random choice between the two named answers):

| Model | BBQ amb / disamb / S | EsBBQ amb / disamb / S |
|---|---|---|
| Salamandra-2B | 7.7 / 50.5 / 2.2 | 24.4 / 49.8 / 2.0 |
| Llama-3.2-1B | 10.0 / 52.3 / 1.9 | 23.5 / 49.9 / 1.4 |
| Llama-3.2-3B | 7.5 / 73.5 / 4.5 | 14.8 / 70.0 / 6.2 |
| Llama-3.1-8B | 8.4 / 82.0 / 6.1 | 13.5 / 74.2 / 8.8 |
| Salamandra-7B | 7.7 / 80.4 / 9.2 | 8.8 / 73.4 / 9.1 |
| Qwen3-4B-Base | 15.0 / 86.7 / 15.1 | 12.0 / 88.8 / 20.1 |

  - Qwen3-4B-Base (4B) has the highest disambiguated accuracy and the highest S in both languages, above the 7-8B models; S does not follow size across families (it grows with size within each family). No causal explanation is claimed.
  - Low ambiguous accuracy in every model is an aversion to the "unknown" option, not an inability: disambiguated accuracy is high.
  - EsBBQ values are reconstructed from category-level results (about ±0.1 pp); exact values will come from the per-item logs.
- **Base vs instruct in the A/B/C format** (English BBQ only; a different task from the table above, so these numbers are not comparable with it). Same prompt for base and instruct (options A/B/C); they differ in the chat template (instruct) and the final `Answer:` line (base). All values in %:

| Llama-3.2-3B | Task | Amb. acc | Disamb. acc | Amb. bias | Disamb. bias |
|---|---|---|---|---|---|
| Base, standard format | `bbq` | 7.51 | 73.54 | 4.12 | 2.24 |
| Base, A/B/C | `bbq_base_mc` | 7.25 | 72.06 | 11.49 | 4.88 |
| Instruct, A/B/C | `bbq_instruct_mc` | 59.03 | 78.50 | 7.29 | 3.63 |

  - Llama-3.2-3B: the A/B/C format moves the base model by −0.3 pp (ambiguous accuracy) and −1.5 pp (disambiguated); the instruct model with the same options and the chat template is 51.8 pp higher in ambiguous accuracy (this difference mixes instruction tuning and chat template; see the standard-format table below). The ambiguous bias score of the base model is multiplied by about 2.8 by the format (4.12 → 11.49) and Religion goes from 6.00 to 16.00.
  - Caveats: one run per model, no noise estimate (the standard-format baseline of the 3B is from the older configuration; its repeat is in E2). The "standard" row of the 3B comes from the paper's Table IV and the READMEs. Bias scores of A/B/C runs are only comparable with other A/B/C runs.

| Llama-3.1-8B | Task | Amb. acc | Disamb. acc | Amb. bias | Disamb. bias |
|---|---|---|---|---|---|
| Base, standard format | `bbq` | 8.35 | 82.02 | 5.59 | 2.08 |
| Base, A/B/C | `bbq_base_mc` | 38.28 | 82.74 | 10.03 | 5.01 |
| Instruct, A/B/C | `bbq_instruct_mc` | 81.31 | 89.86 | 6.04 | 2.08 |

  - Llama-3.1-8B (2026-10-30): the A/B/C format raises the base model's ambiguous accuracy from 8.35 to 38.28 % (+29.9 pp) while the disambiguated accuracy stays the same (82.02 → 82.74 %); the instruct model with the same options and the chat template is a further 43.0 pp higher (tuning and chat template are mixed here). Overall accuracy: 45.19 (standard), 60.51 (A/B/C base), 85.59 (A/B/C instruct). The ambiguous bias score of the base rises from 5.59 to 10.03 with the format and falls to 6.04 in the instruct model; the disambiguated one goes 2.08 → 5.01 → 2.08.
  - Across the two Llama models the effect of the format on the base model's ambiguous accuracy differs (3B: about 0; 8B: +29.9 pp); the instruct models in A/B/C with chat template are 43–52 pp higher in both (tuning and chat template mixed). Sources: `instruct_models_bbq_mc_results_20261030_103101.csv` (contains both instruct models) and `base_models_bbq_abc_results_20261030_102243.csv` (contains the three base models), Drive: `fair_pruning/results/`.
  - **Instruct models in the standard format** (2026-10-30, run 8 of `02_Evaluate_BBQ.ipynb`, `PROMPT_FORMAT = "standard_instruct"`: task `bbq`, same prompt as the base models, no chat template). All values in %, comparable with the "Base, standard format" rows:

| Model | Overall acc. | Amb. acc | Disamb. acc | Amb. bias | Disamb. bias |
|---|---|---|---|---|---|
| Llama-3.2-3B base | 40.52 | 7.51 | 73.54 | 4.12 | 2.24 |
| Llama-3.2-3B-Instruct | 47.21 | 10.92 | 83.50 | 8.77 | 3.24 |
| Llama-3.1-8B base | 45.19 | 8.35 | 82.02 | 5.59 | 2.08 |
| Llama-3.1-8B-Instruct | 49.41 | 8.47 | 90.35 | 13.87 | 1.88 |

  - Reading: with the same task and prompt, instruction tuning raises disambiguated accuracy (+10.0 pp in the 3B, +8.3 pp in the 8B) but leaves ambiguous accuracy at 8–11 % (+3.4 and +0.1 pp), and the ambiguous bias score is higher (×2.1 and ×2.5). The jump of 43–52 pp in ambiguous accuracy appears only with A/B/C plus chat template, so it is not caused by instruction tuning alone. Chat template and letter format are not separated (instruct in A/B/C without chat template not run; decided not to run it, 2026-10-30). Caveats: one run per model; the 3B base row is from the older configuration. The base-model claim that low ambiguous accuracy reflects aversion to "unknown" is not contradicted, but the instruct models do not show it in the standard format either. Per-item logs: `checkpoints_bbq_standard_instruct/results/lm_evals/` (not opened). GPU and duration not recorded.
  - Practical consequence noted by Pere (inference, not tested): to act on an instruct model the neurons would have to be located with the chat template. Whether the located neurons differ with and without template has not been measured.
  - **Salamandra-7B base, A/B/C** (2026-10-30, task `bbq_base_mc`): overall accuracy 28.23 %, ambiguous 3.74 %, disambiguated 52.73 %, ambiguous bias 2.12 %, disambiguated bias 2.63 %. In the standard format it had 44.02 / 7.65 / 80.40 %. Disambiguated accuracy is close to the value of choosing between the two named people at random, and the overall accuracy is below the 33 % of choosing among three options at random, so the model does not use the A/B/C format as intended and its bias scores in this format are not interpretable. The effect of the format on accuracy seen in Llama-3.2-3B (under 1.5 pp) does not hold for this model. Whether the model favours one letter has not been checked (the per-item log is in `checkpoints_bbq_abc/results/lm_evals/`, not opened).
  - Sources: `base_models_bbq_abc_results_latest.csv` and `instruct_models_bbq_mc_results_latest.csv` (Drive: `fair_pruning/results/`).
- **Files:** [results/bias-benchmarks-base/README.md](results/bias-benchmarks-base/README.md) (all tables) · [custom_tasks/bbq_instruct/README.md](custom_tasks/bbq_instruct/README.md) · [custom_tasks/esbbq/README.md](custom_tasks/esbbq/README.md) · Drive: `fair_pruning/checkpoints_bbq/`, `checkpoints_mbbq/`, `checkpoints_bbq_instruct/`, `checkpoints_bbq_abc/`, `results/`. Per-item logs of the new models: `checkpoints_bbq/results/lm_evals/` (`meta_llama_llama_3.1_8b_bbq.json`, `bsc_lt_salamandra_7b_bbq.json`, about 700 MB each) and `checkpoints_mbbq/results/lm_evals/` (one `*_esbbq_<category>.json` per category and model; the Salamandra-7B and Llama-3.1-8B files were located by name and size, not opened)
- **Notebooks:** [02_Evaluate_BBQ](notebooks/02_Evaluate_BBQ.ipynb) (switch `PROMPT_FORMAT`: `standard`, `abc` (task `bbq_base_mc`) or `standard_instruct`) · [02_Evaluate_MBBQ](notebooks/02_Evaluate_MBBQ.ipynb) · [02_Evaluate_BBQ_Instruct](notebooks/02_Evaluate_BBQ_Instruct.ipynb). Each has an evaluation log (dates, versions, GPU) in its first cells.
- **Open items:**
  - [x] Acceptance: both new models are well above chance (disambiguated accuracy above 60 % in BBQ and EsBBQ; decision of 2026-10-29). The threshold itself was not written in the plan (it says "clearly above chance"); record it here if a formal value is wanted.
  - [x] GLU/OptiPFair compatibility (2026-10-29): both models report COMPATIBLE for pruning and for bias visualization (architecture `llama`, 32 layers, GLU structure; MLP expansion 350 % in Llama-3.1-8B and 269 % in Salamandra-7B).
  - [x] Per-item logs of the new models located on Drive (paths above).
  - [x] Qwen3-4B-Base (2026-10-29): compatible with OptiPFair (architecture `qwen3`, 36 layers, hidden size 2560, MLP expansion 380 %); BBQ and EsBBQ evaluated; per-item logs on Drive (`checkpoints_bbq/results/lm_evals/qwen_qwen3_4b_base_bbq.json` and `checkpoints_mbbq/results/lm_evals/qwen_qwen3_4b_base_esbbq_<category>.json`, located by name and size, not opened).
  - [x] Instruct question, design decided (2026-10-29): English BBQ only (EsBBQ in A/B/C is not done); destination an appendix or a short paragraph; base results are untouched and no intervention is run on instruct models. Pairs: Llama-3.2-3B base/Instruct and Salamandra-7B base/Instruct, all in A/B/C.
  - [x] Llama-3.2-3B base in A/B/C (2026-10-30), notebook `02_Evaluate_BBQ.ipynb` with `PROMPT_FORMAT = "abc"`; results in the table above.
  - [x] Salamandra-7B base in A/B/C (2026-10-30): result above; the model does not follow the format.
  - [x] Instruct models in the standard `bbq` format (2026-10-30): Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct run with `PROMPT_FORMAT = "standard_instruct"` (checkpoints `checkpoints_bbq_standard_instruct`, files `instruct_models_bbq_standard_results_*`, copied to `results/`); results in the table above. The Instruct block of E1 is closed; instruct in A/B/C without chat template is not run (it would only separate chat template from letter format).
  - [x] Llama-3.1-8B base and Instruct in A/B/C (2026-10-30), as second pair (a base that follows the format); results above. Not run: Salamandra-7B-Instruct (its base does not follow the format; would need a check that the Salamandra chat template is applied).
  - [x] A/B/C consolidated files copied to `results/` (2026-10-30): `base_models_bbq_abc_results_latest.csv` and `base_models_bbq_abc_results_20261030_102243.csv/json` (Llama-3.2-3B, Salamandra-7B, Llama-3.1-8B); `instruct_models_bbq_mc_results_latest.csv` and `instruct_models_bbq_mc_results_20261030_103101.csv/json` (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct). The JSON files were regenerated from the CSVs on Drive (same byte size as the Drive originals).
  - [x] Consolidated CSVs and JSONs of the five base models copied to the repository (2026-10-29): BBQ in `results/`, EsBBQ in `results/bias-benchmarks-base/`. The instruct CSV stays on Drive.
  - [ ] Commit and push what is pending.
  - Later (writing phase): the figures in `results/figures/` are from 2025-12-28 and show three models; [04_Graphics_Bias_BBQ](notebooks/04_Graphics_Bias_BBQ.ipynb) hard-codes the three models and reads old JSONs, so it needs a redesign, not just a rerun.
  - Moved: the Llama-3.2-3B baseline repeat to E2; the bootstrap intervals and the check of the 3B EsBBQ per-item files to E5.

## E2 — Random and cross controls (pilot in Llama-3.2-3B)

- **Status:** In progress (design closed 2026-11-02; S1, S2, S3 and S4 done on 2026-11-02/03; second identical baseline rerun and S5 (10 random sets) done 2026-11-03; next: S6, analysis). Step-by-step plan with gates: S0–S7 below.
- **Goal:** test the strongest result of the paper (Religion·Top-20, ambiguous bias 6.00 % → 1.50 %) against random neurons of the same layers, with a measure that also checks for damage.
- **Why this configuration:** scoping of the stored May 2026 runs (12 BBQ experiments in Llama-3.2-1B/3B and 16 EsBBQ in Llama-3.2-1B/Salamandra-2B), using the change in the magnitude of the bias score (|zeroed| − |baseline|) against the same change in the untouched categories. Religion·Top-20 in the 3B ranks first in both ambiguous (6.00 → 1.50) and disambiguated bias (3.84 → −1.67: −2.17 in magnitude, with a sign change). Salamandra-2B Religion·Top-40 (EsBBQ) also ranks first in both, but the model is at chance in disambiguated accuracy (~50 %), so its disambiguated bias is not interpretable. No run in the 1B is first in both measures. Caveat: the configuration was chosen after seeing the old-configuration results, so its effect may shrink on repetition.
- **Design:**
  - **Evaluation:** full BBQ, standard `bbq` task, current protocol (lm-eval 0.4.9.2, bfloat16, `batch_size=16`, 0-shot, `log_samples=True`). Scores: `en`, type `bias`.
  - **Real sets:** Religion·Top-20 and Race·Top-20, regenerated with the paper's rule (global top-N of `results/neuron_analysis/llama-3.2-3B/en/<Category>_bias_scores.json`, keys `down_proj_input_layer_N`, descending score). Checked on the stored scores: Religion has 14/20 neurons in layer 27 (the last) and one each in layers 17, 19, 21, 23, 24 and 26; Race has 18/20 in layer 27 and one each in layers 23 and 24; no tie at the cutoff.
  - **Random control:** 10 sets for Religion·Top-20 with fixed seeds 0–9 (10–19 reserved for an extension), `numpy.random.default_rng(seed)`; same number of neurons per layer as the real set, sampled without replacement from the 8192 neurons of each layer and excluding the real ones. Each manifest (layer → indices, seed, numpy version) is saved before evaluating.
  - **Noise control:** repeat the unpruned baseline with the current configuration (`checkpoints_bbq_rerun`) and compare aggregates (global and per category) with the old baseline. There is no per-item log of the old 3B baseline, so an item-level comparison is only possible between two identical reruns (optional; decide after timing S1).
  - **Cross control:** 2×2 matrix, Religion neurons and Race neurons evaluated on the items of both categories (it comes from the same full-BBQ runs). The May 2026 runs already contain this matrix for Top-20 in the old configuration (ambiguous bias: Religion neurons → Religion 1.50, Race 1.80; Race neurons → Religion 7.67, Race −1.16).
  - **Measure used in the scoping (the primary measure of the analysis is to be redefined with the results in view, 2026-11-03):** change in the magnitude of the target category's disambiguated bias score (|zeroed| − |baseline|; negative = less bias). **Target-category disambiguated accuracy** (computed from the per-item logs) is reported; its role in the analysis is not fixed (author, 2026-11-03: E2 looks for neurons that affect the target variable more than random ones, not for a lower bias or a higher accuracy). **Stability panel** (from the logs): flips against the baseline (correct → incorrect, incorrect → correct, towards/against the stereotype), rate of "unknown" answers, KL over the three options (needs the option loglikelihoods in the logs; to verify in S1), effect on the other categories, WikiText per set. **Ambiguous scores:** descriptive only (author, 2026-11-03: the base models are at the floor in ambiguous accuracy, so those scores are not used as reference; an Instruct-model test would need the prompts prepared first).
  - **Logs:** baseline rerun in `checkpoints_bbq_rerun`. Evaluation notebook: [09_E2_Controls_Llama3B](notebooks/09_E2_Controls_Llama3B.ipynb) (`PHASE = "real"` or `"random"`). BBQ checkpoints `checkpoints_e2/llama_3.2_3B/<set_id>_bbq.json`; per-item BBQ logs `checkpoints_e2/results/lm_evals/llama_3.2_3b__<set_id>_bbq.json` (about 700 MB each; `utils` names them after `config._name_or_path`, which the notebook sets per set so that sets do not overwrite each other); capabilities checkpoints `checkpoints_e2_cap/llama_3.2_3B/<set_id>_cap.json`. The notebook writes a summary per set to Drive (`fair_pruning/results/e2/per_set/<set_id>_summary.json`: BBQ by category with flips against the baseline, raw capabilities, neurons, environment, sha256 of the manifest) as soon as each set finishes, and the consolidated CSV/JSON to `fair_pruning/results/e2/`.
  - **Capabilities per set:** baseline and real sets with the battery of 07B (WikiText, MMLU 5-shot, ARC-Challenge, HellaSwag, HellaSwag-es); random sets WikiText only (switch `CAPS_RANDOM`). WikiText is evaluated with `batch_size=1` (with 16 it ran out of memory on an A100-80GB trying to allocate 46.7 GiB); the rest keep `batch_size=16`.
  - **Decision B (2026-11-02):** no numeric gate is fixed for S3 beyond the direction check; results are reviewed jointly before the random sets are run.
- **Decision criterion:** the criterion written on 2026-11-02 is superseded by decision of the author (2026-11-03). The measures and criteria of the analysis will be defined with the results of all sets in view.
- **Scope:** it allows saying that the selected neurons have more effect than random neurons **of the same layers**. It does not show that the effect is demographic-specific: layer-matched random sets include last-layer scale effects (E0), so a magnitude-matched control and non-demographic control pairs would be needed.
- **Run steps (each with a gate; do not move on until it is met):**
  - **S0.** Record the decisions here; file the scoping script. Gate: done.
  - **S1.** Noise baseline of the 3B in `checkpoints_bbq_rerun` (`02_Evaluate_BBQ.ipynb`, `PROMPT_FORMAT = "standard_rerun"`, Llama-3.2-3B only). Gate: GPU, duration and versions recorded in the notebook log; per-item log opened (option loglikelihoods present; item counts per category).
    - **Result (run 9, 2026-11-02):** A100-SXM4-40GB, lm-eval 0.4.9.2, torch 2.10.0, transformers 4.57.3, 39 min 11 s (10:08:57 to 10:48:08); Drive `fair_pruning/checkpoints_bbq_rerun/`; number of requests not recorded. Overall / ambiguous / disambiguated accuracy 40.51 / 7.48 / 73.53 % (old run: 40.52 / 7.51 / 73.54 %); aggregate bias score ambiguous 4.12 → 4.03, disambiguated 2.24 → 2.37.
    - **Bias score per category, old run → rerun (pp, ambiguous / disambiguated):** Age 16.30 → 15.65 / 8.48 → 8.91; Disability status 4.63 → 4.11 / 0.51 → 0.51; Gender identity 10.86 → 11.00 / 8.04 → 8.25; Nationality 4.74 → 4.61 / 3.52 → 3.79; Physical appearance 17.89 → 17.77 / 9.02 → 8.67; Race/ethnicity 1.34 → 0.76 / 1.34 → 0.99; Race × gender 1.40 → 1.30 / 0.50 → 0.55; Race × SES 1.52 → 1.77 / −0.69 → 0.06; Religion 6.00 → 6.00 / 3.84 → 3.84; SES 1.37 → 1.57 / 0.17 → 0.29; Sexual orientation 4.86 → 4.17 / 0.93 → 0.93.
    - **Reading:** the largest difference is 0.75 pp (Race × SES, disambiguated, which crosses zero), the signs are mixed, and Religion (target category of E2) is identical in both contexts. The cause is not attributed: batch size (8 → 16), library versions and possibly the GPU changed between the runs. Per-category standard errors are N/A in the lm-eval aggregates (to be computed from the per-item log). Results of the May 2026 zeroed runs are therefore not strictly comparable with new runs in the categories that move (Race, Race × SES, Age, Sexual orientation); the new baseline is the reference for all E2 comparisons.
    - **Per-item log opened and validated (2026-11-02):** `checkpoints_bbq_rerun/results/lm_evals/meta_llama_llama_3.2_3b_bbq.json` (about 700 MB): 58,492 items in a single group `bbq`, with category, context condition, question polarity and the loglikelihood of 12 options per item (the two non-"unknown" answers, in the order of the item, and ten phrasings of "unknown"; the "unknown" answer of the item can be `ans0`, `ans1` or `ans2`). Prediction = argmax of the 12; accuracy per category recomputed from the items; bias recomputed with the BBQ formula (disambiguated: 2 * biased / non-unknown - 1; ambiguous: the same multiplied by 1 - ambiguous accuracy). Check against lm-eval's own per-item fields: 0 mismatches in 58,492 items, and the bias of the 11 categories equals lm-eval's to two decimals. Gate met.
    - **Baseline per category recomputed from the items (rerun; accuracy ambiguous / disambiguated, % of "unknown" answers in disambiguated items):** Age 0.00 / 78.80 (0.00); Disability status 0.00 / 68.25 (0.00); Gender identity 0.00 / 79.62 (0.00); Nationality 5.26 / 79.55 (0.52); Physical appearance 2.28 / 70.81 (1.90); Race/ethnicity 0.00 / 78.05 (0.00); Race × SES 36.79 / 50.39 (37.67); Race × gender 0.00 / 80.66 (0.00); Religion 3.67 / 79.00 (0.17); SES 0.29 / 79.75 (0.00); Sexual orientation 0.93 / 78.24 (0.00). Items per category and context: Religion 600, Sexual orientation 432, Physical appearance 788, Disability status 778, Nationality 1540, Age 1840, Gender identity 2836, SES 3432, Race/ethnicity 3440, Race × SES 5580, Race × gender 7980 (one Religion item = 0.17 pp of accuracy).
    - **Reading:** in ambiguous items the correct answer is always "unknown", so ambiguous accuracy equals the rate of "unknown" answers: the model almost never chooses it (0.00 % in four categories, 3.67 % in Religion). Race × SES is the exception (about 37 % of "unknown" answers in both contexts, disambiguated accuracy 50.39 %): its bias rests on a smaller share of answers and it is the category that moved most between runs; cause not investigated. The loglikelihoods are stored in bfloat16 resolution. Details of the log format and the metric definitions: `claude/BBQ_log_format.md` (project documents).
    - **Second identical rerun (2026-11-03, `checkpoints_bbq_rerun2`, `02_Evaluate_BBQ.ipynb`, same configuration; A100; BBQ scoring of 701,904 requests in 31 min 57 s):** compared item by item with the first rerun (58,492 items, 12 loglikelihoods each; script run in Colab, result in Drive `fair_pruning/results/e2/rerun_comparison.json`): 58,492 items (100.00 %) have identical loglikelihoods bit for bit, 0 items change the argmax over the 12 options, 0 change the person 0 / person 1 / unknown prediction; max and mean absolute difference 0. Both runs were done in different Colab sessions, so they cover the change of machine and driver between sessions (the GPU model of each session was not recorded; the BBQ of the real sets of S3 ran on an A100-SXM4-80GB and the first rerun on an A100-SXM4-40GB). Aggregates equal to the first rerun (40.51 / 7.48 / 73.53 %; ambiguous bias 4.03).
  - **S2.** Regenerate the manifests of Religion·Top-20 and Race·Top-20 with the paper's rule. First look for the original `llama-3.2-3B_neuron_manifest.json` in the `RESULTS_DIR` of notebook 07B on Drive. Gate: layer distribution as above.
    - **Done 2026-11-02.** The original `llama-3.2-3B_neuron_manifest.json` is not on Drive (searched by name; the `results/` folder holds none). Regenerated from the stored scores with the paper's rule (script `e2_make_manifest.py`, not yet filed in the repository). Manifest: Drive `fair_pruning/results/llama-3.2-3B_E2_manifest.json` (real sets with score and layer of each neuron, `by_layer` indices in the format of `zero_neurons_mlp`, sha256 of the score files, numpy version, checks).
    - It reproduces the May printout of 07B in layers, top-1 neuron and score, and gap 1→2: Religion·Top-20 layers 17, 19, 21, 23, 24, 26, 27 (14 in layer 27), top-1 L27 neuron 1348 (0.690583); Race·Top-20 layers 23, 24, 27 (18 in layer 27), top-1 L27 neuron 3313 (0.767721). The full 20-neuron list of May cannot be compared because that manifest was not saved. Religion·Top-20 and Race·Top-20 share 5 neurons, so the 2×2 cross is not between disjoint sets.
  - **S3.** Evaluate the two real interventions in the current configuration (full BBQ, per-item logs). Existing results for comparison: [results/bias-benchmarks-zeroed/](results/bias-benchmarks-zeroed/) (`llama-3.2-3B_bbq_zeroed_results.*`). Gate: the direction of the May effect is reproduced; if it disappears, stop and record it as a finding.
    - **Done 2026-11-02** (notebook 09, `PHASE = "real"`; Drive `checkpoints_e2`, `checkpoints_e2_cap`). BBQ and MMLU/ARC/HellaSwag on an A100-SXM4-80GB (torch 2.10.0+cu128, transformers 4.57.3, lm-eval 0.4.9.2, optipfair 0.5.0; about 40 min of BBQ per set from the checkpoint timestamps; whole phase 2 h 28 min). WikiText was re-run on 2026-11-03 on an NVIDIA L4 with `batch_size=1` (4 min 43 s for the three sets, model cached) after the first attempt failed with CUDA out of memory in all three sets; BBQ/capabilities of the three sets and WikiText therefore come from different GPUs. Baseline BBQ = the rerun of S1 (not repeated).
    - **Gate:** the direction of the May effect is reproduced (Religion·Top-20 lowers both ambiguous and disambiguated Religion bias); the size differs from May (see below).
    - **BBQ bias score (pp; baseline → Religion·Top-20 / Race·Top-20):** Religion ambiguous 6.00 → 2.67 / 7.67; Religion disambiguated 3.84 → −1.33 / 1.00; Race/ethnicity ambiguous 0.76 → 1.92 / −0.81; Race/ethnicity disambiguated 0.99 → 2.03 / 2.73. Aggregate bias ambiguous 4.03 → 4.19 / 4.45, disambiguated 2.37 → 2.41 / 2.93.
    - **Accuracy (%, baseline → Religion·Top-20 / Race·Top-20):** overall 40.51 → 39.80 / 39.82; pooled ambiguous 7.48 → 6.15 / 6.78; pooled disambiguated 73.53 → 73.44 / 72.86. Religion ambiguous 3.67 → 2.00 / 2.00 (10 items); Religion disambiguated 79.00 → 76.67 / 77.83 (−2.33 / −1.17); Race/ethnicity disambiguated 78.05 → 76.42 / 76.42.
    - **Changed answers against the baseline (percentage of items whose prediction among person 0 / person 1 / unknown differs):** between 4.05 % and 12.48 % in every category for both sets, including categories unrelated to either set (e.g. Religion·Top-20: Disability 5.98, Sexual orientation 4.51, Race × gender 10.77; Religion 11.08; Race·Top-20: Religion 7.42, Race/ethnicity 12.46). Two identical baseline runs are bit-for-bit equal (see S1), so these changes are not run-to-run noise.
    - **Capabilities (baseline / Religion·Top-20 / Race·Top-20):** WikiText word perplexity 9.5344 / 9.7939 / 9.7773 (+0.26 / +0.24; the 07B baseline was 9.5372); MMLU 58.06 / 57.75 / 56.76; ARC-Challenge acc_norm 46.33 / 45.82 / 45.56; HellaSwag acc_norm 74.04 / 73.75 / 73.79; HellaSwag-es acc_norm 58.95 / 58.60 / 58.76.
    - **Reading (no claim yet; random sets pending):** (i) Religion·Top-20 lowers Religion disambiguated bias more than Race·Top-20 does (−5.17 vs −2.84 pp signed), but Race·Top-20 also moves it, and with 600 Religion items per context the sampling error of that bias score is large (rough estimate about 4–5 pp; not computed from the logs); (ii) Race·Top-20 does not lower Race/ethnicity bias (disambiguated +1.74 pp); (iii) the ambiguous change in Religion does not come from more "unknown" answers (ambiguous accuracy falls 3.67 → 2.00); (iv) Religion·Top-20 costs 2.33 pp of Religion disambiguated accuracy (14 items); (v) both sets cost about 0.2–0.3 WikiText perplexity points.
    - **Against May 2026 (stored runs, whose neuron lists were not saved):** Race·Top-20 on Religion ambiguous gives 7.67, the same value as in May; Religion·Top-20 gives 2.67 for Religion ambiguous (May 1.50) and −1.33 for disambiguated (May −1.67). The regenerated Religion·Top-20 set may differ from the May one, or the difference may come from the configuration (the May runs used another batch size and library versions; identical runs in the current configuration are bit-for-bit equal, so it is not run-to-run variation within one configuration); it cannot be told apart with the available data. The paper figure 6.00 → 1.50 is not reproduced in this configuration.
  - **S4.** Generate and save the 10 random manifests (seeds 0–9) before evaluating. Gate: same per-layer counts as the real set and no overlap with it.
    - **Done 2026-11-02** (generated together with S2, same script and manifest, before any evaluation): `rand_religion_seed0` … `rand_religion_seed9`, `numpy.random.default_rng(seed)`, layers in ascending order, `Generator.choice(..., replace=False)` over the 8192 neurons of each layer excluding only the real neurons of the set being mimicked (decision of 2026-11-02: overlaps with the other category's set are allowed). Checks passed: 20 distinct neurons per set, no overlap with Religion·Top-20, same layer distribution (14 in layer 27, one each in 17, 19, 21, 23, 24, 26); maximum overlap between two random sets: 1 neuron. The script was also run twice in the sandbox with the repository scores and gave identical manifests.
  - **S5.** Evaluate the 10 random sets (full BBQ, per-item logs, WikiText). Gate: 10 complete runs.
    - **Done 2026-11-03** (notebook 09, `PHASE = "random"`, seeds 0–9; Drive `checkpoints_e2`, `checkpoints_e2_cap`; one summary per set in `fair_pruning/results/e2/per_set/rand_religion_seed<N>_summary.json`; manifest sha256 e633b604…c0de in all ten). Environment: torch 2.10.0+cu128, transformers 4.57.3, lm-eval 0.4.9.2, optipfair 0.5.0, `batch_size=16`, WikiText `batch_size=1`.
    - **GPU:** seeds 2–9 on an A100-SXM4-40GB (about 41 min of BBQ per set from the checkpoint timestamps). Seeds 0 and 1 started on an NVIDIA L4 (seed 0: 10:48 to 12:34 UTC, about 1 h 46 min, as recorded in its first summary; seed 1: BBQ checkpoint at 14:22 UTC, same pace) and the run was then moved to the A100. The summaries of seeds 0 and 1 were rewritten at 14:59 UTC in the A100 session, so their `env.gpu` field says A100-SXM4-40GB although their BBQ ran on the L4 (inferred from the timestamps; the GPU of the WikiText of seed 1 is not known). The `env` field of a summary describes the session that wrote it, not the one that evaluated. Real sets: BBQ on an A100-SXM4-80GB; baseline: A100-SXM4-40GB.
    - **Religion, random sets (pp for bias and accuracy; % of changed answers against the baseline; WikiText word perplexity):**

      | set | amb. bias | disamb. bias | amb. acc. | disamb. acc. | changed | WikiText |
      |---|---|---|---|---|---|---|
      | baseline | 6.00 | 3.84 | 3.67 | 79.00 | – | 9.5344 |
      | seed 0 | 6.67 | 2.84 | 3.67 | 78.17 | 1.58 | 9.5388 |
      | seed 1 | 4.67 | 3.84 | 3.67 | 78.67 | 1.67 | 9.5359 |
      | seed 2 | 7.00 | 3.17 | 3.67 | 79.00 | 1.08 | 9.5331 |
      | seed 3 | 7.33 | 3.51 | 3.67 | 79.50 | 1.25 | 9.5322 |
      | seed 4 | 6.33 | 4.17 | 3.67 | 78.17 | 1.33 | 9.5458 |
      | seed 5 | 5.33 | 3.84 | 3.67 | 79.00 | 1.17 | 9.5337 |
      | seed 6 | 6.00 | 3.17 | 3.67 | 79.00 | 1.33 | 9.5337 |
      | seed 7 | 7.33 | 4.17 | 3.67 | 78.50 | 1.08 | 9.5343 |
      | seed 8 | 6.00 | 2.84 | 3.67 | 79.17 | 1.25 | 9.5401 |
      | seed 9 | 5.00 | 3.51 | 3.67 | 79.17 | 1.17 | 9.5344 |
      | random mean ± SD (n = 10) | 6.17 ± 0.95 | 3.51 ± 0.50 | 3.67 ± 0 | 78.84 ± 0.44 | 1.29 ± 0.20 | 9.5362 |
      | random range | 4.67–7.33 | 2.84–4.17 | 3.67 | 78.17–79.50 | 1.08–1.67 | 9.5322–9.5458 |
      | Religion·Top-20 | 2.67 | −1.33 | 2.00 | 76.67 | 11.08 | 9.7939 |
      | Race·Top-20 | 7.67 | 1.00 | 2.00 | 77.83 | 7.42 | 9.7773 |

    - **Changed answers in all categories:** random sets 0.26–4.19 % per category and seed (the next highest value after Physical appearance in seed 4, 4.19 %, is 2.60 %); real sets 4.05–12.48 %.
    - **Reading (descriptive; no analysis criterion applied yet):** the ten random sets keep the Religion ambiguous accuracy at the baseline value (3.67 %, identical in all); the real sets lower it to 2.00 %. SD is the sample SD of 10 sets. Seeds 0 and 1 (L4) have the two highest Religion changed-answer rates among the random sets (1.58 and 1.67 vs 1.08–1.33 in seeds 2–9), consistent with a small GPU effect on this measure; with n = 2 against 8 this is only an indication. Measures and criteria of the analysis: to be decided (see S6).
  - **S6.** Analysis of all sets and stability panel. Measures and criteria to be defined with the results in view (author, 2026-11-03); the script is discussed before it is written.
  - **S7.** Record the outcome here and decide the next step (E3 or other).
- **Data checks open:**
  - Llama-3.2-1B BBQ `exp2` is named Top-20 but zeroes 5 neurons (the EsBBQ file calls the same experiment Top-5).
  - Llama-3.2-3B baseline (`results/bias-benchmarks-base/meta_llama_llama_3.2_3b.json`): ambiguous and disambiguated Race_ethnicity bias scores are identical to 15 digits (0.013372093023255704). In the rerun of 2026-11-02 they differ (0.0076 and 0.0099); the cause of the equality in the old file is not explained.
  - Paper, Table VIII: −5.51 pp is the signed change of the disambiguated bias of Religion (3.84 → −1.67); in magnitude it is −2.17.
- **Notebooks to reuse:** [06_zero_bias_neurons](notebooks/06_zero_bias_neurons.ipynb) (zeroing) · [07_EvalPrunedModels_Llama3B](notebooks/07_EvalPrunedModels_Llama3B.ipynb) and [07B](notebooks/07B_EvalPrunedModels_Llama3B.ipynb) (evaluation pattern). Neuron scores: `results/neuron_analysis/llama-3.2-3B/`.
- **Open items:**
  - [x] (done 2026-11-02, S1; per-item log opened and validated) Repeat the Llama-3.2-3B BBQ baseline with the current configuration in `checkpoints_bbq_rerun` (per-item logs; it is the noise control of this experiment).
  - [x] (regenerated and saved 2026-11-02, S2; original not found) Neuron indices of the published Religion·Top-20 run: [results/bias-benchmarks-zeroed/README.md](results/bias-benchmarks-zeroed/README.md) describes a `llama-3.2-1B_neuron_manifest.json`, but no manifest file is in the repository and none was found on Drive (searched by name). For the 3B, regenerate the set from `results/neuron_analysis/llama-3.2-3B/` with the same selection rule as the paper and save the manifest this time.
  - [x] (done 2026-11-03) Second identical baseline rerun: bit-for-bit equal to the first one (0 of 58,492 items change; see S1). The items that change in S3 are an effect of the intervention.
  - [ ] Decide how the 12 option loglikelihoods are collapsed to three options for the KL (proposal in `claude/BBQ_log_format.md`).
  - [ ] Define the measures and criteria of the E2 analysis (S6) with the results of all sets in view; discuss the script before writing it.
  - [ ] (future, author's idea 2026-11-03) Test with an Instruct model, which requires preparing the prompts first; ambiguous scores of the base models are not used as reference.
  - [ ] Colab environment: the first re-execution failed with `ImportError: GenerationMixin` after `pip install -r requirements.txt`; fixed on 2026-11-03 by changing the numpy line in `requirements.txt` (commits bfb30ac, 6b777af, ad70a34). Check that the change does not affect the other notebooks.

## E3 — Intervention and controls in the new models

- **Status:** Pending. Depends on E1 (model choice) and E2 (whether the causal claim holds).
- **Goal:** repeat the design validated in E2 in Llama-3.1-8B and Salamandra-7B.
- **Design:** BiasScore localisation and zeroing of the top-K neurons, saving the exact neuron manifest. Reduced grid: 2 categories (for example Religion and Race) and 2 small values of K. Random controls (10 sets with the same layer distribution) only for the main K, plus the cross matrix between the two categories. Per-item logs are saved (they feed E4 and E5).
- **Left for a second iteration:** the 5 full categories and the complete cross comparison, depending on compute.
- **Run steps:**
  1. Compute BiasScores for each new model ([03_neuron_bias_detection_en](notebooks/03_neuron_bias_detection_en.ipynb)); first pass the GLU compatibility check.
  2. Select top-K neurons per category and save the manifests.
  3. Evaluate the real interventions, then the random and cross controls, as in E2.
- **Open items:**
  - [ ] Exact K values (take them from the pilot models; to define).
  - Plan: same K values as the rest of the models; Salamandra-7B in English and Spanish (as decided 2026-10-29, to confirm when E3 starts).

## E4 — Capabilities

- **Status:** Pending.
- **Goal:** check that the interventions do not degrade general capabilities, in every configuration the paper cites and in the new ones.
- **Design:** the paper's battery (WikiText, MMLU, ARC-Challenge, HellaSwag, EN/ES). Random sets of E2/E3: WikiText only; the full battery only for the real configurations. Per-item (per-document in WikiText) predictions are saved so retention can be computed with bootstrap intervals instead of the mean of ratios.
- **Missing today:**
  - Religion·Top-20 in Llama-3.2-3B (the strongest result, with no capability evaluation).
  - Salamandra.
  - The E3 configurations in the new models, plus their capability baseline.
  - K=40 stays out.
- **Files and notebooks:** [02_Evaluate_Base_Capabilities](notebooks/02_Evaluate_Base_Capabilities.ipynb) (baselines) · [07B_EvalPrunedModels_Llama3B](notebooks/07B_EvalPrunedModels_Llama3B.ipynb) (zeroed) · [results/capabilities_zeroed/](results/capabilities_zeroed/)
- **Open items:**
  - [ ] Capability baselines for Llama-3.1-8B and Salamandra-7B (not run yet).

## E5 — Statistics (bootstrap)

- **Status:** Pending.
- **Goal:** confidence intervals for the bias changes, in Llama-3.2-3B and the new models.
- **Design:** paired bootstrap over items (same items before and after the intervention, bias recomputed at each resample), from the per-item logs of E1–E3 with no extra evaluations. A table with the number of items per category and context. For capabilities, lm-eval's standard error.
- **Per-item logs available today:**
  - BBQ: found on Drive for Llama-3.1-8B and Salamandra-7B (about 700 MB each). No per-item BBQ file found for Llama-1B, Salamandra-2B or Llama-3B (3B to be repeated in E1).
  - EsBBQ: found on Drive for Llama-3.2-3B (Feb 2026), Llama-3.2-1B and Salamandra-2B (May 2026, one file per category), and Llama-3.1-8B and Salamandra-7B (Oct 2026). Contents not yet verified.
  - Zeroed runs: not verified.
- **Open items:**
  - [ ] Open one of the Llama-3.2-3B EsBBQ per-item JSONs (Drive, Feb 2026) and check whether it holds the 10 tasks or only its own category: the five files listed have exactly the same size (246,871,379 bytes), which suggests each one holds all tasks, but this is not verified. Also open one Salamandra-7B and one Llama-3.1-8B file to confirm that per-item predictions are present.
  - [ ] Bootstrap confidence intervals for the new-model baselines.
  - [ ] Design and discuss the bootstrap script before writing it (number of resamples, handling of the per-category aggregation).
  - [ ] Decide whether the old Llama-1B and Salamandra-2B results are repeated or presented as descriptive.

## E6 — DPO pilot on selected neurons

- **Status:** Not planned for P0.
- **Idea (kept for P1 or a secondary paper):** train only the localised neurons with DPO, rest of the model frozen, on a synthetic dataset (no BBQ) of biased and unbiased responses; minimal pilot in one model, one category, K=20, compared with random neurons of the same layers; measure BBQ bias (trained and other categories), capabilities and abstention.

## Changelog

- 2026-11-03: E2 S5 done (10 random Religion sets, seeds 0–9; table in S5). The decision criterion of 2026-11-02 is superseded by decision of the author; measures and criteria of the analysis to be defined with the data. The `env.gpu` field of the summaries of seeds 0 and 1 does not correspond to the GPU of their BBQ (L4).
- 2026-11-03: E2 second identical baseline rerun done: bit-for-bit equal to the first one (0 of 58,492 items change), so the 4–12 % of changed answers in S3 is not run-to-run noise.
- 2026-11-03: E2 S3 done (real sets Religion·Top-20 and Race·Top-20 with full BBQ and capabilities; WikiText re-run with `batch_size=1` after an out-of-memory failure); the 4–12 % of changed answers in every category needs a second identical baseline rerun; Colab import error fixed through `requirements.txt`; notebook 09 now saves per-set summaries to Drive.
- 2026-11-02: E2 S1 done (noise baseline of the 3B in `checkpoints_bbq_rerun`: aggregates and Religion equal to the old run, other categories differ by up to 0.75 pp; per-item log opened and its recomputation validated against lm-eval: 0 mismatches in 58,492 items) and S2/S4 done (manifest with Religion·Top-20, Race·Top-20 and 10 random sets saved on Drive).
- 2026-11-02: E2 design closed (Religion·Top-20 in Llama-3.2-3B, full BBQ, fixed seeds 0–9; primary measure = change in the magnitude of the disambiguated bias with disambiguated accuracy as validity condition; decision criterion fixed); no run yet.
- 2026-11-02: E1 closed.
- 2026-10-30: Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct evaluated in the standard `bbq` format: ambiguous accuracy stays at 8–11 %, disambiguated +8 to +10 pp, ambiguous bias higher; earlier wording that attributed the A/B/C gain to instruction tuning corrected.
- 2026-10-30: Llama-3.1-8B base and Instruct evaluated in A/B/C: the format raises the base's ambiguous accuracy 8.35 → 38.28 % (it did not in the 3B); instruct tuning adds a further 43 pp.
- 2026-10-30: Salamandra-7B base evaluated in A/B/C: it does not follow the format (disambiguated accuracy 80.4 → 52.7 %). No decision yet on further pair runs.
- 2026-10-30: A/B/C pair study: task `bbq_base_mc` and `PROMPT_FORMAT` switch added; Llama-3.2-3B base evaluated in A/B/C (accuracy unchanged by the format, bias score ×2.8); Salamandra-7B base running.
- 2026-10-29: Qwen3-4B-Base added and evaluated (BBQ, EsBBQ); instruct-model question reopened; Llama-3.1-8B and Salamandra-7B compatible with OptiPFair; per-item logs located; file created.
- 2026-10-28: E1 baselines finished for Llama-3.1-8B and Salamandra-7B (BBQ, EsBBQ) and Llama-3.2-3B-Instruct (A/B/C format); READMEs and evaluation logs updated.
- 2026-10-27: E0 documented in `docs/`; Llama-3.1-8B BBQ baseline run.
