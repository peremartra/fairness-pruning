# R1 — Experiments tracker and run guide (P0: Fairness Pruning)

Last updated: 2026-10-29

Purpose: one place to see the status of each experiment (E0–E6) and to consult before launching the next one. Numbers are kept to the minimum needed to read the status; the full tables live in the linked files.

How to maintain it: when something changes, (1) update the row in the Status table, (2) edit only the affected experiment section, (3) add a dated line to the Changelog at the bottom.

Status values: **Done** · **In progress** · **Pending** · **Not planned**

## Status

| Exp | What | Status | Models | Headline | Section |
|---|---|---|---|---|---|
| E0 | Last-layer checks on stored scores | Done | Llama-3.2-1B, 3B, Salamandra-2B | Last-layer share in top-1 %: 34–71 % raw, 12–31 % after per-layer normalisation | [E0](#e0--last-layer-checks) |
| E1 | New models and baselines | Done (commit pending) | Llama-3.1-8B, Salamandra-7B | Disambiguated accuracy above 60 % in BBQ and EsBBQ; both OptiPFair-compatible | [E1](#e1--new-models-and-baselines) |
| E2 | Random and cross controls (pilot) | Pending | Llama-3.2-3B | Not started | [E2](#e2--random-and-cross-controls-pilot-in-llama-32-3b) |
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

- **Status:** Done (2026-10-29). Baselines finished on 2026-10-28; the data files are copied to the repository and only the commit is pending; the figures are left for the writing phase.
- **Goal:** add Llama-3.1-8B and Salamandra-7B, and check that their baselines are clearly above chance. An instruct model (Llama-3.2-3B-Instruct) was evaluated and discarded for P0 (2026-10-29). Per-item logs are reused later for bootstrap (E5). Qwen and Gemma were discarded; the line stays with Llama and Salamandra.
- **Design (as run):** BBQ (English) and EsBBQ (Spanish), 0-shot, full dataset, bfloat16, `batch_size=16`, `log_samples=True`. Llama-3.2-3B-Instruct was also evaluated, in English only, with a different task (`bbq_instruct_mc`: options A/B/C, chat template); its numbers are not comparable with the base-model ones and it is not used in P0. The position-bias diagnostic of the plan (is the ~50 % disambiguated accuracy of Llama-3.2-1B and Salamandra-2B in EsBBQ position bias or chance?) is dropped (2026-10-29); it will be run only if a reviewer asks. It would need the per-item logs of those two models, which exist on Drive.
- **Results** (item-weighted; reference for "above chance": about 50 % in disambiguated items is random choice between the two named answers):

| Model | BBQ amb / disamb / S | EsBBQ amb / disamb / S |
|---|---|---|
| Salamandra-2B | 7.7 / 50.5 / 2.2 | 24.4 / 49.8 / 2.0 |
| Llama-3.2-1B | 10.0 / 52.3 / 1.9 | 23.5 / 49.9 / 1.4 |
| Llama-3.2-3B | 7.5 / 73.5 / 4.5 | 14.8 / 70.0 / 6.2 |
| Llama-3.1-8B | 8.4 / 82.0 / 6.1 | 13.5 / 74.2 / 8.8 |
| Salamandra-7B | 7.7 / 80.4 / 9.2 | 8.8 / 73.4 / 9.1 |

  - Low ambiguous accuracy in every model is an aversion to the "unknown" option, not an inability: disambiguated accuracy is high.
  - EsBBQ values are reconstructed from category-level results (about ±0.1 pp); exact values will come from the per-item logs.
- **Instruct side note** (A/B/C format, not comparable with the table above): ambiguous accuracy 59.0 %, disambiguated 78.5 %, S 17.8 %; kept in the README only.
- **Files:** [results/bias-benchmarks-base/README.md](results/bias-benchmarks-base/README.md) (all tables) · [custom_tasks/bbq_instruct/README.md](custom_tasks/bbq_instruct/README.md) · [custom_tasks/esbbq/README.md](custom_tasks/esbbq/README.md) · Drive: `fair_pruning/checkpoints_bbq/`, `checkpoints_mbbq/`, `checkpoints_bbq_instruct/`, `results/`. Per-item logs of the new models: `checkpoints_bbq/results/lm_evals/` (`meta_llama_llama_3.1_8b_bbq.json`, `bsc_lt_salamandra_7b_bbq.json`, about 700 MB each) and `checkpoints_mbbq/results/lm_evals/` (one `*_esbbq_<category>.json` per category and model; the Salamandra-7B and Llama-3.1-8B files were located by name and size, not opened)
- **Notebooks:** [02_Evaluate_BBQ](notebooks/02_Evaluate_BBQ.ipynb) · [02_Evaluate_MBBQ](notebooks/02_Evaluate_MBBQ.ipynb) · [02_Evaluate_BBQ_Instruct](notebooks/02_Evaluate_BBQ_Instruct.ipynb). Each has an evaluation log (dates, versions, GPU) in its first cells.
- **Open items:**
  - [x] Acceptance: both new models are well above chance (disambiguated accuracy above 60 % in BBQ and EsBBQ; decision of 2026-10-29). The threshold itself was not written in the plan (it says "clearly above chance"); record it here if a formal value is wanted.
  - [x] GLU/OptiPFair compatibility (2026-10-29): both models report COMPATIBLE for pruning and for bias visualization (architecture `llama`, 32 layers, GLU structure; MLP expansion 350 % in Llama-3.1-8B and 269 % in Salamandra-7B).
  - [x] Per-item logs of the new models located on Drive (paths above).
  - [x] Instruct model discarded for P0.
  - [x] Consolidated CSVs and JSONs of the five base models copied to the repository (2026-10-29): BBQ in `results/`, EsBBQ in `results/bias-benchmarks-base/`. The instruct CSV stays on Drive.
  - [ ] Commit and push.
  - Later (writing phase): the figures in `results/figures/` are from 2025-12-28 and show three models; [04_Graphics_Bias_BBQ](notebooks/04_Graphics_Bias_BBQ.ipynb) hard-codes the three models and reads old JSONs, so it needs a redesign, not just a rerun.
  - Moved: the Llama-3.2-3B baseline repeat to E2; the bootstrap intervals and the check of the 3B EsBBQ per-item files to E5.

## E2 — Random and cross controls (pilot in Llama-3.2-3B)

- **Status:** Pending.
- **Goal:** test the strongest result of the paper (Religion·Top-20, ambiguous bias 6.00 % → 1.50 %) against random neurons.
- **Design:**
  - **Real set:** the Religion Top-20 neurons already used in the paper (indices to be recomputed or recovered, see open items).
  - **Random control:** 10 sets with the same number of neurons per layer as the real set. Extend only if the result justifies it.
  - **Noise control:** repeat the unpruned baseline with the same settings (this is the baseline repeat of E1) and compare it item by item with the first run.
  - **Cross control:** 2×2 matrix, Religion neurons and Race neurons evaluated on the BBQ items of both categories.
  - **Measures:** change in ambiguous and disambiguated bias, accuracy of the category, and WikiText as capability cost.
- **Decision criterion (fixed before seeing results):** if the real effect is clearly outside the distribution of the random sets, move on to the new models (E3). If it falls inside, rethink the central claim.
- **Scope:** it allows saying that the selected neurons have more effect than random neurons **of the same layers**. It does not show that the effect is demographic-specific: layer-matched random sets include last-layer scale effects (E0), so a magnitude-matched control and non-demographic control pairs would be needed.
- **Run steps:**
  1. Reproduce the real intervention under the current configuration and compare it with the published values. Existing results: [results/bias-benchmarks-zeroed/](results/bias-benchmarks-zeroed/) (`llama-3.2-3B_bbq_zeroed_results.*`).
  2. Run the noise control (new checkpoint directory).
  3. Generate the 10 random sets, record the seeds and save each set as a manifest (layer → neuron indices) before evaluating.
  4. Evaluate each set on BBQ with per-item logs; compute category bias, category accuracy and WikiText.
  5. Run the 2×2 cross matrix.
  6. Apply the decision criterion and record the outcome here.
- **Design questions to settle before writing code:**
  - Full BBQ or only the target categories (cost versus comparability with the published numbers).
  - The numerical rule for "clearly outside the random distribution" (for example rank among the 11 values), fixed in advance.
  - Seeds for the random sets.
- **Notebooks to reuse:** [06_zero_bias_neurons](notebooks/06_zero_bias_neurons.ipynb) (zeroing) · [07_EvalPrunedModels_Llama3B](notebooks/07_EvalPrunedModels_Llama3B.ipynb) and [07B](notebooks/07B_EvalPrunedModels_Llama3B.ipynb) (evaluation pattern). Neuron scores: `results/neuron_analysis/llama-3.2-3B/`.
- **Open items:**
  - [ ] Repeat the Llama-3.2-3B BBQ baseline with the current configuration in `checkpoints_bbq_rerun` (per-item logs; it is the noise control of this experiment).
  - [ ] Neuron indices of the published Religion·Top-20 run: [results/bias-benchmarks-zeroed/README.md](results/bias-benchmarks-zeroed/README.md) describes a `llama-3.2-1B_neuron_manifest.json`, but no manifest file is in the repository and none was found on Drive (searched by name). For the 3B, regenerate the set from `results/neuron_analysis/llama-3.2-3B/` with the same selection rule as the paper and save the manifest this time.

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

- 2026-10-29: E1 closed (both new models compatible with OptiPFair, logs located, Instruct discarded); file created.
- 2026-10-28: E1 baselines finished for Llama-3.1-8B and Salamandra-7B (BBQ, EsBBQ) and Llama-3.2-3B-Instruct (A/B/C format); READMEs and evaluation logs updated.
- 2026-10-27: E0 documented in `docs/`; Llama-3.1-8B BBQ baseline run.
