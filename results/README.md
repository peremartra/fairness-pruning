# Results Directory

This folder contains all generated artifacts for baseline and post-intervention (zeroed) experiments used in the fairness-pruning project.

The purpose of this README is to document the current folder structure and where each type of artifact lives.
Detailed metric interpretation is intentionally delegated to each subdirectory README.

## Current Structure

| Path | Purpose |
| :--- | :--- |
| `bias-benchmarks-base/` | Baseline bias benchmark outputs for unpruned models: BBQ + EsBBQ for five base models, and BBQ with A/B/C options for one instruct model |
| `bias-benchmarks-zeroed/` | Post-zeroing benchmark outputs and intervention manifests |
| `capabilities_zeroed/` | Capability retention evaluations for zeroed variants (raw artifacts for Llama-3.2-1B and Llama-3.2-3B; consolidated exports currently for Llama-3.2-1B) |
| `generations/` | Baseline paired generations used as analysis input |
| `neuron_analysis/` | Per-neuron bias/fairness scores for pruning candidate selection |
| `figures/bias_path/` | Bias-path visualizations and summary CSVs |

## Top-Level Files In This Folder

These files are stored directly under `results/` (outside subfolders):

- Baseline capability result snapshots:
  - `base_models_results_20251206_150732.json`
  - `base_models_results_20251206_175322.json`
- Baseline BBQ aggregate exports:
  - `base_models_bbq_results_20251207_191101.json` (first three models, December 2025)
  - `base_models_bbq_results_latest.csv` and `base_models_bbq_results_20261028_091856.json` (five base models, October 2026)
- Baseline EsBBQ/model-specific JSON artifacts:
  - `esbbq_final_results_llama-3.2-1B.json`
  - `meta_llama_llama_3.2_1b.json`
  - `meta_llama_llama_3.2_3b.json`
  - `bsc_lt_salamandra_2b.json`

Per-item model predictions (raw lm-eval JSON, hundreds of MB) are not stored in this repository; they are on Drive under `fair_pruning/checkpoints_bbq/results/lm_evals/`, `fair_pruning/checkpoints_mbbq/` and `fair_pruning/checkpoints_bbq_instruct/results/lm_evals/`.

`bias-benchmarks-base/` also holds `base_models_mbbq_results_latest.csv` and `base_models_mbbq_results_20261028_151839.json` (EsBBQ, five base models). The CSV of the instruct model (A/B/C format) is only on Drive.

## Subdirectory Documentation

Use the README in each subfolder for schemas, provenance, and generation workflow details:

- `bias-benchmarks-base/README.md`
- `bias-benchmarks-zeroed/README.md`
- `capabilities_zeroed/README.md`
- `generations/README.md`
- `neuron_analysis/README.md`
- `figures/bias_path/README.md`

