# Notebooks

This folder contains the experiment notebooks for the Fairness Pruning workflow, from baseline evaluation to neuron-level analysis, intervention, and post-pruning validation.

Most notebooks are designed for Google Colab and include checkpoint/resume logic so long runs can be resumed after interruptions.

## Recommended execution order

1. 02_Evaluate_Base_Capabilities.ipynb
2. 02_Evaluate_BBQ.ipynb
3. 02_Evaluate_MBBQ.ipynb
4. 02_Evaluate_BBQ_Instruct.ipynb
5. 04_Graphics_Base_Capabilities.ipynb
6. 04_Graphics_Bias_BBQ.ipynb
7. 03_neuron_bias_detection_en.ipynb
8. 05_bias_path_analysis.ipynb
9. 06_zero_bias_neurons.ipynb
10. 07_EvalPrunedModels_Llama3B.ipynb
11. 07B_EvalPrunedModels_Llama3B.ipynb
12. 08_EvalLlamaPrunedEsp.ipynb
13. 08_EvalSalamandraPrunedEsp.ipynb

Note: 02_esbbq_lm_eval_harness.ipynb is kept as a legacy/validation notebook and is not required in the main pipeline.

## Quick I/O map (Input -> Output)

| Notebook | Main input(s) | Main output(s) |
|---|---|---|
| 02_Evaluate_Base_Capabilities.ipynb | Base model ids + lm-eval task suite | checkpoints/ + results/base_models_results_latest.csv + timestamped CSV/JSON + summary CSV |
| 02_Evaluate_BBQ.ipynb | Base model ids + BBQ task | checkpoints_bbq/ + per-item raw lm-eval JSON (checkpoints_bbq/results/lm_evals/, Drive only) + results/base_models_bbq_results_latest.csv + timestamped CSV/JSON + summary CSV |
| 02_Evaluate_MBBQ.ipynb | Base model ids + EsBBQ task set (10 categories) | checkpoints_mbbq/ + per-item raw lm-eval JSON per category (checkpoints_mbbq/results/lm_evals/, Drive only) + results/base_models_mbbq_results_latest.csv + timestamped CSV/JSON + category breakdown CSV |
| 02_Evaluate_BBQ_Instruct.ipynb | Instruct model ids + custom task `bbq_instruct_mc` (BBQ with A/B/C options, chat template) | checkpoints_bbq_instruct/ + per-item raw lm-eval JSON + results/instruct_models_bbq_mc_results_latest.csv + timestamped CSV/JSON + summary CSV |
| 02_esbbq_lm_eval_harness.ipynb (legacy) | EsBBQ task config generated in-notebook | esbbq_results/esbbq_final_results.json + esbbq_results/esbbq_raw_lm_eval_results.json |
| 03_neuron_bias_detection_en.ipynb | Prompt-pair dataset + model/tokenizer + OptiPFair scoring | Per-category bias_scores (.pt/.json), fairness_scores (.pt/.json), comparison_summary.json |
| 04_Graphics_Base_Capabilities.ipynb | Consolidated capability result JSON files in results/ | PNG/PDF figures in results/figures/ + summary_metrics_*.csv |
| 04_Graphics_Bias_BBQ.ipynb | Bias result JSON files (BBQ + EsBBQ) in results/bias/ | PNG/PDF figures in results/figures/bias/ + bias_summary_*.csv + bias_categories_*.csv |
| 05_bias_path_analysis.ipynb | results/neuron_analysis/{model}/{lang}/{Category}_bias_scores.json | Bias-path figures (PNG/PDF) + overlap/summary CSVs in results/figures/bias_path/ |
| 06_zero_bias_neurons.ipynb | Precomputed neuron scores in results/neuron_analysis/ + category set + Top-K threshold | In-memory zeroed model + neuron_indices mapping (optional saved model + zeroed_neuron_indices.json) |
| 07_EvalPrunedModels_Llama3B.ipynb | Zeroing experiment configs + neuron manifests + English BBQ | checkpoints_bbq_zeroed/ + results/bias-benchmarks-zeroed/llama-3.2-3B_* (generation + BBQ + capabilities CSV/JSON + manifests) |
| 07B_EvalPrunedModels_Llama3B.ipynb | Zeroing experiment configs + fairness manifests + English BBQ + capability follow-up | checkpoints_bbq_zeroed/ + checkpoints_capabilities_zeroed/ + results/bias-benchmarks-zeroed/llama-3.2-3B_* (generation + BBQ + capabilities CSV/JSON + manifests) |
| 08_EvalLlamaPrunedEsp.ipynb | Zeroing experiment configs + neuron manifests + Spanish EsBBQ | checkpoints_bbq_zeroed/ + results/bias-benchmarks-zeroed/llama-3.2-1B_esbbq_zeroed_results.csv/.json + ES manifest |
| 08_EvalSalamandraPrunedEsp.ipynb | Zeroing experiment configs + neuron manifests + Spanish EsBBQ | checkpoints_bbq_zeroed/ + results/bias-benchmarks-zeroed/salamandra-2B_esbbq_zeroed_results.csv/.json + ES manifest |

## Notebook-by-notebook guide

### 02_Evaluate_Base_Capabilities.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_Base_Capabilities.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_Base_Capabilities.ipynb)

Purpose:
Builds baseline capability metrics for unpruned models.

What it does:
- Runs a broad lm-eval benchmark suite (English + Spanish + language modeling tasks).
- Supports multi-model execution with robust checkpointing.
- Consolidates per-task metrics into machine-readable summary files.

Main outputs:
- Drive checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints/
- Consolidated results in results/base_models_results_latest.csv and timestamped CSV/JSON variants
- Additional summary CSV in the same results area

### 02_Evaluate_BBQ.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_BBQ.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_BBQ.ipynb)

Purpose:
Runs English bias baseline evaluation using BBQ on unpruned models.

What it does:
- Evaluates each base model on bbq with checkpoint/resume support.
- Logs per-item predictions (`log_samples=True`) and fixes `batch_size=16`.
- Aggregates accuracy metrics and exports a unified table.
- Keeps an evaluation log (dates, models, library versions, GPU) in its first markdown cells.

Main outputs:
- Drive checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_bbq/
- Per-item raw lm-eval JSON in checkpoints_bbq/results/lm_evals/ (about 700 MB per model, kept on Drive only)
- Consolidated results in results/base_models_bbq_results_latest.csv plus timestamped CSV/JSON
- BBQ summary CSV

### 02_Evaluate_MBBQ.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_MBBQ.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_MBBQ.ipynb)

Purpose:
Runs Spanish bias baseline evaluation with EsBBQ (MBBQ) across categories.

What it does:
- Evaluates models over EsBBQ categories: Age, Disability Status, Gender, LGBTQIA+, Nationality, Physical Appearance, Race/Ethnicity, Religion, SES, and Spanish Region.
- Runs one task per call with checkpoint/resume after each category, saves per-item raw lm-eval JSON and exports consolidated metrics.
- Adds category-level bias analysis. Its summary cells average the 10 categories without weighting; for cross-model and cross-language comparisons use item-weighted aggregates (see results/bias-benchmarks-base/README.md).

Main outputs:
- Drive checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_mbbq/
- Per-item raw JSON, one file per category: checkpoints_mbbq/results/lm_evals/ (runs of October 2026) and checkpoints_mbbq/<size>/results/lm_evals/ (runs of February 2026)
- Consolidated files in results/base_models_mbbq_results_latest.csv and timestamped CSV/JSON
- MBBQ summary and category breakdown CSV files

### 02_Evaluate_BBQ_Instruct.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_BBQ_Instruct.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/02_Evaluate_BBQ_Instruct.ipynb)

Purpose:
Runs the English bias baseline for an instruction-tuned model (Llama-3.2-3B-Instruct) with BBQ presented as a multiple-choice question with options A/B/C.

What it does:
- Registers the custom task `bbq_instruct_mc` (custom_tasks/bbq_instruct/) in lm-eval and evaluates with the model's chat template (`apply_chat_template=True`).
- Scores the log-likelihood of the answer letter; dataset and metric code are the same as lm-eval's default `bbq`.
- Results are not comparable with the base-model `bbq` results, because prompt and scoring differ.

Main outputs:
- Drive checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_bbq_instruct/
- Per-item raw lm-eval JSON in checkpoints_bbq_instruct/results/lm_evals/
- Consolidated results in results/instruct_models_bbq_mc_results_latest.csv plus timestamped CSV/JSON (they do not overwrite the base-model files)

### 02_esbbq_lm_eval_harness.ipynb (legacy)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/02_esbbq_lm_eval_harness.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/02_esbbq_lm_eval_harness.ipynb)

Purpose:
Legacy notebook for direct EsBBQ evaluation and metric validation with lm-evaluation-harness.

What it does:
- Builds task configuration and custom aggregation logic in-notebook.
- Produces raw and final EsBBQ JSON artifacts for validation/debugging.

Main outputs:
- Local files under esbbq_results/ (for example: esbbq_final_results.json)

Status:
Superseded by 02_Evaluate_MBBQ.ipynb for the current production workflow.

### 03_neuron_bias_detection_en.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/03_neuron_bias_detection_en.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/03_neuron_bias_detection_en.ipynb)

Purpose:
Computes neuron-level bias and fairness scores from demographic prompt pairs.

What it does:
- Uses OptiPFair analysis utilities to score neurons per category.
- Produces both tensor and JSON exports for downstream analysis.
- Generates cross-category comparison summaries.

Main outputs:
- Category files such as bias_scores.pt/.json and fairness_scores.pt/.json
- comparison_summary.json
- Stored under model/language experiment directories in results

### 04_Graphics_Base_Capabilities.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/04_Graphics_Base_Capabilities.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/04_Graphics_Base_Capabilities.ipynb)

Purpose:
Creates visual analytics for baseline capability benchmarks.

What it does:
- Loads consolidated model result JSON files.
- Builds grouped charts, perplexity plots, MMLU deep dives, and cross-lingual gap visuals.
- Exports publication-ready figures and summary tables.

Main outputs:
- Figures in results/figures as PNG/PDF
- Summary metrics CSV in results/figures/

### 04_Graphics_Bias_BBQ.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/04_Graphics_Bias_BBQ.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/04_Graphics_Bias_BBQ.ipynb)

Purpose:
Visualizes bias behavior across BBQ (English) and EsBBQ (Spanish).

What it does:
- Compares ambiguous vs disambiguated bias patterns by model.
- Plots amb-disamb gap comparisons and top-category heatmaps.
- Exports both figures and tabular summaries.

Main outputs:
- Figures in results/figures/bias as PNG/PDF
- Summary/category CSV files in results/figures/bias/

### 05_bias_path_analysis.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/05_bias_path_analysis.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/05_bias_path_analysis.ipynb)

Purpose:
Analyzes where bias is localized in model depth and neuron space.

What it does:
- Consolidates and normalizes neuron bias scores.
- Runs Top-K candidate extraction, layer/category heatmaps, and neuron-level heatmaps.
- Computes overlap analysis (Jaccard) and cross-lingual consistency diagnostics.

Main outputs:
- Inputs read from results/neuron_analysis/{model}/{lang}/...
- Figures and analysis CSVs in results/figures/bias_path/

### 06_zero_bias_neurons.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/06_zero_bias_neurons.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/06_zero_bias_neurons.ipynb)

Purpose:
Builds a fairness-zeroed model by silencing shared high-bias neurons.

What it does:
- Loads precomputed scores.
- Extracts Top-K candidate neurons per category.
- Computes N-way intersections to identify shared biased superneurons.
- Applies zero_neurons_mlp and runs before/after qualitative checks.

Main outputs:
- In-memory zeroed model for evaluation
- Optional saved model and zeroed_neuron_indices.json (save block included but commented)

### 07_EvalPrunedModels_Llama3B.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/07_EvalPrunedModels_Llama3B.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/07_EvalPrunedModels_Llama3B.ipynb)

Purpose:
Evaluates English BBQ and selected capability tasks on zeroed Llama-3.2-3B variants.

What it does:
- Builds multiple pruning experiments from neuron score manifests.
- Applies zeroing per experiment and evaluates BBQ with checkpointing.
- Consolidates JSON/CSV outputs and stores reproducibility manifests.

Main outputs:
- Checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_bbq_zeroed/
- Zeroed benchmark outputs in results/bias-benchmarks-zeroed/
- Capability-zeroed outputs and backups (including drive backup path)

### 07B_EvalPrunedModels_Llama3B.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/07B_EvalPrunedModels_Llama3B.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/07B_EvalPrunedModels_Llama3B.ipynb)

Purpose:
Extended pruning-evaluation pipeline for English BBQ on zeroed Llama-3.2-3B variants, including fairness-focused experiment tracking.

What it does:
- Runs bias-oriented and fairness-oriented pruning experiment grids.
- Executes BBQ evaluation with checkpoint/resume per experiment.
- Exports generation artifacts, manifests, and capability follow-up results for selected experiments.

Main outputs:
- Checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_bbq_zeroed/
- Capability checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_capabilities_zeroed/
- Consolidated artifacts in results/bias-benchmarks-zeroed/ (JSON/CSV + manifests)

### 08_EvalLlamaPrunedEsp.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/08_EvalLlamaPrunedEsp.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/08_EvalLlamaPrunedEsp.ipynb)

Purpose:
Evaluates Spanish EsBBQ on zeroed Llama-3.2-1B experiments.

What it does:
- Reuses the pruning-experiment pattern from notebook 07.
- Runs esBBQ for each experiment and consolidates outputs.

Main outputs:
- Checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_bbq_zeroed/
- Consolidated files in results/bias-benchmarks-zeroed/llama-3.2-1B_esbbq_zeroed_results.csv and JSON
- Neuron manifest for Spanish experiments in results/bias-benchmarks-zeroed/

### 08_EvalSalamandraPrunedEsp.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peremartra/fairness-pruning/blob/main/notebooks/08_EvalSalamandraPrunedEsp.ipynb)
[![View in NBViewer](https://img.shields.io/badge/NBViewer-Open-orange?logo=jupyter)](https://nbviewer.org/github/peremartra/fairness-pruning/blob/main/notebooks/08_EvalSalamandraPrunedEsp.ipynb)

Purpose:
Evaluates Spanish EsBBQ on zeroed Salamandra-2B experiments.

What it does:
- Mirrors notebook 08_EvalLlamaPrunedEsp.ipynb with Salamandra-2B configuration.
- Runs experiment-wise zeroing, esBBQ evaluation, and consolidation.

Main outputs:
- Checkpoints in /content/drive/MyDrive/fair_pruning/checkpoints_bbq_zeroed/
- Consolidated files in results/bias-benchmarks-zeroed/salamandra-2B_esbbq_zeroed_results.csv and JSON
- Spanish neuron manifest in results/bias-benchmarks-zeroed/

## Practical notes

- Baseline artifacts are under results/bias-benchmarks-base/.
- The first markdown cells of the 02_Evaluate_* notebooks hold an evaluation log (dates, models, library versions, GPU) for each run.
- Post-intervention artifacts are under results/bias-benchmarks-zeroed/.
- Visualization notebooks write figures under results/figures/.
- Several notebooks assume Google Drive paths in Colab; adjust path variables if running locally.
