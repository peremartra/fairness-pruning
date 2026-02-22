# Notebooks

This directory contains the Jupyter notebooks used for the experiments in this repository.

---

### 📘 Notebook Descriptions

#### `02_Evaluate_Base_Capabilities.ipynb`

* **Description:** Establishes the general performance baselines for the unpruned models (Base Models) before applying any bias mitigation techniques.
* **What it executes:** Evaluates the models (*Llama-3.2-1B*, *Llama-3.2-3B*, and *Salamandra-2B*) across a comprehensive suite of 15 `lm_eval` benchmarks, including language modeling (WikiText, Lambada), reasoning and knowledge in English (MMLU, HellaSwag, ARC, GSM8K, etc.), and multilingual/Spanish capabilities (MMLU-ES, Belebele).
* **What it obtains:** General performance metrics such as Accuracy, Exact Match, and Perplexity.
* **Where it saves the results:** Raw results per model are temporarily saved in `/content/drive/MyDrive/fair_pruning/checkpoints/`. Once consolidated, the final output is exported to the local repository or Drive folder under the path `results/base_models_results_latest.csv` (and its `.json` counterpart).

#### `02_Evaluate_BBQ.ipynb`

* **Description:** Performs the specific baseline evaluation for social biases using the original English benchmark BBQ (Bias Benchmark for QA).
* **What it executes:** Evaluates the base models (unpruned) using the native `bbq` task from `lm-evaluation-harness` framework.
* **What it obtains:** Accuracy (`acc` and `acc_norm`) against questions exposing social biases across 9 sociodemographic dimensions in English.
* **Where it saves the results:** Generates intermediate checkpoints in `/content/drive/MyDrive/fair_pruning/checkpoints_bbq/` and consolidates the final results table in `results/base_models_bbq_results_latest.csv` (and `.json`).

#### `02_Evaluate_MBBQ.ipynb`

* **Description:** This is the main and definitive pipeline for evaluating biases in the Spanish and Catalan context using the MBBQ (EsBBQ) adaptation.
* **What it executes:** Downloads and implements custom tasks (YAMLs and `esbbq_utils.py`) integrated with `lm_eval`. It runs the evaluation of the 3 base models across the 10 social categories of EsBBQ (Age, Gender, Religion, Socioeconomic Status, etc.).
* **What it obtains:** Detailed metrics capturing model behavior, including Accuracy and Bias Score in both ambiguous and disambiguated contexts (`acc_ambig`, `acc_disambig`, `bias_score_ambig`, `bias_score_disambig`).
* **Where it saves the results:** Stores raw *lm_eval* dumps in `checkpoints_mbbq/[model]/results/lm_evals/` and exports the final consolidated files to `results/base_models_mbbq_results_latest.csv` and `results/base_models_mbbq_results_latest.json`.

#### `02_esbbq_lm_eval_harness.ipynb` (legacy)

* **Description:** A standalone notebook focused on testing and validating the strict mathematical calculation of the Bias Score formulas in Spanish (including comma-separated metadata cleaning).
* **What it executes:** Directly launches `lm-evaluation-harness` version `0.4.8`, generating aggregation functions "on the fly" within the execution environment instead of relying on the GitHub repository files.
* **What it obtains:** Validated bias scores matching exactly the ones reported in the BSC-CNS paper for the EsBBQ dataset.
* **Where it saves the results:** Usually exports an individual JSON (e.g., `esbbq_final_results_salamandra2b.json`) to a locally specified path or Drive (results/bias path). *(Note: Following the code integration into GitHub, the functionality of this notebook has been entirely covered by `02_Evaluate_MBBQ.ipynb`, keeping this file only for testing/legacy purposes).*

#### `04_Graphics_Base_Capabilities.ipynb`

* **Description:** An analytical visualization tool to compare the baseline performances extracted from the previous notebooks.
* **What it executes:** Downloads (or reads locally) the `.json` files containing the consolidated base capabilities and MMLU results generated previously. It processes the data using Pandas and generates plots with Seaborn/Matplotlib.
* **What it obtains:** A collection of high-level synthetic charts (grouped bar charts by task type, perplexity comparisons, performance analysis by MMLU/MMLU_ES categories, and EN-ES cross-lingual gap evaluations).
* **Where it saves the results:** Automatically exports the figures in `.png` and `.pdf` formats alongside a summary `.csv` in the `results/figures/` folder (e.g., `results/figures/group_language_modeling_[timestamp].png`).
