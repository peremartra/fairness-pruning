# Capabilities Zeroed

This directory stores capability evaluation artifacts for zeroed (fairness-pruned) model variants.

In the current repository state, raw capability benchmarking artifacts are available for Llama-3.2-1B and Llama-3.2-3B.
Consolidated cross-experiment exports in results/ are currently available for Llama-3.2-1B.

Generation source:

- Notebook: 07_EvalPrunedModels.ipynb (capabilities section after zeroing)
- Notebook: 07B_EvalPrunedModels_Llama3B.ipynb (capabilities section after zeroing)
- Workflow reference: notebooks/README.md

## Directory structure

- llama_3.2_1B/
	- Per-experiment raw capability outputs (one JSON file per experiment run).
- llama_3.2_3B/
	- Per-experiment raw capability outputs (one JSON file per experiment run).
- results/
	- Consolidated cross-experiment exports (flat CSV + flat JSON array, currently for Llama-3.2-1B).

## File inventory

| Path | Type | What it contains | Generated from |
|---|---|---|---|
| llama_3.2_1B/exp2_capabilities.json | JSON | Raw capability metrics for experiment exp2 | 07_EvalPrunedModels.ipynb |
| llama_3.2_1B/exp9_capabilities.json | JSON | Raw capability metrics for experiment exp9 | 07_EvalPrunedModels.ipynb |
| llama_3.2_1B/exp10_capabilities.json | JSON | Raw capability metrics for experiment exp10 | 07_EvalPrunedModels.ipynb |
| llama_3.2_1B/exp11_capabilities.json | JSON | Raw capability metrics for experiment exp11 | 07_EvalPrunedModels.ipynb |
| llama_3.2_3B/exp1_capabilities.json | JSON | Raw capability metrics for experiment exp1 | 07B_EvalPrunedModels_Llama3B.ipynb |
| llama_3.2_3B/exp4_capabilities.json | JSON | Raw capability metrics for experiment exp4 | 07B_EvalPrunedModels_Llama3B.ipynb |
| results/llama-3.2-1B_capabilities_zeroed_results.csv | CSV | Consolidated table with baseline vs zeroed values and retention percentage | 07_EvalPrunedModels.ipynb (consolidation step) |
| results/llama-3.2-1B_capabilities_zeroed_results.json | JSON | Same consolidated records as CSV, exported as JSON array | 07_EvalPrunedModels.ipynb (consolidation step) |

## Per-experiment JSON schema

Each file under llama_3.2_1B/exp*_capabilities.json and llama_3.2_3B/exp*_capabilities.json follows this structure:

- metadata
	- model_name
	- model_key
	- experiment_id
	- experiment_name
	- n_neurons
	- completed
	- completed_at
- results
	- one object per evaluated task:
		- wikitext
		- mmlu
		- arc_challenge
		- hellaswag
		- hellaswag_es

Metric examples by task:

- wikitext: word_perplexity,none, byte_perplexity,none, bits_per_byte,none
- mmlu: accuracy + category summaries + subcategories
- arc_challenge / hellaswag / hellaswag_es: normalized accuracy fields

## Consolidated CSV/JSON schema

Both consolidated files in results/ represent the same normalized rows.

Columns/fields:

- experiment_id
- experiment_name
- n_neurons
- task
- metric
- baseline
- zeroed
- retention_pct

Where:

- baseline is the reference metric from the base model benchmark.
- zeroed is the metric measured on the corresponding zeroed model variant.
- retention_pct is the relative preservation ratio used in the notebook consolidation.

## How these files were generated

1. Build zeroed model variants
- In 07_EvalPrunedModels.ipynb, neuron selections from predefined experiments are applied with zero_neurons_mlp.
- In 07B_EvalPrunedModels_Llama3B.ipynb, neuron selections from predefined experiments are applied with zero_neurons_mlp.

2. Run capability tasks per experiment
- For each selected experiment, notebook 07 writes one raw file to llama_3.2_1B/exp*_capabilities.json.
- For each selected experiment, notebook 07B writes one raw file to llama_3.2_3B/exp*_capabilities.json.

3. Consolidate and export
- The notebook parses raw files, aligns baseline and zeroed metrics, computes retention_pct, and exports:
	- results/llama-3.2-1B_capabilities_zeroed_results.csv
	- results/llama-3.2-1B_capabilities_zeroed_results.json

## Very brief result summary

- Llama-3.2-1B
	- Evaluated experiments in this folder: exp2, exp9, exp10, exp11
	- Total consolidated records: 20 rows (4 experiments x 5 tasks)
	- Tasks covered: wikitext, mmlu, arc_challenge, hellaswag, hellaswag_es
	- retention_pct range in the consolidated export: 97.93 to 101.34
	- Mean retention_pct across all rows: 99.49

- Llama-3.2-3B
	- Evaluated experiments in this folder: exp1, exp4
	- Raw capability files available: 2 (2 experiments x 5 tasks)
	- Tasks covered: wikitext, mmlu, arc_challenge, hellaswag, hellaswag_es
	- MMLU accuracy range across raw files: 0.5682 to 0.5786
	- Consolidated retention export under results/ is not present yet for this model

This is the reason the results/ consolidation currently contains only Llama-3.2-1B capability evaluations: it was the model consolidated during the current post-zeroing capability validation cycle.

## Usage notes

- Use files in llama_3.2_1B/ and llama_3.2_3B/ when you need full per-task raw outputs (including MMLU subcategories).
- Use files in results/ when you need compact analysis-ready tables for plotting or comparison.
