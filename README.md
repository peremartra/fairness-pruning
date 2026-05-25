# Fairness Pruning: Bias Mitigation Through Activation-Guided MLP Width Pruning

Official implementation and experiment repository for the Fairness Pruning workflow.

This project studies how to reduce social bias in LLMs by identifying neurons with high demographic sensitivity and low structural relevance, then applying targeted MLP neuron zeroing/pruning.

## What Is In Scope

- Bias analysis at neuron level using OptiPFair.
- Baseline evaluation on capability and bias benchmarks.
- Post-intervention (zeroed) evaluation to measure fairness/performance retention.
- English and Spanish workflows, plus custom task configurations for EsBBQ, CaBBQ, and VeritasQA.

## Current Models Covered

- meta-llama/Llama-3.2-1B
- meta-llama/Llama-3.2-3B
- BSC-LT/salamandra-2b

Model coverage depends on stage:

- Baseline bias/capability: all three models.
- Baseline generations: Llama-3.2-1B and Llama-3.2-3B.
- Zeroed evaluations: Llama-3.2-1B (BBQ + EsBBQ + capabilities) and Salamandra-2B (EsBBQ).

## Repository Structure

| Path | Purpose |
|---|---|
| custom_tasks/ | lm-eval YAML task configs for EsBBQ, CaBBQ, and VeritasQA |
| datasets/ | Prompt-pair datasets and dataset utilities/docs |
| notebooks/ | End-to-end experiment notebooks (baseline, analysis, intervention, validation) |
| results/ | Generated artifacts (baseline + zeroed) and figures |
| tests/ | Automated validation for task configs |
| utils.py | Shared helper utilities |
| requirements.txt | Python dependencies |

See the local READMEs for details:

- datasets/README.md
- notebooks/README.md
- results/README.md
- custom_tasks/esbbq/README.md
- custom_tasks/cabbq/README.md

## Experiment Pipeline (Current)

1. Baseline capabilities
- Notebook: notebooks/02_Evaluate_Base_Capabilities.ipynb
- Output: baseline capability files under results/

2. Baseline bias benchmarks
- Notebooks: notebooks/02_Evaluate_BBQ.ipynb, notebooks/02_Evaluate_MBBQ.ipynb
- Output: baseline bias files under results/ and results/bias-benchmarks-base/

3. Neuron-level bias and fairness scoring
- Notebook: notebooks/03_neuron_bias_detection_en.ipynb
- Output: per-model/language artifacts under results/neuron_analysis/

4. Bias path visualization and overlap analysis
- Notebook: notebooks/05_bias_path_analysis.ipynb
- Output: figures and CSV summaries under results/figures/bias_path/

5. Zeroing experiments and post-intervention evaluation
- Notebooks:
  - notebooks/07_EvalPrunedModels.ipynb
  - notebooks/08_EvalLlamaPrunedEsp.ipynb
  - notebooks/08_EvalSalamandraPrunedEsp.ipynb
- Output:
  - results/bias-benchmarks-zeroed/
  - results/capabilities_zeroed/

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run config validation tests

Offline checks only:

```bash
pytest tests/ -m "not network"
```

Full checks (includes HuggingFace dataset loading):

```bash
pytest tests/
```

### 3) Run notebooks in order

Use the sequence documented in notebooks/README.md.

Important runtime note:
- Several notebooks are Colab-oriented and reference Google Drive paths.
- If running locally, adapt path variables before execution.

## Custom lm-eval Tasks

The repository includes ready-to-use YAML task definitions under custom_tasks/:

- custom_tasks/esbbq/: 10 EsBBQ categories
- custom_tasks/cabbq/: 10 CaBBQ categories
- custom_tasks/veritasQA/: VeritasQA task definitions

These configs are automatically validated by tests/test_task_configs.py for:

- Required fields and parse correctness
- Task/file naming consistency
- Dataset subset presence
- Dataset loadability and required columns (network tests)

## Results Layout

The results directory is organized by artifact type:

- results/bias-benchmarks-base/: baseline BBQ/EsBBQ benchmark exports
- results/bias-benchmarks-zeroed/: post-zeroing benchmark outputs and manifests
- results/capabilities_zeroed/: capability retention for zeroed variants
- results/generations/: baseline paired generations
- results/neuron_analysis/: per-neuron bias/fairness scores
- results/figures/bias_path/: bias-path visual analytics

For exact file inventories and schemas, use results/README.md and subdirectory READMEs.

## Citation and References

- OptiPFair: https://github.com/peremartra/optipfair
- EsBBQ / CaBBQ resources: https://github.com/langtech-bsc/EsBBQ-CaBBQ
- Additional bibliography in REFERENCES.MD
