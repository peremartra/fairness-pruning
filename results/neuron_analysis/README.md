# Neuron-Level Bias Analysis

Per-neuron bias and fairness-pruning scores for unpruned baseline models, computed using the `optipfair` library. These scores are the core input to the fairness-aware pruning step of the project.

**Analysis Framework:** `optipfair` (`analyze_neuron_bias`, `compute_fairness_pruning_scores`)  
**Analysis Dates:** April 27, 2026

## Source Datasets

| Language | Dataset |
|----------|---------|
| English | [`oopere/fairness-pruning-pairs-en`](https://huggingface.co/datasets/oopere/fairness-pruning-pairs-en) |
| Spanish | [`oopere/fairness-pruning-pairs-es`](https://huggingface.co/datasets/oopere/fairness-pruning-pairs-es) |

Prompt pairs used here are the same as those in `../generations/`. Each pair contains two prompts identical except for a single demographic attribute (e.g., *old* vs. *young*).

## Models

| Model | Parameters | English (EN) | Spanish (ES) |
|-------|-----------|:---:|:---:|
| meta-llama/Llama-3.2-1B | 1B | ✅ Complete | ✅ Complete |
| meta-llama/Llama-3.2-3B | 3B | ✅ Complete | ✅ Complete |

---

## Score Types

### Bias Scores (`{Category}_bias_scores`)

Computed by `analyze_neuron_bias`. For each neuron, the score is the **mean absolute difference in activation** between the two prompts in a pair, averaged across all pairs in the category.

- **Keys:** layer projection names — e.g., `gate_proj_layer_0`, `up_proj_layer_5`
- **Values:** float tensor of shape `[hidden_size]` (8192 for Llama-3.2)
- **Interpretation:** A **high value** indicates the neuron responds very differently depending on the demographic attribute — i.e., the neuron is a strong carrier of bias for that category.

**Configuration used:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `target_layers` | `["gate_proj", "up_proj"]` | GLU MLP projections analysed |
| `aggregation` | `mean` | Bias averaged across all sequence positions |

### Fairness-Pruning Scores (`{Category}_fairness_scores`)

Computed by `compute_fairness_pruning_scores`. Combines the bias score with the neuron's structural importance into a single pruning signal:

$$\text{FairnessPruningScore} = \text{bias\_weight} \times \text{BiasScore}_{\text{norm}} + (1 - \text{bias\_weight}) \times (1 - \text{ImportanceScore}_{\text{norm}})$$

- **Keys:** integer layer indices — e.g., `0`, `1`, …
- **Values:** float tensor of shape `[hidden_size]`
- **Interpretation:** A **high value** means the neuron is a good pruning candidate — it has **low structural importance** and/or **high bias contribution**. These are the neurons targeted for zeroing in the fairness-pruning step.

**Configuration used:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bias_weight` | `0.45` | Balanced: slightly weighted towards importance |
| `top_percent` | `1.0` | Top 1% of neurons per layer reported in summary |

The `bias_weight=0.45` setting means both factors contribute roughly equally, with a slight priority on preserving structurally important neurons.

---

## Categories

| Category | Description |
|----------|-------------|
| Age | Stereotypes related to age (e.g., *old* vs. *young*) |
| Gender | Stereotypes related to gender (e.g., *male* vs. *female*) |
| PhysicalAppearance | Stereotypes related to physical traits |
| RaceEthnicity | Stereotypes related to race or ethnicity |
| Religion | Stereotypes related to religious affiliation |

---

## Directory Structure

```
neuron_analysis/
├── llama-3.2-1B/
│   ├── en/
│   └── es/
└── llama-3.2-3B/
    ├── en/
    └── es/
```

## Files per Language Folder

| File | Description |
|------|-------------|
| `{Category}_bias_scores.json` | Per-neuron bias scores, JSON format |
| `{Category}_bias_scores.pt` | Per-neuron bias scores, PyTorch tensor format |
| `{Category}_fairness_scores.json` | Per-neuron fairness-pruning scores, JSON format |
| `{Category}_fairness_scores.pt` | Per-neuron fairness-pruning scores, PyTorch tensor format |
| `comparison_summary.json` | Experiment config and top biased neurons per layer and category |

## Score File Format

**Bias score files** (`.json`) — `Dict[str, {"shape": [...], "values": [...]}]`:
```json
{
  "gate_proj_layer_0": { "shape": [8192], "values": [0.012, 0.021, ...] },
  "up_proj_layer_0":   { "shape": [8192], "values": [0.008, 0.019, ...] }
}
```

**Fairness score files** (`.json`) — `Dict[int, {"shape": [...], "values": [...]}]`:
```json
{
  "0": { "shape": [8192], "values": [0.57, 0.58, ...] },
  "1": { "shape": [8192], "values": [0.61, 0.55, ...] }
}
```

The `.pt` files contain the same data as PyTorch tensors, ready for direct use in pruning scripts.
