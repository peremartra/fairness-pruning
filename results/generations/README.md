# Model Generations

Baseline text generations for unpruned models using stereotyped vs. anti-stereotyped prompt pairs from the fairness pruning datasets.

**Generation Framework:** HuggingFace Transformers  
**Decoding:** Greedy (do_sample=False), max_new_tokens=50, repetition_penalty=1.1, dtype=float16  
**Generation Dates:** April 27, 2026

## Source Datasets

| Language | Dataset | Pairs |
|----------|---------|-------|
| English | [`oopere/fairness-pruning-pairs-en`](https://huggingface.co/datasets/oopere/fairness-pruning-pairs-en) | 70 |
| Spanish | [`oopere/fairness-pruning-pairs-es`](https://huggingface.co/datasets/oopere/fairness-pruning-pairs-es) | 100 |

Each dataset entry is a **prompt pair**: two prompts that are identical except for a single demographic attribute (e.g., *old* vs. *young*, *male* vs. *female*). The model's continuation for both prompts is recorded, enabling direct comparison of how the attribute influences the generated text.

## Models

| Model | Parameters | English (EN) | Spanish (ES) |
|-------|-----------|:---:|:---:|
| meta-llama/Llama-3.2-1B | 1B | ✅ Complete | ✅ Complete |
| meta-llama/Llama-3.2-3B | 3B | ✅ Complete | ✅ Complete |

---

## Pairs per Category

### English Dataset

| Category | Pairs |
|----------|-------|
| Age | 10 |
| Gender | 15 |
| PhysicalAppearance | 15 |
| RaceEthnicity | 15 |
| Religion | 15 |
| **Total** | **70** |

### Spanish Dataset

| Category | Pairs |
|----------|-------|
| Age | 15 |
| Gender | 20 |
| PhysicalAppearance | 15 |
| RaceEthnicity | 15 |
| Religion | 35 |
| **Total** | **100** |

---

## Directory Structure

```
generations/
├── llama-3.2-1b/
│   ├── en/
│   └── es/
└── llama-3.2-3b/
    ├── en/
    └── es/
```

## Files per Language Folder

| File | Description |
|------|-------------|
| `{Category}_generations.csv` | Prompt pairs and model responses, CSV format |
| `{Category}_generations.json` | Prompt pairs and model responses, JSON format |
| `generations_summary.json` | Experiment metadata and pair counts for the model/language |

## Generation File Format

Each generation file (`.json`) contains a list of objects with the following fields:

| Field | Description |
|-------|-------------|
| `template_id` | Identifier of the prompt template |
| `attribute_1` | First demographic attribute (e.g., `old`) |
| `attribute_2` | Contrasting demographic attribute (e.g., `young`) |
| `prompt_1` | Prompt text using `attribute_1` |
| `response_1` | Model continuation for `prompt_1` |
| `prompt_2` | Prompt text using `attribute_2` |
| `response_2` | Model continuation for `prompt_2` |

These paired responses are the direct input to the neuron-level analysis in `../neuron_analysis/`.
