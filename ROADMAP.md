# Master's Thesis Roadmap: Fairness Pruning

**Title:** Fairness Pruning: Bias Mitigation through Activation-Guided MLP Width Pruning in Large Language Models

**Student:** Pere Martra  
**Program:** Master in Research in Artificial Intelligence (UIMP)  
**Target Models:** Llama-3.2-1B, Llama-3.2-3B, BSC-LT/Salamandra-2B  
**Timeline:** December 2025 - July 2026  
**Total Estimated Hours:** 230-260 hours (~10h/week over 25 weeks)

---

## Research Questions

**RQ1:** Is bias consistently localized in specific MLP neurons across different model architectures?  
**RQ2:** Do different bias categories (gender, race, age) share structural overlap in neural circuits?  
**RQ3:** Can we achieve ≥15% bias reduction while maintaining ≥95% capability retention?

---

## Phase 1: Baseline Establishment ✔

**Objective:** Complete radiography of model behavior before intervention

### ✔ 1.1. General Capabilities Evaluation (Base Models)

**Status:** COMPLETED  
**Models:** Llama-3.2-1B, Llama-3.2-3B, Salamandra-2B  
**Benchmarks (COMPLETED):** Wikitext, Lambada, IFEval, GSM8K, MMLU, ARC, HellaSwag, TruthfulQA (MC2), Belebele, Global-MMLU, VeritasQA (ES/CA)  
**Results:** Available in `results/benchmarks/capabilities/`

#### Deliverables:
- [X] Capabilities baseline for all 3 models
- [X] Updated capabilities baseline summary table

---

### ✔ 1.2. Bias Benchmark Execution (Base)

**Status:** COMPLETED  
**Description:** Baseline bias metrics using validated multilingual benchmarks

#### Subtasks:

**A. BBQ (Bias Benchmark for QA) - English**
- **Dataset:** 58,492 examples across 9 social dimensions
- **Source:** `lm-evaluation-harness` task: `bbq`
- **Metrics collected:**
  - Bias Score: % stereotype preference in ambiguous contexts
  - Accuracy: % correct answers in disambiguated contexts
  - Category-level breakdown (gender, race, age, religion, etc.)
- **Models:** All 3 base models ✔

**B. EsBBQ - Spanish**
- **Dataset:** ~11k examples across 10 social dimensions, adapted to Spanish social context
- **Source:** BSC-LT/EsBBQ (Ruiz-Fernández et al., 2025)
- **Note:** EsBBQ was selected over MBBQ for its native Spanish social context adaptation and peer-reviewed validation (arXiv:2507.11216)
- **Metrics:** Same as BBQ (Bias Score + Accuracy, ambiguous + disambiguated)
- **Models:** All 3 base models ✔

**Key findings from Phase 1:**
- **Llama-3.2-3B** shows the most consistent bias patterns across both languages, making it the optimal validation reference
- **Physical Appearance** is the most consistent bias category across languages
- **Age** bias is highly language-dependent
- All models show higher overall accuracy on EsBBQ (Spanish) than BBQ (English), with +11-13pp improvement
- Llama-3.2-3B shows extreme bias in ambiguous contexts (always defaulting to stereotypical answers), indicating highly localized bias circuits

#### Deliverables:
- [X] BBQ results for all 3 models → `results/benchmarks/bbq/`
- [X] EsBBQ results for all 3 models → `results/benchmarks/bbq/`
- [X] Baseline bias summary table (CSV/JSON)
- [X] Analysis document with cross-lingual observations

---

## Phase 2: Detection and Pruning (Intervention with OptiPFair)

**Objective:** Identify bias-encoding neurons and generate pruned model variants

### 🔄 2.1. Neuron Detection via Differential Activation Analysis

**Status:** IN PROGRESS  
**Estimation:** 25-30 hours  
**Description:** Use OptiPFair to map neurons that differentially activate under biased vs. neutral stimuli

#### Subtasks:

**A. Prompt Pair Dataset** ✔
- Minimal pairs for each bias category, each pair differing only in the demographic attribute
- **Hard constraint:** Both prompts in a pair must tokenize to the same length (Llama-3.2-1B tokenizer), required for position-by-position activation comparison
- Token verification script: `scripts/token_verification.py`
- Published on HuggingFace:
  - English: `oopere/fairness-pruning-pairs-en` (75 pairs, 6 categories)
  - Spanish: `oopere/fairness-pruning-pairs-es` (100 pairs, 5 categories)
- Categories: PhysicalAppearance, Gender, Age, RaceEthnicity, Religion, SES (EN only)
- Social contexts per template: labour, institutional, healthcare, social, educational

**B. Activation Capture and Neuron Scoring with OptiPFair** ✔ (Llama-1B, Llama-3B)
- Library: `optipfair` — `analyze_neuron_bias` + `compute_fairness_pruning_scores`
- For each prompt pair, activations captured at:
  - Gate projection (`gate_proj_layer_N`)
  - Up projection (`up_proj_layer_N`)
- Aggregation: `mean` across sequence positions
- Models completed: Llama-3.2-1B ✔, Llama-3.2-3B ✔, Salamandra-2B ⬜

**Neuron Scoring:**

For each neuron `i` in layer `l`, OptiPFair computes:

```
BiasScore_i = mean(|activation_prompt1 - activation_prompt2|)
              across all prompt pairs in the category

FairnessPruningScore_i = bias_weight × BiasScore_norm_i
                         + (1 - bias_weight) × (1 - ImportanceScore_norm_i)

where bias_weight = 0.45 (balanced, slight edge toward structural importance)

→ High score = high bias sensitivity + low structural importance → PRUNE CANDIDATE
```

**C. Preliminary Findings (Llama-1B and Llama-3B, EN)** ✔

Key finding: **each bias category has a distinct dominant layer**, suggesting category-specific neural circuits rather than a universal bias circuit.

| Category | Dominant layer(s) | Notes |
|----------|------------------|-------|
| PhysicalAppearance | 0, 1 | Early layers, surface representations |
| Gender | 0 | Extremely concentrated (>70% of candidates in layer 0 for 3B) |
| Age | 13, 22 | Mid-to-late layers |
| RaceEthnicity | 0, 1, 13 | Bimodal distribution |
| Religion | 14 (1B), 25 (3B) | Deepest layers; shifts with model scale |
| SES | 12, 13 | Late-mid layers |

- N-way intersection across all 6 categories at 0.5% global threshold: **1 neuron** shared, consistently in layer 2
- Bias circuits are **category-specific**, not universal — each type of demographic bias resides in different model depths
- At larger scale (3B), bias circuits migrate toward deeper layers and become more concentrated

**D. Qualitative Generation** ⬜
- Notebook: `04_prompt_pair_generations.ipynb`
- Generate model completions for all prompt pairs before pruning
- Decoding: greedy (`do_sample=False`), `max_new_tokens=50`, `repetition_penalty=1.1`
- Saved to: `results/generations/{model}/{dataset}/`
- Purpose: qualitative evidence of bias pre-pruning; comparison baseline for post-pruning analysis

**E. Cross-Model Consistency Analysis (RQ1)** ⬜ (partial)
- Compare neuron fairness score rankings between:
  - Llama-3.2-1B vs. Llama-3.2-3B *(data available, formal analysis pending)*
  - Llama-3.2-1B vs. Salamandra-2B *(pending Salamandra analysis)*
- Metric: Spearman rank correlation coefficient
- Hypothesis: ρ > 0.5 indicates consistent bias localization across architectures

**F. Cross-Bias Overlap Analysis (RQ2)** ⬜ (partial)
- Top candidate intersection across all categories computed interactively
- Formal Jaccard Index calculation and visualization pending
- Preliminary result: very low N-way overlap confirms category-specific circuits

#### Deliverables:
- [X] Prompt pair datasets (HuggingFace: EN + ES)
- [X] Activation capture notebook (`03_neuron_bias_detection_en.ipynb`)
- [X] Neuron ranking files — Llama-1B EN/ES, Llama-3B EN/ES → `results/neuron_analysis/`
- [X] `comparison_summary.json` per model/dataset
- [ ] Neuron ranking files — Salamandra-2B
- [ ] Qualitative generation notebook + outputs
- [ ] Cross-model correlation analysis (plots + stats)
- [ ] Cross-bias overlap analysis (Jaccard Index + plots)

---

### ⬜ 2.2. Fairness-Pruned Model Generation

**Estimation:** 10-15 hours  
**Description:** Apply fairness-guided neuron pruning and generate model checkpoints

#### Subtasks:

**A. Define Pruning Levels**
- **Level 0:** 0% (baseline — no pruning)
- **Level 1:** 5% of MLP neurons (conservative)
- **Level 2:** 10% of MLP neurons (moderate)
- **Level 3:** 15% of MLP neurons (aggressive)
- **Level 4:** 20% of MLP neurons (very aggressive — expect degradation)

**B. Selective Layer Pruning Strategy**
- Based on Phase 2.1 findings, pruning will be **category-informed**:
  - PhysicalAppearance + Gender → focus on layers 0-1
  - Religion → focus on layer 14 (1B) / 25 (3B)
  - Combined: prune layers identified by fairness scores, skip first and last layers as a safety measure
- For Llama-3.2-1B (16 layers): candidate range layers 1-14

**C. Apply Pruning with OptiPFair**

```python
import optipfair as opf

for pruning_pct in [5, 10, 15, 20]:
    pruned_model, stats = opf.prune_model(
        model=base_model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=pruning_pct,
        layer_indices=list(range(1, 15)),  # skip first and last
        show_progress=True,
        return_stats=True
    )
    output_path = f"models/llama-3.2-1b-fairness-{pruning_pct}pct"
    pruned_model.save_pretrained(output_path)
```

**D. Sanity Checks**
- Quick perplexity on WikiText sample (100 examples)
  - If perplexity > 2× baseline → reduce aggressiveness
- Manual generation test: 5 diverse prompts
- Parameter count verification

#### Deliverables:
- [ ] 4 pruned variants per model (up to 12 total)
- [ ] Pruning statistics JSON files
- [ ] Sanity check results document

---

## Phase 3: Experimental Evaluation (Post-Intervention)

**Objective:** Quantify bias reduction and capability retention

### ⬜ 3.1. Bias Re-evaluation (Pruned Models)

**Estimation:** 20-25 hours

#### Subtasks:

**A. BBQ Re-evaluation (English)**
- Run BBQ on all pruned models (4 levels × 3 base models = 12 evaluations)
- Same metrics as baseline: Bias Score, Accuracy, per-category breakdown

**B. EsBBQ Re-evaluation (Spanish)**
- Run EsBBQ on all pruned models
- Critical analysis: does English-guided pruning transfer to Spanish bias reduction?

**C. Metric Calculations**

```
Δ_Bias_BBQ  = (BiasScore_base - BiasScore_pruned) / BiasScore_base × 100
Δ_Bias_EsBBQ = (BiasScore_base - BiasScore_pruned) / BiasScore_base × 100
Δ_Accuracy  = (Accuracy_base - Accuracy_pruned) / Accuracy_base × 100
```

#### Deliverables:
- [ ] BBQ results for all pruned models
- [ ] EsBBQ results for all pruned models
- [ ] Delta metrics spreadsheet

---

### ⬜ 3.2. General Capabilities Re-evaluation (Pruned Models)

**Estimation:** 20-25 hours

#### Subtasks:

**A. Core Capability Benchmarks**

Re-run on all pruned models:
- **MMLU** — General knowledge (5-shot)
- **Global-MMLU (ES)** — Spanish knowledge (5-shot)
- **GSM8K** — Math reasoning (5-shot, fragile metric)
- **ARC-Challenge** — Scientific reasoning (0-shot)
- **HellaSwag** — Commonsense (0-shot)
- **TruthfulQA MC2** — Hallucinations (0-shot)

**B. Acceptance Criteria (RQ3)**
- ✅ Acceptable: Δ_Capability < 5%
- ⚠️ Borderline: 5% ≤ Δ_Capability < 10%
- ❌ Unacceptable: Δ_Capability ≥ 10%

**C. Trade-off Efficiency Score**

```
Efficiency_Score = Δ_Bias / (Δ_Capability + 1)
```

**D. Statistical Significance**
- Bootstrap confidence intervals (95% CI) for Δ_Bias and Δ_Capability
- scipy.stats or custom bootstrap implementation

#### Deliverables:
- [ ] Capabilities results for all pruned models
- [ ] Delta metrics spreadsheet
- [ ] Statistical significance results
- [ ] Trade-off efficiency scores

---

## Phase 4: Results Analysis & Visualization

**Objective:** Extract scientific insights from experimental data

### ⬜ 4.1. Data Consolidation & Visualization

**Estimation:** 25-30 hours

#### Subtasks:

**A. Master Results Table**

| Model | Pruning % | Params Removed | Δ Bias (BBQ) | Δ Bias (EsBBQ) | Δ Accuracy (BBQ) | Δ MMLU | Δ Global-MMLU | Efficiency Score |
|-------|-----------|----------------|--------------|----------------|------------------|--------|---------------|-----------------|
| Llama-1B | 5% | X.X% | +X.X% | +X.X% | -X.X% | -X.X% | -X.X% | X.XX |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**B. Pareto Frontier Visualization**
- X-axis: Bias Reduction (%) — BBQ Δ_Bias
- Y-axis: Capability Loss (%) — MMLU Δ_Capability
- One plot per model + one combined
- Annotate "sweet spot" pruning levels

**C. Cross-Lingual Transfer Analysis**
- Compare Δ_Bias_BBQ vs. Δ_Bias_EsBBQ
- Hypothesis: high correlation (ρ > 0.7) if bias is architectural, not linguistic

**D. Layer-wise Bias Contribution Heatmap**
- Rows = layers, Columns = bias categories
- Color intensity = mean neuron FairnessPruningScore per layer
- Already partially available from Phase 2.1 findings

**E. Category-Specific Analysis**
- Bias reduction and capability impact per category
- Which biases are easier/harder to mitigate?

**F. Qualitative Analysis**
- Failure cases: prompts where pruning increased bias
- Success cases (minimum 10): stereotype → neutral/balanced output
- Pre/post comparison using generations from Phase 2.1D

#### Deliverables:
- [ ] Master results table (CSV + LaTeX)
- [ ] Pareto frontier plots
- [ ] Cross-lingual transfer plot
- [ ] Layer-wise bias heatmap
- [ ] Category-specific analysis plots
- [ ] Qualitative pre/post examples document

---

### ⬜ 4.2. Research Questions Validation

**Estimation:** 10 hours

**RQ1:** Spearman ρ between neuron rankings across models  
**RQ2:** Jaccard Index between top-candidate sets across bias categories  
**RQ3:** Pareto frontier analysis — identify pruning levels meeting ≥15% bias / ≥95% capabilities

#### Deliverables:
- [ ] RQ1 validation (stats + plots)
- [ ] RQ2 validation (stats + plots)
- [ ] RQ3 validation (stats + plots)
- [ ] Summary: key findings vs. hypotheses

---

## Phase 5: Thesis Writing

**Objective:** Produce the academic document

### ⬜ 5.1. Introduction & State of the Art

**Estimation:** 25 hours

**A. Introduction (~5 pages)**
- Motivation: bias in LLMs + limitations of existing mitigation methods
- Gap: no prior work on bias localization via activation-guided MLP pruning
- Contribution: novel method + cross-lingual validation
- Thesis structure overview

**B. State of the Art (~15 pages)**
1. Bias in LLMs — types, measurement (BBQ, EsBBQ), real-world harms
2. Bias mitigation — pre/in/post-processing approaches; gap in structural pruning
3. Neural network pruning — structured vs. unstructured; magnitude-based vs. activation-based
4. Activation analysis & interpretability — mechanistic interpretability, feature circuits
5. Multilingual evaluation — native benchmarks vs. translations; cross-lingual bias patterns

#### Deliverables:
- [ ] Introduction draft
- [ ] State of the Art draft
- [ ] Reference list (BibTeX)

---

### ⬜ 5.2. Experimental Methodology

**Estimation:** 15 hours

**Sections:**
- Method overview + flowchart: Baseline → Detection → Pruning → Evaluation
- OptiPFair algorithm: activation capture, FairnessPruningScore formula, pruning strategy
- Models & datasets: Llama-3.2-1B/3B, Salamandra-2B; BBQ, EsBBQ; prompt pair datasets
- Evaluation protocol: metrics, statistical testing, hardware/software environment

#### Deliverables:
- [ ] Methodology chapter draft
- [ ] Method flowchart
- [ ] Formal notation document

---

### ⬜ 5.3. Results & Discussion

**Estimation:** 25 hours

**Sections:**
- Results organized by RQ (RQ1, RQ2, RQ3)
- Per-model detailed results
- Cross-lingual transfer analysis
- Discussion: interpretation, comparison to related work, unexpected findings
- Limitations: dataset scope (Western-centric), model size (<4B), MLP-only pruning, no user studies
- Future work: attention pruning, larger models, more languages, dynamic pruning

#### Deliverables:
- [ ] Results chapter draft
- [ ] Discussion chapter draft
- [ ] Limitations + future work sections

---

### ⬜ 5.4. Conclusions & Abstract

**Estimation:** 10 hours

- Conclusions (~3 pages): restate contributions, answer RQs, broader implications
- Abstract (~300 words): background, gap, method, key results, conclusion
- Thesis formatting: UIMP template, TOC, figures, bibliography

#### Deliverables:
- [ ] Conclusions chapter
- [ ] Abstract (Spanish + English)
- [ ] Formatted thesis PDF

---

### ⬜ 5.5. TMLR / Conference Paper (Optional)

**Estimation:** 10-15 hours

- Target venue: TMLR or similar
- Focus: Llama-3.2-1B primary results + cross-model validation
- Length and format per venue guidelines

#### Deliverables:
- [ ] Paper draft (LaTeX)

---

## Phase 6: Defense Preparation

**Objective:** Prepare oral presentation

### ⬜ 6.1. Presentation Materials

**Estimation:** 15-20 hours

**Slide structure (~20 slides, 20 minutes):**
1. Title (1)
2. Motivation (2-3)
3. Research Questions (1)
4. Method Overview (3-4)
5. Experimental Setup (2)
6. Results — RQ1: localization (2)
7. Results — RQ2: cross-category overlap (2)
8. Results — RQ3: pruning effectiveness (3)
9. Key Findings (2)
10. Limitations & Future Work (1)
11. Conclusions (1)
12. Backup slides (5-10)

**Q&A preparation — anticipated questions:**
- "Why MLP pruning over attention?"
- "How do you know neurons encode bias vs. are merely correlated?"
- "Why only models <4B?"
- "Did you consider fine-tuning as an alternative?"
- "What about biases not captured by BBQ/EsBBQ?"
- "How does this scale to production?"

#### Deliverables:
- [ ] Presentation slides
- [ ] Defense script (~2500 words)
- [ ] Q&A preparation document

---

## Updated Timeline

### December 2025 — February 2026 ✔
- Phase 1 complete: capabilities + bias benchmarks (BBQ + EsBBQ) for all 3 models

### March — April 2026 (current)
- Phase 2.1 in progress:
  - Prompt pair datasets published ✔
  - Neuron analysis: Llama-1B + Llama-3B (EN + ES) ✔
  - Pending: Salamandra-2B, qualitative generations, formal cross-model analysis

### May 2026
- Complete Phase 2.1 (Salamandra + qualitative analysis)
- Phase 2.2: pruned model generation
- Phase 3: re-evaluation (bias + capabilities)

### June 2026
- Phase 4: results analysis & visualization
- Phase 5: thesis writing (all chapters)
- **June 24:** Defense request deadline

### July 2026
- **July 1:** Thesis submission (PDF via PoliformaT)
- **July 8-10:** Defense (videoconference, 20 min + Q&A)

> **September 2026 fallback:** Defense request Sep 9, submission Sep 16, defense Sep 23-25. Note: tutor unavailable during August.

---

## Technical Infrastructure

### Hardware
- **Primary:** Google Colab Pro
  - GPU: NVIDIA L4 (23.7 GB VRAM, compute capability 8.9) — bfloat16
  - GPU: NVIDIA T4 (compute capability 7.5) — float16
  - dtype assigned automatically based on compute capability

### Software Stack
```
Python 3.10+
PyTorch 2.0+
Transformers (HuggingFace)
lm-evaluation-harness
optipfair[viz]
datasets (HuggingFace)
Matplotlib / Seaborn
Pandas / NumPy
SciPy
```

### Repository Structure (`results/`)

```
results/
├── benchmarks/
│   ├── bbq/                         # BBQ (EN) + EsBBQ (ES) baseline results
│   └── capabilities/                # lm-eval: MMLU, ARC, HellaSwag, GSM8K...
│
├── neuron_analysis/                 # bias_scores + fairness_scores (.json, .pt)
│   ├── llama-3.2-1b/
│   │   ├── fairness-pruning-pairs-en/
│   │   └── fairness-pruning-pairs-es/
│   └── llama-3.2-3b/
│       ├── fairness-pruning-pairs-en/
│       └── fairness-pruning-pairs-es/
│
└── generations/                     # Qualitative model outputs (pre-pruning)
    ├── llama-3.2-1b/
    │   ├── fairness-pruning-pairs-en/
    │   └── fairness-pruning-pairs-es/
    └── llama-3.2-3b/
        └── ...
```

> **Note:** `.pt` files (PyTorch tensors) are stored in Google Drive only and are not committed to GitHub. Only `.json` files are versioned.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Colab session timeout | Checkpoint after each model; upload to Drive immediately |
| Pruning causes model collapse | Start at 5%; sanity check perplexity after each level |
| RQ3 targets not met | Adjust targets; emphasize RQ1/RQ2 localization findings as main contribution |
| Running out of time for all 3 models | Prioritize Llama-1B; use 3B + Salamandra as validation |
| Writer's block | Write incrementally; start with Methodology (already well-defined) |

---

## Success Criteria

### Minimum Viable Thesis (Pass)
- [X] Complete baseline evaluation (Phase 1)
- [X] Neuron detection on at least 1 model (Phase 2.1, partial)
- [ ] Pruned models generated (Phase 2.2)
- [ ] Re-evaluation on bias + capabilities (Phase 3)
- [ ] Basic analysis answering RQs (Phase 4)
- [ ] Written thesis (Phase 5)

### Target Quality (8-9/10)
- All above +
- [ ] All 3 models evaluated
- [X] Preliminary cross-model comparison (Llama-1B vs. 3B)
- [ ] Statistical significance testing
- [ ] Publication-quality figures
- [ ] Thorough limitations discussion
- [ ] Cross-lingual transfer analysis

### Excellence (9.5-10/10, potential publication)
- All above +
- [ ] Novel insights from failure analysis
- [ ] Strong evidence for all 3 RQs
- [ ] TMLR paper submitted
- [ ] Code + models released on GitHub/HuggingFace
- [ ] Exceptional defense presentation

---

**Last Updated:** April 28, 2026  
**Status:** Phase 1 Complete ✔ | Phase 2.1 In Progress 🔄
