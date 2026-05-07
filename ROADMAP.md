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

### ✅ 2.1. Neuron Detection via Differential Activation Analysis

**Status:** COMPLETED
**Estimation:** 25-30 hours
**Description:** Use OptiPFair to map neurons that differentially activate under biased vs. neutral stimuli

#### Subtasks:

**A. Prompt Pair Dataset** ✅
- Minimal pairs for each bias category, each pair differing only in the demographic attribute
- **Hard constraint:** Both prompts in a pair must tokenize to the same length (Llama-3.2-1B tokenizer), required for position-by-position activation comparison
- Token verification script: `scripts/token_verification.py`
- Published on HuggingFace:
  - English: `oopere/fairness-pruning-pairs-en` (75 pairs, 6 categories)
  - Spanish: `oopere/fairness-pruning-pairs-es` (100 pairs, 5 categories)
- Full construction process documented in `datasets/README.md` and replicable from `datasets/` in the repository
- Categories: PhysicalAppearance, Gender, Age, RaceEthnicity, Religion, SES (EN only)
- Social contexts per template: labour, institutional, healthcare, social, educational

**B. Activation Capture and Neuron Scoring with OptiPFair** ✅
- Library: `optipfair` — `analyze_neuron_bias` + `compute_fairness_pruning_scores`
- For each prompt pair, activations captured at:
  - Gate projection (`gate_proj_layer_N`)
  - Up projection (`up_proj_layer_N`)
- Aggregation: `mean` across sequence positions
- Models completed: Llama-3.2-1B ✅, Llama-3.2-3B ✅, Salamandra-2B ✅

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

**C. Bias Path Analysis** ✅
- Notebook: `notebooks/05_bias_path_analysis.ipynb`
- Covers Sprints 1–4: data consolidation, Layer × Category heatmaps, Jaccard overlap, cross-lingual consistency, cross-model comparison
- Input: `results/neuron_analysis/{model}/{lang}/{Category}_bias_scores.json`
- Output: `results/figures/bias_path/` (PNG + PDF, 300 dpi)

**Sprint 1 — Data Consolidation, Validation & Normalization** ✅
- Load and validate all bias score files for all models and languages
- Min-Max normalization per category (global across all layers and neurons)
- Top-K extraction at three thresholds: Top-0.1%, Top-1%, Top-5%

**Sprint 2 — Bias Path Visualization** ✅
- Layer × Category heatmap per model and language: three panels (gate_proj | up_proj | GLU combined)
- Cross-model comparison heatmap: Llama-1B vs Llama-3B vs Salamandra-2B side by side on relative depth axis
- Layer × Neuron-bin heatmap: spatial localization of bias signal within each layer, for all model × language × category combinations (bin_size=512)
- Violin plots: removed — information more effectively conveyed by neuron heatmaps

**Sprint 3 — Overlap Analysis** ✅
- Jaccard Index matrices between all category pairs, at Top-0.1%, Top-1% and Top-5%
- N-way intersection across all categories simultaneously
- Cross-lingual consistency: EN vs ES Jaccard per category and model

**Sprint 4 — Cross-Model Summary** ✅
- Dominant layers table by model, category and projection (gate_proj, English)
- Visual cross-model comparison figures already generated in Sprint 2

**Key findings from Phase 2.1:**

| Finding | Detail |
|---------|--------|
| Dominant layers | Consistently final ~25% of each model across all architectures: L13-L15 (1B), L24-L27 (3B), L21-L23 (Salamandra) |
| PhysicalAppearance | Dominant category in all models and languages; coherent signal across both GLU projections → strongest pruning candidate |
| Gender | Sparse high-intensity encoding; few extreme outlier neurons, largely language-agnostic (highest cross-lingual Jaccard: 0.29-0.32) |
| Age | Weakest and least consistent category; signal not jointly sustained by both projections |
| Category circuits | Near-zero pairwise Jaccard (< 0.16) confirms category-specific bias circuits, not a universal bias circuit |
| N-way shared neurons | 24–68 neurons (Top-1%) shared across all categories simultaneously — less than 0.05% of the neuron pool |
| Salamandra Religion (ES) | Religion becomes dominant category at deepest layers in Spanish — unique inversion not observed in Llama models |
| Cross-lingual | Bias localization pattern (dominant layers, category ranking) is structurally consistent across languages, but specific neurons differ |

**D. Cross-Model Consistency Analysis (RQ1)** ✅
- Visual and numerical evidence from Sprint 2 (side-by-side heatmaps) and Sprint 4 (dominant layers table)
- Dominant layer positions are scale-invariant across all three architectures
- Spearman ρ between neuron rankings excluded: direct neuron-to-neuron comparison across architectures with different layer counts and neuron pool sizes requires non-trivial alignment methodology that would not add substantive information beyond the visual and tabular evidence already available

**E. Cross-Bias Overlap Analysis (RQ2)** ✅
- Jaccard Index computed for all category pairs at three thresholds, for all models and languages
- N-way intersection confirms: less than 0.05% of neurons are shared across all five categories simultaneously
- Finding: bias circuits are category-specific, not universal — each demographic bias type resides in a distinct subset of MLP neurons

#### Deliverables:
- [x] Prompt pair datasets (HuggingFace: EN + ES)
- [x] Dataset construction documented in `datasets/README.md`, replicable from repository
- [x] Neuron ranking files — Llama-1B EN/ES, Llama-3B EN/ES, Salamandra-2B EN/ES → `results/neuron_analysis/`
- [x] `comparison_summary.json` per model/dataset
- [x] Bias path analysis notebook (`notebooks/05_bias_path_analysis.ipynb`)
- [x] Layer × Category heatmaps — all models and languages
- [x] Cross-model comparison heatmaps (relative depth axis)
- [x] Layer × Neuron-bin heatmaps — all model × language × category combinations
- [x] Jaccard overlap matrices — all models, languages and thresholds
- [x] N-way intersection analysis
- [x] Cross-lingual consistency table (`cross_lingual_overlap.csv`)
- [x] Dominant layers summary (`dominant_layers_summary.csv`)
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

**Objective:** Produce the academic document in IEEE double-column format (12–14 pages), targeting UIMP defense in July 2026 and subsequent TMLR submission.

**Language note:** Draft in Spanish; English translation at final stage before submission.

---

### ⬜ 5.1. Abstract

**Estimation:** 2 hours  
**Target length:** ~half column (~150 words)

- Background: bias in LLMs and limitations of post-hoc mitigation
- Gap: no prior work on activation-guided MLP pruning for fairness
- Method: differential activation analysis + FairnessPruningScore + selective pruning
- Key results: bias reduction achieved, capability retention, cross-lingual transfer
- One-sentence conclusion

#### Deliverables:
- [ ] Abstract draft (Spanish)
- [ ] Abstract final (English)

---

### ⬜ 5.2. Section 1 — Introduction

**Estimation:** 5 hours  
**Target length:** ~1 page (2 columns)

**Subsections:**
- **1.1 Motivation** — bias in LLMs: documented harms, real-world deployment risks, limitations of existing mitigation approaches (fine-tuning, prompting, post-hoc filters)
- **1.2 Gap** — no prior work characterizes demographic bias as spatially localized in MLP neurons that can be surgically removed; existing interpretability work stops at description, not intervention
- **1.3 Contributions** — three explicit contributions: (1) bias localization evidence across three architectures and two languages, (2) novel FairnessPruningScore combining bias sensitivity and structural importance, (3) empirical validation of ≥15% bias reduction with <5% capability degradation
- **1.4 Document structure** — one-paragraph roadmap of the remaining sections

#### Deliverables:
- [ ] Section 1 draft

---

### ⬜ 5.3. Section 2 — Related Work

**Estimation:** 10 hours  
**Target length:** ~1.5 pages

**Subsection 2.1 — Demographic Bias & Mitigation Strategies**
- Taxonomy of bias types in LLMs (representation, allocation, stereotype)
- Measurement benchmarks: BBQ (Parrish et al., 2022), EsBBQ (Ruiz-Fernández et al., 2025) — justify why these two and not others (CrowS-Pairs excluded: methodological justification)
- Mitigation landscape: pre-processing, in-training (RLHF, DPO), post-processing; gap in structural/surgical approaches
- Why existing methods are insufficient for deployment-time surgical correction

**Subsection 2.2 — Pruning & Interpretability**
- Structured vs. unstructured pruning; magnitude-based (Han et al.) vs. activation-based; GLU-specific methods
- OptiPFair and PPM/MAW method (Martra, 2025 — arXiv:2512.22671); FairnessPruningScore formula as extension
- Mechanistic interpretability: feature circuits (Anthropic Transformer Circuits), neuron-level analysis
- Gap: interpretability work describes circuits, pruning work targets efficiency — this work is the first to connect both for fairness

#### Deliverables:
- [X] Section 2 draft
- [ ] BibTeX reference list (target: 25–35 references)

---

### ⬜ 5.4. Section 3 — Methodology

**Estimation:** 10 hours  
**Target length:** ~2 pages

**Subsection 3.1 — Pipeline Overview**
- End-to-end flow: Baseline → Prompt Pair Construction → Activation Capture → Neuron Scoring → Pruning → Evaluation
- Pipeline figure (to be produced as vector graphic, placed here)

**Subsection 3.2 — Prompt Pair Datasets**
- Token-equality hard constraint and its methodological justification
- Construction procedure: candidate definition → token verification → template design
- Dataset statistics: EN (75 pairs, 6 categories), ES (100 pairs, 5 categories)
- Published on HuggingFace; SES exclusion from Spanish justified; CrowS-Pairs exclusion justified here

**Subsection 3.3 — Neuron Bias Detection**
- GLU combined score (gate × up, normalized): methodological justification over analyzing projections separately
- `analyze_neuron_bias`: mean absolute differential activation, aggregation across pairs
- FairnessPruningScore formula with notation:
$$\text{FPS}_i = \alpha \cdot \hat{B}_i + (1-\alpha) \cdot (1 - \hat{I}_i), \quad \alpha = 0.45$$
- Importance score via PPM (static, weight-based); why activation-based importance was not used

**Subsection 3.4 — Fairness Pruning**
- Selective layer pruning via `layer_indices` in OptiPFair
- Pruning levels: 5%, 10%, 15%, 20%
- Layer selection rationale derived from Phase 2.1 findings (dominant layers ~final 25%)
- Decision on symmetric pruning across selected layers (asymmetric discarded for vLLM compatibility — forward-reference to Section 6.2)

**Subsection 3.5 — Evaluation Protocol**
- Capabilities benchmarks: suite of 12 tasks via lm-evaluation-harness (EN + ES, 0-shot and 5-shot as appropriate)
- Bias benchmarks: BBQ + EsBBQ, same configuration as baseline
- Delta metrics: Δ_Bias and Δ_Capability as percentage change from baseline
- Trade-off efficiency score: Δ_Bias / (Δ_Capability + 1)
- Statistical testing: bootstrap CI (95%) for all reported deltas

#### Deliverables:
- [X] Section 3 draft
- [ ] Pipeline figure (vector, IEEE-ready)
- [ ] Formal notation document

---

### ⬜ 5.5. Section 4 — Experimental Setup

**Estimation:** 3 hours  
**Target length:** ~0.5 pages

- **Models:** Llama-3.2-1B (primary — 16 layers, 8192 intermediate), Llama-3.2-3B (validation), Salamandra-2B (cross-lingual validation)
- **Hardware:** Google Colab Pro — NVIDIA L4 (bfloat16) for Llama-3B and Salamandra; T4 (float16) for Llama-1B
- **Software:** PyTorch 2.x, HuggingFace Transformers, lm-evaluation-harness, optipfair v0.3.x
- **Reproducibility:** HuggingFace datasets published, tokenizer version pinned (Llama-3.2-1B), seeds fixed, generation config documented (greedy, repetition_penalty=1.1)

#### Deliverables:
- [X] Section 4 draft

---

### ⬜ 5.6. Section 5 — Results

**Estimation:** 20 hours  
**Target length:** ~4–5 pages

**Subsection 5.1 — Baseline Capabilities and Bias**
- Capabilities table: 12 benchmarks × 3 models (EN + ES)
- BBQ + EsBBQ bias baseline: overall, ambiguous, disambiguated per model
- Cross-lingual accuracy shift (+11–13pp on EsBBQ): note and flag for later discussion

**Subsection 5.2 — Bias Localization Analysis (RQ1 + RQ2)**
- Layer × Category heatmaps: GLU combined score, one per model
- Key finding: dominant layers consistently in final ~25% across all three architectures
- Jaccard matrices: near-zero pairwise overlap confirms category-specific circuits (RQ2)
- N-way intersection: 24–68 shared neurons at Top-1% across all five categories simultaneously
- Cross-lingual consistency table: bias localization pattern structurally stable across EN and ES; specific neurons differ
- Salamandra anomaly: Religion becomes dominant category at deepest layers in Spanish

**Subsection 5.3 — Pruning Results (RQ3)**
- Results table: Δ_Bias (BBQ) × Δ_Capability (MMLU) × pruning level × model
- Pareto frontier figure: bias reduction vs. capability loss, annotated "sweet spot"
- Whether RQ3 target (≥15% bias, <5% capability) is met and at which pruning level
- Cross-lingual transfer: Δ_Bias (BBQ) vs. Δ_Bias (EsBBQ) — does EN-guided pruning reduce Spanish bias?
- Bootstrap CI table for primary results

**Subsection 5.4 — Cross-Model Validation**
- Llama-3B validation: does bias localization pattern hold at scale? Do pruning results generalize?
- Salamandra-2B validation: cross-architecture and cross-lingual robustness of the method
- Discussion of scale effects on bias concentration depth

#### Deliverables:
- [ ] Section 5 draft
- [ ] All result tables (CSV + LaTeX)
- [ ] Pareto frontier figure
- [ ] Cross-lingual transfer figure
- [ ] Bootstrap CI tables

---

### ⬜ 5.7. Section 6 — Discussion

**Estimation:** 5 hours  
**Target length:** ~1 page

**Subsection 6.1 — Two Encoding Strategies: Distributed vs. Sparse Bias**
- Contrast between categories with broad distributed signal (PhysicalAppearance, Age) and categories with sparse high-intensity encoding (Gender — few extreme outlier neurons)
- Implications for pruning: sparse encoding is harder to detect in layer averages but highly targetable once identified; distributed encoding requires broader intervention
- Connection to mechanistic interpretability literature

**Subsection 6.2 — Limitations**
- Dataset scope: Western-centric attributes and social contexts; limited coverage of non-binary gender, intersectional bias, Global South demographics
- Model size: all models <4B parameters; unknown generalization to larger-scale architectures
- MLP-only pruning: attention mechanisms not analyzed; potential bias residue in attention heads
- Asymmetric per-layer pruning discarded for vLLM compatibility — limits precision of intervention
- No user studies: bias reduction measured via benchmarks only; real-world impact unverified

#### Deliverables:
- [ ] Section 6 draft

---

### ⬜ 5.8. Section 7 — Conclusions

**Estimation:** 3 hours  
**Target length:** ~0.5 pages

**Subsection 7.1 — Answers to RQ1, RQ2, RQ3**
- RQ1: Yes — bias is spatially localized in the final ~25% of MLP layers; pattern is architecture-invariant across all three models
- RQ2: Yes — Jaccard < 0.16 across all category pairs; bias circuits are category-specific, not universal
- RQ3: [outcome pending Phase 3 results] — target ≥15% bias reduction with <5% capability degradation

**Subsection 7.2 — Implications and Future Work**
- Practical: surgical bias mitigation without retraining, deployable at inference time
- Theoretical: demographic bias is not uniformly distributed — it is structured and locatable
- Future work: attention mechanism analysis, SAEs for finer circuit decomposition (Anthropic Transformer Circuits), larger models (>7B), dynamic pruning at inference time, asymmetric per-layer pruning once vLLM compatibility constraints are relaxed

#### Deliverables:
- [ ] Section 7 draft
- [ ] Full formatted thesis PDF (IEEE template, Overleaf)
- [ ] Abstract final version (English, ~150 words)

---

### ⬜ 5.9. TMLR Paper (Post-Defense)

**Estimation:** 10–15 hours

- Condense thesis to TMLR format (~8–10 pages)
- Focus: Llama-3.2-1B primary results + cross-model validation as secondary
- Expand Related Work beyond what fits in the thesis
- Address anticipated reviewer questions (attention mechanisms, larger models, user studies)

#### Deliverables:
- [ ] TMLR paper draft (LaTeX)

---
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
