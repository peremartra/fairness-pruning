# Master's Thesis Roadmap: Fairness Pruning

**Title:** Fairness Pruning: Bias Mitigation through Activation-Guided MLP Width Pruning in Large Language Models

**Student:** Pere Martra  
**Program:** Master in Research in Artificial Intelligence (UIMP)  
**Target Models:** Llama-3.2-1B, Llama-3.2-3B, BSC-LT/Salamandra-2B  
**Timeline:** December 2025 - June 2026  
**Total Estimated Hours:** 230-260 hours (~10h/week over 25 weeks)

---

## Research Questions

**RQ1:** Is bias consistently localized in specific MLP neurons across different model architectures?  
**RQ2:** Do different bias categories (gender, race, age) share structural overlap in neural circuits?  
**RQ3:** Can we achieve ≥15% bias reduction while maintaining ≥95% capability retention?

---

## Phase 1: Baseline Establishment

**Objective:** Complete radiography of model behavior before intervention

### ✔ 1.1. General Capabilities Evaluation (Base Models)

**Status:** COMPLETED  
**Models:** Llama-3.2-1B, Llama-3.2-3B, Salamandra-2B  
**Benchmarks (COMPLETED):** Wikitext, Lambada, IFEval, GSM8K, MMLU, ARC, HellaSwag, TruthfulQA (MC2), Belebele, Global-MMLU, VeritasQA (ES/CA)
**Results:** Available in `results/`

#### Deliverables:
- [X] VeritasQA results for all 3 models
- [X] Updated capabilities baseline summary table

---

### [x] 1.2. Bias Benchmark Execution (Base)

**Estimation:** 15-20 hours  
**Description:** Establish baseline bias metrics using validated multilingual benchmarks

#### Subtasks:

**A. BBQ (Bias Benchmark for QA) - English**
- **Dataset:** 58,492 examples across 9 social dimensions
- **Source:** `lm-evaluation-harness` task: `bbq`
- **Metrics to collect:**
  - Bias Score: % stereotype preference in ambiguous contexts
  - Accuracy: % correct answers in disambiguated contexts
  - Category-level breakdown (gender, race, age, religion, etc.)
- **Models:** All 3 base models
- **Command:**
  ```bash
  lm_eval --model hf \
          --model_args pretrained=meta-llama/Llama-3.2-1B \
          --tasks bbq \
          --output_path results/bbq_base_llama_1b.json
  ```

**B. MBBQ (Multilingual BBQ) - Spanish**
- **Dataset:** ~11k examples (Spanish subset)
- **Source:** HuggingFace `Veranep/MBBQ` or GitHub implementation
- **Reason for MBBQ over EsBBQ:** Peer-reviewed (ACL 2024), culturally validated, includes control dataset
- **Metrics:** Same as BBQ (Bias Score + Accuracy)
- **Models:** All 3 base models
- **Integration:** 
  - Download from HuggingFace Datasets
  - Adapt evaluation script for `lm-eval` or standalone
- **Expected finding:** Spanish typically shows higher bias than English

**C. Execution & Storage**
- Run BBQ and MBBQ on all 3 models
- Save results in JSON format: `results/bias_baseline/`
- Create summary table with all bias metrics
- **Colab Pro consideration:** Use persistent sessions, save checkpoints frequently

**D. Validation**
- Verify results align with published baselines (if available)
- Document any anomalies or unexpected patterns
- Calculate 95% confidence intervals using bootstrap (if time permits)

#### Deliverables:
- [X] BBQ results for all 3 models
- [X] MBBQ results for all 3 models
- [X] Baseline bias summary table (CSV/Excel)
- [X] Brief analysis document noting initial observations

---

## Phase 2: Detection and Pruning (Intervention with OptiPFair)

**Objective:** Identify bias-encoding neurons and generate pruned model variants

### ⬜ 2.1. Neuron Detection via Differential Activation Analysis

**Estimation:** 25-30 hours  
**Description:** Use OptiPFair to map neurons that differentially activate under biased vs. neutral stimuli

#### Subtasks:

**A. Prompt Pair Design**
- Extract templates from BBQ/MBBQ datasets
- Create minimal pairs for each bias category:
  - **Gender:** "The man/woman worked as..." 
  - **Race:** "The white/Black person applied for..."
  - **Age:** "The young/elderly candidate presented..."
  - **Religion:** "The Christian/Muslim community organized..."
- **Target:** 50-100 pairs per category × 4 categories = 200-400 total pairs
- **Format:** Store in CSV: `prompt_stereotype, prompt_antistereotype, category, subcategory`

**B. Activation Capture with OptiPFair**
- Use OptiPFair's activation analysis functionality
- For each prompt pair, capture activations at:
  - MLP output (`mlp_output_layer_N`)
  - Gate projection (`gate_proj_layer_N`)
  - Up projection (`up_proj_layer_N`)
  - Down projection (`down_proj_layer_N`)
- Models: Start with Llama-3.2-1B (primary), then validate on 3B and Salamandra
- **Colab consideration:** Process in batches of 50 prompts, save intermediate results

**C. Neuron Scoring Metric Definition**

For each neuron `i` in layer `l`:

```
BiasScore_i = mean(|activation_stereotype - activation_antistereotype|) 
              across all prompt pairs

ImportanceScore_i = mean(|activation|) 
                    across general dataset (WikiText sample ~500 prompts)

FairnessPruningScore_i = BiasScore_i / (ImportanceScore_i + ε)
                         where ε = 1e-6 (numerical stability)

→ High score = high bias sensitivity, low general importance → PRUNE CANDIDATE
```

**D. Neuron Ranking Generation**
- Rank all neurons by `FairnessPruningScore_i`
- Generate ranking files per model: `rankings/llama_1b_neuron_scores.json`
- Visualize: Histogram of scores, identify "outlier" high-bias neurons
- Save top-K candidates for each pruning percentage level

**E. Cross-Model Consistency Analysis (RQ1)**
- Compare neuron rankings between:
  - Llama-3.2-1B vs. Llama-3.2-3B
  - Llama-3.2-1B vs. Salamandra-2B
- Metric: Spearman rank correlation coefficient
- Hypothesis: ρ > 0.5 indicates consistent bias localization
- Visualize: Scatter plot of neuron scores across models

**F. Cross-Bias Overlap Analysis (RQ2)**
- For each bias category, identify top 10% neurons
- Calculate Jaccard Index between categories:
  - J(gender, race), J(gender, age), J(race, religion), etc.
- Hypothesis: J > 0.3 indicates shared bias circuits
- Visualize: Venn diagram or heatmap of overlap

#### Deliverables:
- [ ] Prompt pair dataset (CSV)
- [ ] Activation capture code (Jupyter notebook)
- [ ] Neuron ranking files for all models
- [ ] Cross-model correlation analysis (plots + stats)
- [ ] Cross-bias overlap analysis (plots + stats)

---

### ⬜ 2.2. Fairness-Pruned Model Generation

**Estimation:** 10-15 hours  
**Description:** Apply neuron pruning masks and generate model checkpoints

#### Subtasks:

**A. Define Pruning Levels**
- **Level 0:** 0% (baseline - no pruning)
- **Level 1:** 5% of MLP neurons (conservative)
- **Level 2:** 10% of MLP neurons (moderate)
- **Level 3:** 15% of MLP neurons (aggressive)
- **Level 4:** 20% of MLP neurons (very aggressive - expect degradation)

**Rationale:** Proof-of-concept showed 22% bias reduction with 0.13% params removed. These levels are more aggressive to test boundaries.

**B. Selective Layer Pruning Strategy**
- **DO NOT prune:** 
  - First 2 layers (input processing critical)
  - Last 2 layers (output generation critical)
- **PRUNE:** Middle layers based on layer importance analysis
- For Llama-3.2-1B (16 layers): Prune layers 3-14
- Document layer selection rationale in methodology

**C. Apply Pruning with OptiPFair**

```python
from optipfair import prune_model

for pruning_pct in [5, 10, 15, 20]:
    pruned_model, stats = prune_model(
        model=base_model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",  # or custom FairnessPruningScore
        pruning_percentage=pruning_pct,
        layer_indices=list(range(3, 15)),  # selective pruning
        show_progress=True,
        return_stats=True
    )
    
    # Save model
    output_path = f"models/llama-3.2-1b-fairness-{pruning_pct}pct"
    pruned_model.save_pretrained(output_path)
    
    # Save statistics
    with open(f"{output_path}/pruning_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
```

**D. Sanity Checks**
- **Quick perplexity test:** Run on WikiText sample (100 examples)
  - If perplexity > 2× baseline → reduce aggressiveness
- **Manual generation test:** 5 diverse prompts, verify coherent output
- **Parameter count verification:** Confirm expected reduction matches stats

**E. Model Organization**
```
models/
├── llama-3.2-1b-fairness-5pct/
│   ├── model.safetensors
│   ├── config.json
│   ├── pruning_stats.json
│   └── neuron_mask.json
├── llama-3.2-1b-fairness-10pct/
├── llama-3.2-1b-fairness-15pct/
├── llama-3.2-1b-fairness-20pct/
└── [repeat for 3B and Salamandra]
```

**Colab Strategy:** 
- Prune and save models in sequence
- Upload to Google Drive after each pruning level
- Clear VRAM between models

#### Deliverables:
- [ ] 4 pruned variants per model (12 models total)
- [ ] Pruning statistics JSON files
- [ ] Sanity check results document
- [ ] Model organization in Drive/storage

---

## Phase 3: Experimental Evaluation (Post-Intervention)

**Objective:** Quantify bias reduction and capability retention

### ⬜ 3.1. Bias Re-evaluation (Pruned Models)

**Estimation:** 20-25 hours  
**Description:** Run complete bias evaluation suite on all pruned variants

#### Subtasks:

**A. BBQ Re-evaluation (English)**
- Run BBQ on all pruned models (4 levels × 3 base models = 12 evaluations)
- Collect same metrics as baseline: Bias Score, Accuracy, per-category breakdown
- **Colab strategy:** Batch evaluations, save after each model

**B. MBBQ Re-evaluation (Spanish)**
- Run MBBQ on all pruned models
- Compare Spanish bias scores to baseline
- **Critical analysis:** Does English-guided pruning transfer to Spanish?

**C. Metric Calculations**

For each pruned model, calculate:

```
Δ_Bias_BBQ = (BiasScore_base - BiasScore_pruned) / BiasScore_base × 100
Δ_Bias_MBBQ = (BiasScore_base - BiasScore_pruned) / BiasScore_base × 100
Δ_Accuracy_BBQ = (Accuracy_base - Accuracy_pruned) / Accuracy_base × 100
```

**D. Results Organization**
```
results/
├── bias_evaluation/
│   ├── baseline/
│   │   ├── bbq_llama_1b.json
│   │   └── mbbq_llama_1b.json
│   └── pruned/
│       ├── bbq_llama_1b_5pct.json
│       ├── bbq_llama_1b_10pct.json
│       └── ...
```

#### Deliverables:
- [ ] BBQ results for all 12 pruned models
- [ ] MBBQ results for all 12 pruned models
- [ ] Delta metrics spreadsheet (bias only)
- [ ] Initial observations document

---

### ⬜ 3.2. General Capabilities Re-evaluation (Pruned Models)

**Estimation:** 20-25 hours  
**Description:** Ensure models haven't been "lobotomized" - measure capability degradation

#### Subtasks:

**A. Core Capability Benchmarks**

Re-run on all pruned models:
- **MMLU (English):** General knowledge - 5-shot
- **Global-MMLU (Spanish):** Spanish knowledge - 5-shot
- **GSM8K:** Math reasoning - 5-shot (FRAGILE - expect drops)
- **ARC-Challenge:** Scientific reasoning - 0-shot
- **HellaSwag:** Commonsense - 0-shot
- **VeritasQA (ES/CA):** Truthfulness/hallucinations in Spanish/Catalan
  - Check if truthfulness degrades with pruning
  - Complements TruthfulQA with non-English evaluation

**B. Metric Calculations**

For each benchmark and pruning level:

```
Δ_Capability = (Score_base - Score_pruned) / Score_base × 100

Example:
If MMLU_base = 0.5783 and MMLU_pruned_10pct = 0.5550:
Δ_MMLU = (0.5783 - 0.5550) / 0.5783 × 100 = 4.03%
```

**C. Acceptance Criteria**

Based on RQ3 target (≥95% capability retention):
- ✅ **Acceptable:** Δ_Capability < 5%
- ⚠️ **Borderline:** 5% ≤ Δ_Capability < 10%
- ❌ **Unacceptable:** Δ_Capability ≥ 10%

**D. Statistical Significance Testing**

For main results:
- Bootstrap confidence intervals (95% CI) for Δ_Bias and Δ_Capability
- Report p-values to show changes are statistically significant
- Use scipy.stats or custom bootstrap implementation

**E. Trade-off Analysis**

For each model and pruning level, calculate:

```
Efficiency_Score = Δ_Bias / (Δ_Capability + 1)

→ Higher score = better trade-off (more bias reduction per unit capability loss)
```

#### Deliverables:
- [ ] MMLU/Global-MMLU results for all pruned models
- [ ] GSM8K/ARC/HellaSwag results for all pruned models
- [ ] VeritasQA results for all pruned models
- [ ] Delta metrics spreadsheet (capabilities)
- [ ] Statistical significance results
- [ ] Trade-off efficiency scores

---

## Phase 4: Results Analysis & Visualization

**Objective:** Extract scientific insights from experimental data

### ⬜ 4.1. Data Consolidation & Visualization

**Estimation:** 25-30 hours  
**Description:** Create publication-quality figures and tables

#### Subtasks:

**A. Master Results Table**

Create comprehensive table:

| Model | Pruning % | Params Removed | Δ Bias (BBQ) | Δ Bias (MBBQ) | Δ Accuracy (BBQ) | Δ MMLU | Δ Global-MMLU | Δ GSM8K | Efficiency Score |
|-------|-----------|----------------|--------------|---------------|------------------|--------|---------------|---------|------------------|
| Llama-1B | 5% | X.X% | +X.X% | +X.X% | -X.X% | -X.X% | -X.X% | -X.X% | X.XX |
| Llama-1B | 10% | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**B. Pareto Frontier Visualization**

Create 4 plots (one per model + one combined):
- **X-axis:** Bias Reduction (%) [BBQ Δ_Bias]
- **Y-axis:** Capability Loss (%) [MMLU Δ_Capability]
- **Points:** Each pruning level (5%, 10%, 15%, 20%)
- **Pareto optimal region:** Top-left quadrant (high bias reduction, low capability loss)
- **Annotations:** Label "sweet spot" pruning levels

**C. Cross-Lingual Transfer Analysis (RQ-related)**

Create visualization showing:
- **Question:** Do neurons detected via English prompts reduce Spanish bias?
- **Method:** Compare Δ_Bias_BBQ vs. Δ_Bias_MBBQ
- **Expected:** High correlation (ρ > 0.7) if bias is architectural
- **Plot:** Scatter plot with regression line

**D. Layer-wise Bias Contribution**

Visualize which layers contribute most to bias:
- **Heatmap:** Rows = layers, Columns = bias categories
- **Color intensity:** Mean neuron BiasScore per layer
- **Insight:** Do middle layers encode more bias than early/late?

**E. Category-Specific Analysis**

For each bias category (gender, race, age, religion):
- Bias reduction by category
- Capability impact by category
- Identify which biases are easier/harder to mitigate

**F. Failure Analysis**

Document cases where:
1. **Bias increased:** Examples where pruning made bias worse
2. **Capability collapsed:** Tasks that severely degraded
3. **Unexpected patterns:** Anomalies in results

Create qualitative examples table:
- Prompt
- Base model output
- Pruned model output
- Analysis

**G. Cherry-Picked Success Examples**

Minimum 10 examples showing:
1. **Bias reduced:** Stereotype → Neutral/Balanced
2. **Capability maintained:** Complex reasoning preserved
3. **Truthfulness preserved:** No hallucination increase

#### Deliverables:
- [ ] Master results table (Excel/CSV + LaTeX)
- [ ] Pareto frontier plots (4 total)
- [ ] Cross-lingual transfer plot
- [ ] Layer-wise bias heatmap
- [ ] Category-specific analysis plots
- [ ] Failure analysis document with examples
- [ ] Success examples document

---

### ⬜ 4.2. Research Questions Validation

**Estimation:** 10 hours  
**Description:** Explicitly answer RQs with evidence

#### Subtasks:

**RQ1: Is bias consistently localized across architectures?**
- Evidence: Cross-model neuron ranking correlation (from Phase 2.1)
- Metric: Spearman ρ between Llama-1B/3B, Llama-1B/Salamandra
- Threshold: ρ > 0.5 = consistent localization
- Conclusion statement with statistical backing

**RQ2: Do bias categories share structural overlap?**
- Evidence: Jaccard Index between bias categories (from Phase 2.1)
- Metric: J(category_A, category_B) for all pairs
- Threshold: J > 0.3 = significant overlap
- Implication: Shared vs. specialized bias circuits

**RQ3: Can we achieve ≥15% bias reduction with ≥95% capability retention?**
- Evidence: From Pareto frontier and master table
- Identify pruning level(s) that meet criteria
- Report across all 3 models
- If criteria not met: discuss why and adjust expectations

#### Deliverables:
- [ ] RQ1 validation document with stats
- [ ] RQ2 validation document with stats
- [ ] RQ3 validation document with stats
- [ ] Summary: Key findings vs. hypotheses

---

## Phase 5: Thesis Writing & IEEE Paper

**Objective:** Produce academic documents

### ⬜ 5.1. Introduction & State of the Art

**Estimation:** 25 hours

#### Subtasks:

**A. Introduction (~5 pages)**
- Motivation: Bias in LLMs is harmful + existing mitigation methods have limitations
- Gap: No work on bias localization via activation-guided MLP pruning
- Contribution: Novel method + cross-lingual validation
- Thesis structure overview

**B. State of the Art (~15 pages)**

Topics to cover:
1. **Bias in LLMs:**
   - Types of bias (Gallegos et al. 2024 survey)
   - Measurement approaches (BBQ, MBBQ papers)
   - Harms and real-world implications

2. **Bias Mitigation Techniques:**
   - Pre-processing (data debiasing)
   - In-training (adversarial objectives, RLHF)
   - Intra-processing (attention steering)
   - Post-processing (output filtering)
   - **Gap:** No work on structural pruning for bias

3. **Neural Network Pruning:**
   - Structured vs. unstructured pruning
   - Magnitude-based vs. activation-based pruning
   - Existing work: WANDA, SparseGPT, etc.
   - **Gap:** Pruning for capabilities, not fairness

4. **Activation Analysis & Interpretability:**
   - Mechanistic interpretability in LLMs
   - Sparse activations and feature circuits
   - Neuron-level analysis

5. **Multilingual Evaluation:**
   - Justification for native benchmarks (MBBQ, VeritasQA) vs. translations
   - Cross-lingual bias patterns
   - Language-specific vs. architectural bias

#### Deliverables:
- [ ] Introduction draft
- [ ] State of the Art draft
- [ ] Reference list (BibTeX)

---

### ⬜ 5.2. Experimental Methodology

**Estimation:** 15 hours

#### Subtasks:

**A. Method Overview (~3 pages)**
- Flowchart: Baseline → Detection → Pruning → Evaluation
- Formal problem definition
- Hypothesis statement

**B. OptiPFair Algorithm Documentation (~4 pages)**
- Activation capture mechanism
- Neuron scoring formula (FairnessPruningScore)
- Pruning strategy (selective layer pruning)
- Implementation details

**C. Models & Datasets (~3 pages)**
- Model architectures (Llama-3.2-1B/3B, Salamandra-2B)
- Training details (from model cards)
- Evaluation datasets (BBQ, MBBQ, VeritasQA)
- Prompt pair design

**D. Evaluation Protocol (~3 pages)**
- Metrics definition (Bias Score, Accuracy, Δ metrics)
- Statistical testing approach
- Hardware/software environment (Colab Pro, PyTorch, etc.)

#### Deliverables:
- [ ] Methodology chapter draft
- [ ] Flowchart/diagram of method
- [ ] Formal notation document

---

### ⬜ 5.3. Results & Discussion

**Estimation:** 25 hours

#### Subtasks:

**A. Results Section (~10 pages)**
- Present all tables and figures from Phase 4
- Organize by RQ:
  - RQ1 results + evidence
  - RQ2 results + evidence
  - RQ3 results + evidence
- Per-model detailed results
- Cross-lingual analysis
- Ablation studies (if any)

**B. Discussion (~8 pages)**
- Interpretation of main findings
- Why did X pruning level work best?
- Why does cross-lingual transfer work/not work?
- Comparison to related work (if applicable)
- Unexpected findings and explanations

**C. Limitations (~3 pages)**

Be honest and thorough:
- **Dataset limitations:** BBQ/MBBQ are US/Western-centric
- **Model limitations:** Only tested on <4B models
- **Method limitations:** 
  - Only MLP pruning (not attention)
  - Static analysis (no dynamic adaptation)
- **Evaluation limitations:**
  - Benchmarks may not capture all real-world harms
  - No user studies
- **Generalization limits:**
  - Only 3 models tested
  - Only 2 languages (EN, ES)

**D. Future Work (~2 pages)**
- Extend to attention mechanism pruning
- Larger models (7B, 13B, 70B)
- More languages (via MBBQ extensions: Dutch, Turkish, etc.)
- Combine with other debiasing techniques (e.g., RLHF)
- Dynamic pruning (adapt to specific use cases)
- User studies to validate real-world impact

#### Deliverables:
- [ ] Results chapter draft
- [ ] Discussion chapter draft
- [ ] Limitations section
- [ ] Future work section

---

### ⬜ 5.4. Conclusions & Abstract

**Estimation:** 10 hours

#### Subtasks:

**A. Conclusions (~3 pages)**
- Restate main contributions
- Answer each RQ definitively
- Broader implications for AI fairness
- Final thoughts on bias localization hypothesis

**B. Abstract (~300 words)**
- Background (1-2 sentences)
- Gap/Problem (1 sentence)
- Method (2-3 sentences)
- Key results (2-3 sentences)
- Conclusion (1 sentence)

**C. Thesis Formatting**
- Follow UIMP thesis template
- Table of contents, list of figures, list of tables
- Appendices (if needed - e.g., full prompt lists)
- Bibliography (BibTeX → formatted)

#### Deliverables:
- [ ] Conclusions chapter
- [ ] Abstract (Spanish + English)
- [ ] Formatted thesis PDF

---

### ⬜ 5.5. IEEE Conference Paper (Optional)

**Estimation:** 10 hours  
**Description:** Condensed version for potential publication

#### Subtasks:

- **Length:** 6-8 pages (IEEE format)
- **Content:**
  - Introduction (1 page)
  - Method (2 pages)
  - Results (2 pages)
  - Discussion + Conclusion (1 page)
- **Focus:** Main results from Llama-3.2-1B (most extensive experiments)
- **Figures:** 3-4 key plots (Pareto frontier, cross-lingual transfer, etc.)

#### Deliverables:
- [ ] IEEE paper draft (LaTeX)

---

## Phase 6: Defense Preparation

**Objective:** Prepare oral presentation

### ⬜ 6.1. Presentation Materials

**Estimation:** 15-20 hours

#### Subtasks:

**A. Slide Design (20-minute presentation)**

Recommended structure (~20 slides):
1. **Title slide** (1)
2. **Motivation** (2-3): Why bias matters + current limitations
3. **Research Questions** (1)
4. **Method Overview** (3-4): OptiPFair, neuron detection, pruning
5. **Experimental Setup** (2): Models, datasets, metrics
6. **Results - RQ1** (2): Consistent localization evidence
7. **Results - RQ2** (2): Cross-bias overlap evidence
8. **Results - RQ3** (3): Pareto frontier, main trade-offs
9. **Key Findings** (2): Highlight best pruning level, cross-lingual transfer
10. **Limitations & Future Work** (1)
11. **Conclusions** (1)
12. **Backup slides** (5-10): Detailed results, extra analysis

**B. Defense Script**
- Write full script (~2500 words for 20 min)
- Time each section
- Prepare transitions
- Memorize key phrases, not full text

**C. Q&A Preparation**

Anticipate questions:
- "Why did you choose MLP pruning over attention?"
- "How do you know neurons are actually encoding bias vs. just correlated?"
- "Why only small models (<4B)?"
- "Did you consider fine-tuning as an alternative?"
- "What about biases not captured by BBQ?"
- "How does this scale to production systems?"

Prepare concise, honest answers.

**D. Practice Runs**
- Practice alone (3-5 times)
- Practice with tutor/colleagues (2 times)
- Time strictly (stay under 20 minutes)
- Record yourself to identify weak points

#### Deliverables:
- [ ] Presentation slides (PowerPoint/Keynote/Beamer)
- [ ] Defense script document
- [ ] Q&A preparation document
- [ ] Practice session notes

---

## Timeline & Milestones

### December 2025
- ⚠️ Phase 1.1 partially completed (VeritasQA pending)
- Week 1: Complete Phase 1.1 (VeritasQA baseline)
- Week 2-3: Phase 1.2 (Bias benchmarks: BBQ + MBBQ)

### January 2026
- Week 3-4: Phase 2.1 (Neuron detection)
- Week 5-6: Phase 2.2 (Model pruning)

### February 2026
- Week 7-8: Phase 3.1 (Bias re-evaluation)
- Week 9-10: Phase 3.2 (Capabilities re-evaluation)

### March 2026
- Week 11-12: Phase 4.1 (Data analysis & visualization)
- Week 13: Phase 4.2 (RQ validation)

### April 2026
- Week 14-15: Phase 5.1 (Intro & SoTA writing)
- Week 16-17: Phase 5.2 (Methodology writing)

### May 2026
- Week 18-19: Phase 5.3 (Results & Discussion writing)
- Week 20-21: Phase 5.4 (Conclusions & formatting)
- Week 22: Phase 5.5 (Optional IEEE paper)

### June 2026
- Week 23-24: Phase 6.1 (Defense preparation)
- **June 24:** Thesis defense request deadline
- **July 1:** Thesis submission deadline
- **July 8-10:** Defense

---

## Technical Infrastructure

### Hardware
- **Primary:** Google Colab Pro
  - GPU: A100 (when available) or V100
  - RAM: High-RAM runtime
  - **Limitations:** 
    - Session timeout (use persistent checkpointing)
    - Compute units cap (monitor usage)

### Software Stack
```
Python 3.10+
PyTorch 2.0+
Transformers (HuggingFace)
lm-evaluation-harness
optipfair (your library)
Datasets (HuggingFace)
Matplotlib/Seaborn (visualization)
Pandas/NumPy (data processing)
SciPy (statistics)
```

### Data Management
```
Google Drive structure:
/TFM_Fairness_Pruning/
├── models/                  # Pruned model checkpoints
├── results/                 # JSON evaluation results
├── data/                    # Datasets (BBQ, MBBQ, prompts)
├── analysis/                # Plots, tables, notebooks
├── thesis/                  # LaTeX/Word document
└── code/                    # Scripts, notebooks
```

---

## Risk Mitigation

### Technical Risks

**Risk:** Colab session timeout during long evaluations  
**Mitigation:** 
- Use persistent checkpointing (save after each model)
- Run evaluations in batches
- Upload results to Drive immediately

**Risk:** BBQ/MBBQ not available in lm-eval  
**Mitigation:**
- Prepare standalone evaluation scripts
- Verify dataset availability before Phase 1.2

**Risk:** Pruning causes model collapse  
**Mitigation:**
- Start with conservative pruning (5%)
- Sanity checks after each pruning level
- Keep baseline as fallback

### Scope Risks

**Risk:** Results don't meet RQ3 target (≥15% bias reduction, ≥95% capability)  
**Mitigation:**
- Adjust targets if needed (document in limitations)
- Emphasize localization findings (RQ1, RQ2) as main contribution
- Frame as exploratory study establishing bounds

**Risk:** Running out of time for all 3 models  
**Mitigation:**
- Prioritize Llama-3.2-1B (most comprehensive)
- Llama-3.2-3B and Salamandra as validation (less extensive)
- IEEE paper focuses on 1B only

### Writing Risks

**Risk:** Writer's block or slow progress  
**Mitigation:**
- Write incrementally (don't wait until end)
- Start with easiest sections (Methodology, Results)
- Get tutor feedback early and often

---

## Success Criteria

### Minimum Viable Thesis (Pass)
- ✅ Complete baseline evaluation (Phase 1)
- ✅ Neuron detection on at least 1 model (Phase 2.1)
- ✅ Pruned models generated (Phase 2.2)
- ✅ Re-evaluation on bias + capabilities (Phase 3)
- ✅ Basic analysis answering RQs (Phase 4)
- ✅ Written thesis (Phase 5)

### Target Quality (8-9/10)
- All above +
- ✅ All 3 models evaluated
- ✅ Statistical significance testing
- ✅ Publication-quality figures
- ✅ Thorough limitations discussion
- ✅ Cross-lingual transfer analysis

### Excellence (9.5-10/10, potential publication)
- All above +
- ✅ Novel insights from failure analysis
- ✅ Strong evidence for all 3 RQs
- ✅ IEEE paper submitted/accepted
- ✅ Code + models released on GitHub/HuggingFace
- ✅ Exceptional defense presentation

---

## Notes & Considerations

1. **Colab Pro Management:** Monitor compute units closely. Pause non-essential runs.

2. **Version Control:** Use Git for code, Google Drive for results. Commit frequently.

3. **Tutor Communication:** Update Prof. Ureña every 2-3 weeks. Ask for feedback on drafts early.

4. **Reproducibility:** Document all hyperparameters, random seeds. Save all configs.

5. **VeritasQA Integration:** If integration is complex (>5 hours), consider descoping. Truthfulness is secondary to bias.

6. **Realistic Expectations:** Not every experiment will work perfectly. Document failures honestly - reviewers appreciate transparency.

7. **Self-Care:** 10h/week is sustainable. Don't burn out. Take breaks.

---

**Last Updated:** December 6, 2024  
**Status:** Phase 1.1 Complete, Phase 1.2 In Progress
