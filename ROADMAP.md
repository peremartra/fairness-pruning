# Fairness Pruning — Master's Thesis Roadmap

**Title:** Fairness Pruning: Bias Mitigation through Activation-Guided MLP Width Pruning in Large Language Models

**Student:** Pere Martra  
**Program:** UIMP Master in Research in Artificial Intelligence  
**Target:** First convocation defense (July 2026)  

---

## Research Questions

**RQ1:** Is bias consistently localized in specific MLP neurons across different model architectures?  
**RQ2:** Do different bias categories (gender, race, age) share structural overlap in neural circuits?  
**RQ3:** Can we achieve ≥15% bias reduction while maintaining ≥95% capability retention?

---

## Phase Overview

### ✅ Phase 1: Baseline Establishment — COMPLETE

**What was done:**
- Capabilities baselines: 11 benchmarks (WikiText, MMLU, GSM8K, ARC, HellaSwag, etc.) on 3 models
- Bias baselines: BBQ (English) + EsBBQ (Spanish) on all 3 models
- Results stored in `results/bias-benchmarks-base/` and `results/benchmarks/capabilities/`

**Deliverable:** Complete radiography of unpruned model behavior (bias + capabilities)

---

### ✅ Phase 2.1: Neuron Detection — COMPLETE

**What was done:**
- Prompt pair datasets published (HuggingFace: EN 75 pairs, ES 100 pairs)
- Bias scores computed for all models (Llama-1B, Llama-3B, Salamandra-2B)
- Bias path analysis notebook complete (`05_bias_path_analysis.ipynb`)
- Layer × Category heatmaps generated
- Jaccard overlap analysis (RQ2 evidence)
- Cross-lingual consistency verified

**Key findings:**
- Bias localized in final ~25% of each model (scale-invariant across architectures)
- Category-specific circuits confirmed (Jaccard < 0.16 pairwise overlap)
- 24–68 neurons shared across all categories at Top-1% threshold
- PhysicalAppearance is most consistent bias category across languages
- Salamandra shows unique Religion concentration in Spanish at deepest layers

**Deliverable:** Quantitative evidence for RQ1 (localization) and RQ2 (circuit separation)

---

### 🔄 Phase 2.2: Pruning Strategy & Model Generation — IN PROGRESS

**What needs to happen:**
- Define pruning configuration (levels, which models, which strategy)
- Generate pruned model variants
- Quick sanity checks (perplexity, inference stability)

**Decision needed before starting:**
- Pruning levels to test (e.g., 0.1%, 1%, 5%)
- Which model(s) to prune first (recommend Llama-1B as primary)
- Pruning strategy: superneuron zeroing vs. fairness-score guided vs. both

**Deliverable:** Pruned model checkpoints ready for evaluation

---

### ⬜ Phase 3: Post-Intervention Evaluation — PENDING

**What needs to happen:**
- Re-run BBQ + EsBBQ on pruned models
- Re-run capabilities benchmarks (MMLU, ARC, GSM8K, etc.)
- Calculate Δ_Bias and Δ_Capability deltas
- Bootstrap confidence intervals for statistical significance

**Deliverable:** Quantitative results table (bias reduction vs. capability loss trade-off)

---

### ⬜ Phase 4: Analysis & Visualization — PENDING

**What needs to happen:**
- Consolidate all results into master table
- Pareto frontier figure (bias vs. capability loss)
- Cross-lingual transfer analysis (EN bias reduction vs. ES bias reduction)
- Answer RQ3 quantitatively (did we hit ≥15% bias reduction + ≥95% capability?)

**Deliverable:** Publication-ready figures and evidence for all three RQs

---

### ⬜ Phase 5: Thesis Writing — IN PROGRESS (20% complete)

**What's drafted:**
- Introduction + motivation ✅ (partial)
- State of art + related work ✅ (partial)
- Methodology section ✅ (80%)

**What still needs writing:**
- Results section (depends on Phase 3)
- Discussion & limitations
- Conclusions
- Abstract (final)

**Deliverable:** Complete thesis (IEEE format, 12–14 pages, Spanish draft → English final)

---

### ⬜ Phase 6: Defense Preparation — PENDING

**What needs to happen:**
- Create presentation slides (~20 slides, 20 minutes)
- Prepare for likely questions (attention mechanisms, scale, alternatives to pruning, etc.)
- Practice defense narrative

**Deliverable:** Presentation + defense script

---

## Next Steps (Priority Order)

1. **Decide Phase 2.2 configuration** — How many models? Which pruning levels? Which strategy?
2. **Generate pruned models** — Create variants and sanity check them
3. **Run Phase 3 evaluations** — BBQ/EsBBQ + capabilities on all pruned variants
4. **Consolidate Phase 4 results** — Build master table and visualizations
5. **Write Phase 5 sections** — Fill in Results, Discussion, Conclusions as data becomes available
6. **Prepare Phase 6 materials** — Slides and Q&A prep (late June)

---

## Key Dates

| Milestone | Date |
|-----------|------|
| **Request defense** | By June 24 |
| **Submit thesis** | By July 1 |
| **Defense** | July 8–10 |

---

## Repository Status

```
✅ Complete:
  - results/bias-benchmarks-base/       (BBQ + EsBBQ baselines)
  - results/benchmarks/capabilities/    (MMLU, ARC, etc. baselines)
  - results/neuron_analysis/            (bias_scores, fairness_scores)
  - notebooks/05_bias_path_analysis.ipynb
  - notebooks/06_zero_bias_neurons.ipynb (PoC — superneuron zeroing)

⬜ Pending:
  - results/pruned-models/              (generated variants TBD)
  - results/pruned-models-eval/         (bias re-eval + capability re-eval)
  - figures/results/                    (Pareto frontier, heatmaps, etc.)
```

---

## Notes

- **No hard time estimates** — timelines depend on Phase 2.2 decisions and Colab stability
- **Flexible scope** — Phase 3/4 can scale down if needed (focus on primary model Llama-1B)
- **All code public** — github.com/peremartra/fairness-pruning
- **All datasets public** — HuggingFace (oopere/fairness-pruning-pairs-en/es)

---

**Stability analysis of top-K neuron rankings**. Given the moderate dataset size (n=55 EN / n=55 ES per language), the robustness of the identified bias-carrying neurons should be quantified before claiming category-level localization. We will perform a leave-one-template-out (LOTO) bootstrap on the top-K rankings derived from down_proj_input activations: for each of the five templates per category, recompute the per-neuron bias scores excluding that template's pairs and measure the rank stability of the top-50 neurons (Kendall's τ and top-K overlap against the full-dataset ranking). The same procedure will be applied per context (labour, institutional, healthcare, social, educational) to disentangle template-specific from context-specific artifacts. Particular attention will be paid to the Religion L15:166 super-neuron in Llama-3.2-1B: if it remains top-1 across all five LOTO folds and across context-leave-one-out folds, the finding is robust to dataset composition and the n=55 limitation cannot be invoked to challenge the localization claim. Results will be reported as a stability table in the Evaluation section, with non-robust rankings flagged explicitly rather than presented as definitive.

**Last Updated:** May 14, 2026  
**Status:** Phase 1 ✅ | Phase 2.1 ✅ | Phase 2.2 🔄
