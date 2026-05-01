# Fairness Pruning: Bias Mitigation through Activation-Guided MLP Width Pruning in Large Language Models

**Author:** Pere Martra  
**Program:** Master in Research in Artificial Intelligence (UIMP)  
**Format:** IEEE double column, 12-14 pages  
**Target:** UIMP defense (July 2026) · TMLR submission (post-defense)

---

## Table of Contents

- [ ] **Abstract** (~half column)

- [ ] **1. Introduction** (~1 page)
  - [ ] 1.1 Motivation: bias in LLMs and limitations of existing mitigation methods
  - [ ] 1.2 Gap: no prior work on bias localization via activation-guided MLP pruning
  - [ ] 1.3 Contributions
  - [ ] 1.4 Document structure

- [ ] **2. Related Work** (~1.5 pages)
  - [ ] 2.1 Bias in LLMs — types, measurement (BBQ, EsBBQ), real-world harms
  - [ ] 2.2 Bias mitigation — pre/in/post-processing approaches; gap in structural pruning
  - [ ] 2.3 Neural network pruning — structured vs. unstructured; magnitude-based vs. activation-based
  - [ ] 2.4 Mechanistic interpretability — feature circuits, activation analysis, SAEs

- [ ] **3. Methodology** (~2 pages)
  - [ ] 3.1 Pipeline overview (flow figure)
  - [ ] 3.2 Prompt pair datasets — construction, token constraint, HuggingFace
  - [ ] 3.3 Neuron bias detection — OptiPFair, FairnessPruningScore formula
  - [ ] 3.4 Fairness pruning — selective layer pruning with `layer_indices`
  - [ ] 3.5 Evaluation protocol — capabilities benchmarks, bias benchmarks, metrics

- [ ] **4. Experimental Setup** (~0.5 pages)
  - [ ] Models: Llama-3.2-1B (primary), Llama-3.2-3B and Salamandra-2B (validation)
  - [ ] Hardware, dtypes, reproducibility

- [ ] **5. Results** (~4-5 pages)
  - [ ] 5.1 Baseline capabilities and bias (BBQ + EsBBQ)
  - [ ] 5.2 Bias localization analysis — heatmaps, Jaccard, cross-lingual (RQ1 + RQ2)
  - [ ] 5.3 Pruning results — bias reduction vs. capability retention (RQ3)
  - [ ] 5.4 Cross-model validation — Llama-3B and Salamandra

- [ ] **6. Discussion** (~1 page)
  - [ ] 6.1 Two encoding strategies: distributed vs. sparse bias
  - [ ] 6.2 Limitations — Western-centric scope, models <4B, MLP-only pruning

- [ ] **7. Conclusions** (~0.5 pages)
  - [ ] 7.1 Answers to RQ1, RQ2, RQ3
  - [ ] 7.2 Implications and future work

- [ ] **References** (~1-1.5 pages)
