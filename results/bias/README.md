# BBQ Baseline Evaluation Results

Bias Benchmark for QA (BBQ) evaluation results for unpruned base models.  
**Part of the fairness-pruning research project.**

**Evaluation Date:** December 7, 2025  
**Framework:** lm-evaluation-harness  
**Task:** BBQ (0-shot)

---

## Models Evaluated

| Model | Parameters | Status | Evaluation Time |
|-------|------------|--------|-----------------|
| BSC-LT/salamandra-2b | 2B | ✅ Complete | 1h 18m |
| meta-llama/Llama-3.2-1B | 1B | ✅ Complete | 53m |
| meta-llama/Llama-3.2-3B | 3B | ✅ Complete | 2h 1m |

---

## Key Findings

### 1. **Scaling Paradox in Bias**
Llama-3B shows **superior disambiguation capability** (73.5% vs 52.3% in 1B) but exhibits **2.4× higher bias in ambiguous contexts** (4.12% vs 1.70%). Larger models may encode stronger stereotypical associations while being more competent at reasoning when sufficient information is provided.

### 2. **Critical Bias Hotspots**
Three categories show concerning bias levels across all models:
- **Physical Appearance**: 9.9-17.9% (worsens dramatically with scale)
- **Age**: 2.7-16.3% (9pp increase from 1B to 3B)
- **Gender Identity**: 1.5-10.9% (7× amplification in 3B vs 1B)

### 3. **Cross-Architecture Comparison**
Salamandra-2b underperforms Llama-1B (-2.1pp overall accuracy) despite having 2× parameters, potentially due to language domain mismatch (BBQ is English-only; Salamandra optimized for ES/CA).

### 4. **Bidirectional Bias Patterns**
Several categories show negative bias scores (e.g., Race/ethnicity in Llama-1B: -1.7%), indicating bias **against** typical stereotypes rather than absence of bias. Both positive and negative values represent problematic model behavior.

---

## Overall Performance

| Model | Overall Acc | Amb Acc | Disamb Acc | Amb-Disamb Gap |
|-------|-------------|---------|------------|----------------|
| **Salamandra-2b** | 29.05% | 7.66% | 50.45% | -42.79pp |
| **Llama-3.2-1B** | 31.15% | 10.01% | 52.28% | -42.27pp |
| **Llama-3.2-3B** | 40.52% | 7.51% | 73.54% | **-66.03pp** |

**Interpretation:**
- **Amb Acc**: Model performance when context is ambiguous (tests reliance on stereotypes)
- **Disamb Acc**: Model performance when sufficient information is provided
- **Gap**: Larger gaps indicate stronger dependence on contextual information vs stereotypical assumptions

---

## Aggregate Bias Scores

| Model | Ambiguous Bias | Disambiguated Bias | Bias Ratio (Amb/Disamb) |
|-------|----------------|--------------------|-----------------------|
| **Salamandra-2b** | 2.07% | 2.77% | 0.75 |
| **Llama-3.2-1B** | 1.70% | 1.75% | 0.97 |
| **Llama-3.2-3B** | **4.12%** | 2.24% | **1.84** |

**Note:** Bias ratio > 1 indicates models rely more heavily on stereotypes when information is ambiguous.

---

## Bias Breakdown by Category

### Ambiguous Context Bias

| Category | Salamandra-2b | Llama-1B | Llama-3B | Scaling Trend |
|----------|---------------|----------|----------|---------------|
| **Physical_appearance** | 9.90% | 10.15% | **17.89%** | ⬆️ Worsens |
| **Age** | 2.72% | 7.50% | **16.30%** | ⬆️ Worsens |
| **Gender_identity** | 4.37% | 1.55% | **10.86%** | ⬆️ Worsens |
| **Religion** | 2.17% | 5.33% | 6.00% | ⬆️ Worsens |
| **SES** | 5.30% | 3.70% | 1.37% | ⬇️ Improves |
| **Nationality** | 5.00% | 1.43% | 4.74% | ↔️ Mixed |
| **Disability_status** | 4.37% | -0.51% | 4.63% | ↔️ Mixed |
| **Sexual_orientation** | 2.08% | 2.55% | 4.86% | ⬆️ Worsens |
| **Race_x_SES** | 1.54% | 0.61% | 1.52% | ↔️ Stable |
| **Race_ethnicity** | -0.52% | **-1.66%** | 1.34% | ↔️ Mixed |
| **Race_x_gender** | -0.38% | 0.89% | 1.40% | ↔️ Mixed |

### Disambiguated Context Bias

| Category | Salamandra-2b | Llama-1B | Llama-3B | Scaling Trend |
|----------|---------------|----------|----------|---------------|
| **Physical_appearance** | 13.37% | **14.84%** | 9.02% | ⬇️ Improves |
| **Gender_identity** | 6.84% | 1.97% | 8.04% | ↔️ Mixed |
| **Age** | 1.30% | 6.85% | 8.48% | ⬆️ Worsens |
| **SES** | **6.36%** | 4.52% | 0.17% | ⬇️ Improves |
| **Disability_status** | 4.11% | 3.08% | 0.51% | ⬇️ Improves |
| **Race_x_SES** | 3.77% | -1.52% | -0.69% | ⬇️ Improves |
| **Religion** | 2.00% | 3.74% | 3.84% | ⬆️ Worsens |
| **Nationality** | 1.56% | -1.84% | 3.52% | ↔️ Mixed |
| **Race_ethnicity** | 0.15% | 1.37% | 1.34% | ↔️ Stable |
| **Race_x_gender** | 0.38% | 0.18% | 0.50% | ↔️ Stable |
| **Sexual_orientation** | **-4.17%** | -0.71% | 0.93% | ↔️ Mixed |

**Important Note on Negative Values:**  
Negative bias scores indicate the model favors groups **opposite** to typical stereotypes. For example, a -4.17% score means the model systematically associates attributes with the counter-stereotypical group. This represents bias in the opposite direction, not absence of bias.

---

## Implications for Fairness-Pruning Research

### Priority Target Categories
Based on baseline measurements, the following categories should be primary targets for bias mitigation through selective pruning:

1. **Physical Appearance** (9.9-17.9% bias): Highest magnitude across models
2. **Age** (2.7-16.3% bias): Dramatic amplification with model scale
3. **Gender Identity** (1.5-10.9% bias): 7× increase from 1B to 3B

### Key Observations for Pruning Strategy

**Categories that worsen with scale** (candidates for aggressive pruning):
- Physical appearance (+8pp from 1B to 3B)
- Age (+9pp from 1B to 3B)
- Gender identity (+9pp from 1B to 3B)

**Categories with lower/stable bias** (monitor during pruning):
- Race × SES (consistently low)
- Nationality (< 5% in most cases)
- Some disability metrics (mixed results)

**Bidirectional bias** (requires nuanced approach):
- Race/ethnicity (switches from negative to positive with scale)
- Sexual orientation (varies significantly by model)

---

## Methodology

### BBQ Benchmark
The Bias Benchmark for QA evaluates model bias through question-answering tasks with two context types:
- **Ambiguous contexts**: Insufficient information to answer correctly (tests stereotype reliance)
- **Disambiguated contexts**: Sufficient information provided (tests reasoning capability)

**Bias score** = Percentage of times model selects stereotypical answer when both options are equally plausible.

### Evaluation Setup
- **Task**: BBQ (0-shot)
- **Hardware**: Google Colab (L4/A100 GPU)
- **Framework**: lm-evaluation-harness v0.4+
- **Date**: December 7, 2025
- **Scope**: English-language evaluation (ES/CA variants pending)

### Categories Evaluated
11 social bias dimensions: Age, Disability status, Gender identity, Nationality, Physical appearance, Race/ethnicity, Race × Gender, Race × SES, Religion, Socioeconomic status (SES), Sexual orientation.

---

## Next Steps

1. **Spanish/Catalan Evaluation**: Run EsBBQ and CatBBQ to assess Salamandra in native languages
2. **Pruned Model Comparison**: Evaluate fairness-pruned variants against these baselines
3. **Category-Specific Analysis**: Deep dive into Physical appearance, Age, and Gender bias mechanisms
4. **Cross-Lingual Bias Transfer**: Investigate whether bias patterns transfer across languages

---

## Files in This Directory

### Model Result Files
- `bsc_lt_salamandra_2b.json`: Complete BBQ metrics for Salamandra-2b
- `meta_llama_llama_3_2_1b.json`: Complete BBQ metrics for Llama-3.2-1B
- `meta_llama_llama_3_2_3b.json`: Complete BBQ metrics for Llama-3.2-3B

### Metrics Structure
Each JSON contains:
- **metadata**: Model name, evaluation timestamps, completion status
- **results.bbq**: 
  - Overall accuracy metrics (`acc,none`, `accuracy_amb,none`, `accuracy_disamb,none`)
  - Aggregate bias scores (`amb_bias_score,none`, `disamb_bias_score,none`)
  - Per-category bias scores for all 11 social dimensions (ambiguous + disambiguated)

**File size**: ~10KB per model (aggregated metrics only, no raw samples)

---

## References

1. **BBQ Dataset**: Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question Answering"
2. **Evaluation Framework**: EleutherAI lm-evaluation-harness
3. **OptiPFair Library**: [github.com/peremartra/OptiPFair](https://github.com/peremartra/OptiPFair)
4. **Research Repository**: [github.com/peremartra/fairness-pruning](https://github.com/peremartra/fairness-pruning)

---

**Author**: Pere Martra  
**Project**: Fairness-Pruning Research (Rearchitecting LLMs - Manning Publications)  
**License**: Results available for research purposes
