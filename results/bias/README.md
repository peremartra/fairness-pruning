# Bias Evaluation Results

Baseline bias measurements for unpruned models using BBQ (English) and EsBBQ (Spanish) benchmarks.

**Evaluation Framework:** lm-evaluation-harness  
**Task Configuration:** 0-shot  
**Evaluation Dates:** BBQ (Dec 7, 2025), EsBBQ (Dec 21-22, 2025)

## Models Evaluated

| Model | Parameters | BBQ Status | EsBBQ Status |
|-------|-----------|------------|--------------|
| BSC-LT/salamandra-2b | 2B | ✅ Complete | ✅ Complete |
| meta-llama/Llama-3.2-1B | 1B | ✅ Complete | ✅ Complete |
| meta-llama/Llama-3.2-3B | 3B | ✅ Complete | ✅ Complete |

---

## BBQ Baseline Results (English)

### Overall Performance

| Model | Overall Acc | Amb Acc | Disamb Acc | Amb-Disamb Gap |
|-------|-------------|---------|------------|----------------|
| Salamandra-2b | 29.05% | 7.66% | 50.45% | -42.79pp |
| Llama-3.2-1B | 31.15% | 10.01% | 52.28% | -42.27pp |
| Llama-3.2-3B | 40.52% | 7.51% | 73.54% | -66.03pp |

### Aggregate Bias Scores

| Model | Ambiguous Bias | Disambiguated Bias | Ratio (Amb/Disamb) |
|-------|----------------|--------------------|--------------------|
| Salamandra-2b | 2.07% | 2.77% | 0.75 |
| Llama-3.2-1B | 1.70% | 1.75% | 0.97 |
| Llama-3.2-3B | 4.12% | 2.24% | 1.84 |

### Top Bias Categories (Ambiguous Context)

| Category | Salamandra-2b | Llama-1B | Llama-3B |
|----------|---------------|----------|----------|
| Physical Appearance | 9.90% | 10.15% | 17.89% |
| Age | 2.72% | 7.50% | 16.30% |
| Gender Identity | 4.37% | 1.55% | 10.86% |

---

## EsBBQ Baseline Results (Spanish)

### Overall Performance

| Model | Overall Acc | Amb Acc | Disamb Acc | Amb-Disamb Gap |
|-------|-------------|---------|------------|----------------|
| Salamandra-2b | 41.69% | 25.10% | 49.55% | -24.45pp |
| Llama-3.2-1B | 44.98% | 38.19% | 48.19% | -10.00pp |
| Llama-3.2-3B | 54.50% | 18.59% | 71.61% | -53.02pp |

### Aggregate Bias Scores

| Model | Ambiguous Bias | Disambiguated Bias | Ratio (Amb/Disamb) |
|-------|----------------|--------------------|--------------------|
| Salamandra-2b | 1.69% | 1.76% | 0.96 |
| Llama-3.2-1B | 1.03% | 3.31% | 0.31 |
| Llama-3.2-3B | 3.96% | 0.76% | 5.21 |

### Top Bias Categories (Ambiguous Context)

| Category | Salamandra-2b | Llama-1B | Llama-3B |
|----------|---------------|----------|----------|
| Physical Appearance | 1.79% | -0.51% | 9.69% |
| Spanish Region | 4.63% | -0.93% | 8.02% |
| Gender | -0.47% | 0.86% | 8.64% |

**Note:** Spanish Region is a category unique to EsBBQ.

---

## Cross-Lingual Observations

### Performance Shifts

All models show higher overall accuracy on EsBBQ (Spanish) compared to BBQ (English):
- **Salamandra-2b:** +12.64pp (29.05% → 41.69%)
- **Llama-3.2-1B:** +13.83pp (31.15% → 44.98%)
- **Llama-3.2-3B:** +13.98pp (40.52% → 54.50%)

### Ambiguous Context Accuracy

All models perform substantially better in ambiguous Spanish contexts than English:
- **Salamandra-2b:** +17.44pp (7.66% → 25.10%)
- **Llama-3.2-1B:** +28.18pp (10.01% → 38.19%)
- **Llama-3.2-3B:** +11.08pp (7.51% → 18.59%)

### Bias Pattern Consistency

**Llama-3.2-3B** maintains similar bias patterns across languages:
- BBQ: Ambiguous bias (4.12%) > Disambiguated bias (2.24%)
- EsBBQ: Ambiguous bias (3.96%) > Disambiguated bias (0.76%)
- Ratio shift: 1.84 → 5.21

**Llama-3.2-1B** shows inverted pattern in Spanish:
- BBQ: Ambiguous bias (1.70%) ≈ Disambiguated bias (1.75%)
- EsBBQ: Ambiguous bias (1.03%) < Disambiguated bias (3.31%)
- Ratio shift: 0.97 → 0.31

**Salamandra-2b** maintains balanced pattern in both languages:
- BBQ: Ambiguous bias (2.07%) < Disambiguated bias (2.77%)
- EsBBQ: Ambiguous bias (1.69%) < Disambiguated bias (1.76%)
- Ratio: 0.75 → 0.96

### Category-Level Consistency

**Physical Appearance** ranks among top 3 bias categories in both languages for Llama-3.2-3B:
- BBQ: Ambiguous 17.89%, Disambiguated 9.02%
- EsBBQ: Ambiguous 9.69%, Disambiguated 5.44%

**Age** shows high bias in BBQ but not in EsBBQ:
- BBQ: Ambiguous 16.30%, Disambiguated 8.48%
- EsBBQ: Ambiguous 2.71%, Disambiguated 4.47%

**Gender/Gender Identity** increases with model scale in both languages:
- BBQ: Llama-1B (Amb 1.55%, Disamb 1.97%) → Llama-3B (Amb 10.86%, Disamb 8.04%)
- EsBBQ: Llama-1B (Amb 0.86%, Disamb 2.04%) → Llama-3B (Amb 8.64%, Disamb 4.57%)

**SES (Socioeconomic Status)** shows opposite scaling trends:
- BBQ: Llama-1B (Amb 3.70%, Disamb 4.52%) → Llama-3B (Amb 1.37%, Disamb 0.17%)
- EsBBQ: Llama-1B (Amb 2.17%, Disamb 8.69%) → Llama-3B (Amb 6.30%, Disamb 1.34%)

---

## Data Files

**BBQ (English):**
- `bsc_lt_salamandra_2b.json` - Salamandra-2B results
- `meta_llama_llama_3.2_1b.json` - Llama-3.2-1B results
- `meta_llama_llama_3.2_3b.json` - Llama-3.2-3B results

**EsBBQ (Spanish):**
- `esbbq_final_results_salamandra2b.json` - Salamandra-2B results
- `esbbq_final_results_llama-3.2-1B.json` - Llama-3.2-1B results
- `esbbq_final_results_llama-3.2-3B.json` - Llama-3.2-3B results
