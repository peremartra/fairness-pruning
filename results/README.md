# Fairness Pruning: Bias Mitigation through Activation-Guided MLP Width Pruning in Large Language Models

This repository contains the evaluation results and benchmarks for the Master's Thesis (TFM) regarding **Fairness Pruning**. The project aims to eliminate or zero out neurons exhibiting high bias but low structural importance using activation-guided pruning.

## Evaluated Models

The following models have been evaluated to establish a baseline performance before pruning. The primary model for this study is **Llama-3.2-1B**, while **Llama-3.2-3B** and **Salamandra-2B** serve as reference points for performance and cross-lingual capabilities.

* **Primary Model:** `meta-llama/Llama-3.2-1B`
* **Reference (Larger):** `meta-llama/Llama-3.2-3B`
* **Reference (Multi-lingual):** `BSC-LT/salamandra-2b`

## Baseline Evaluation Results

The models were evaluated using `lm_eval` across three main categories: English Core Capabilities, English Reasoning & Knowledge, and Spanish/Cross-Lingual Capabilities.

### 1. English Core Capabilities
Basic language modeling and instruction following performance.

| Metric | Task | Llama-3.2-1B | Llama-3.2-3B | Salamandra-2B |
| :--- | :--- | :---: | :---: | :---: |
| **WikiText** | Word Perplexity (↓) | 11.99 | 9.54 | 11.89 |
| **Lambada** | Perplexity (↓) | 5.43 | 3.88 | 7.27 |
| **IFEval** | Instruction Strict Acc (↑)| 0.1475 | 0.1199 | 0.1691 |

> **Note:** Lower is better for Perplexity. Higher is better for Accuracy.

### 2. English Reasoning & Knowledge
Standard benchmarks for reasoning, general knowledge, and truthfulness.

| Task | Metric | Llama-3.2-1B | Llama-3.2-3B | Salamandra-2B |
| :--- | :--- | :---: | :---: | :---: |
| **GSM8K** | Exact Match (Strict) | 5.53% | 26.16% | 0.00% |
| **MMLU** | Accuracy | 31.98% | 57.83% | 25.12% |
| **ARC-Challenge**| Accuracy (Norm) | 37.20% | 46.16% | 37.37% |
| **HellaSwag** | Accuracy (Norm) | 64.19% | 74.11% | 62.81% |
| **TruthfulQA** | Accuracy (MC2) | 38.54% | 39.18% | 35.93% |

### 3. Spanish / Cross-Lingual Capabilities
Benchmarks specifically selected to test performance in Spanish and cross-lingual understanding.

| Task | Metric | Llama-3.2-1B | Llama-3.2-3B | Salamandra-2B |
| :--- | :--- | :---: | :---: | :---: |
| **Global MMLU (ES)**| Accuracy | 35.09% | 54.95% | 27.52% |
| **ARC (ES)** | Accuracy (Norm) | 30.00% | 39.40% | 31.88% |
| **HellaSwag (ES)** | Accuracy (Norm) | 47.31% | 58.89% | 52.18% |
| **Belebele** | Accuracy | 32.33% | 56.56% | 25.44% |

## Methodology

* **Bias Detection:** The `optipfair` library is used for identifying neurons with high bias contributions.
* **Evaluation Framework:** All quantitative metrics presented above were generated using the `lm_eval` harness.
