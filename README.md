# Fairness Pruning: Bias Mitigation through Activation-Guided MLP Width Pruning in Large Language Models

**Official PyTorch implementation.**

This repository contains the code and experiments for **Fairness Pruning**, a novel method to mitigate bias in Large Language Models (specifically **Llama-3.2** and **Salamandra**) by identifying and pruning neurons with high bias contribution but low structural importance.

Our approach utilizes **activation-guided MLP width pruning** to selectively remove biased components while preserving model performance.

## Key Features
- **Activation-Guided Detection:** Uses [OptiPFair](https://github.com/peremartra/optipfair) to identify biased neurons.
- **Selective Pruning:** Targets specific MLP layers to maximize fairness/performance trade-offs.
- **Multi-Model Support:** Validated on Llama-3.2 (1B/3B) and Salamandra-2B.
