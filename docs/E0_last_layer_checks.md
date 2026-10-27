# E0 — Last-layer checks (stored BiasScores, no GPU) — 2026-10-27

Status: DONE. Repeat when new models are added (scripts: E0_last_layer_check.py, E0_last_layer_overlap.py).

Data: results/neuron_analysis/<model>/<en|es>/<Category>_bias_scores.json, projection down_proj_input. 3 models x EN/ES x 5 categories = 30 cases.
Scale proxy: per-layer median of the score (we only have |a1-a2|, not baseline activation scale). This is an assumption.

Usage (read-only over stored scores):
- `python3 -I docs/E0_last_layer_check.py results/neuron_analysis <out_csv>`
- `python3 -I docs/E0_last_layer_overlap.py results/neuron_analysis <out_csv>`

## 1. Scale check
- Paper criterion (layer with max mean score) reproduces: 30/30 last layer (raw).
- Also 30/30 with scale-free criteria (mean score/layer median; p99/median).
- Last-layer median score = 1.5-3.8x the median of other layers.
- Tail p99/median: last layer 6-10 (Llama), 12-23 (Salamandra) vs 3-5 elsewhere.
- Share of last layer in global top-1%, raw -> median-normalised (uniform 4-6%):
  - Llama-1B 49-71% -> 18-24%
  - Llama-3B 34-55% -> 12-17%
  - Salamandra 47-61% -> 24-31%
  - Overall 34-71% -> 12-31%
- Jaccard raw vs normalised top-1% selection: 0.31-0.58.
- Part of the reported concentration is layer scale; the last layer remains the most over-represented layer after normalising.

## 2. Cross-category overlap (within layer)
- Last layer, top-40 per category: mean pairwise Jaccard 0.19-0.31 (random 0.002-0.004); 2-11 of 40 neurons in all 5 categories.
- Full-layer Spearman across categories: 0.47-0.66 last layer, 0.45-0.57 mid layer, 0.10-0.24 first layer (Salamandra first layer higher in top-N).
- Shared component grows with depth; not exclusive to the last layer.

## 3. Position (final)
- This does not invalidate the result. A neuron with a larger activation difference, whether due to scale or any other cause, has more potential influence on the output. Raw magnitude is therefore a reasonable criterion for deciding what to intervene on.
- What we cannot claim is that these neurons are specific to the demographic attribute. This is the statement that goes into the paper (limitations); it may also mean token prediction rather than stereotype processing (Geva et al. 2022, to verify).
- Not shown by E0: generic high-variance neurons vs demographically selective ones. That needs random last-layer, magnitude-matched controls and non-demographic control pairs (second iteration).

## 4. What goes into the paper
- Limitation sentence: no claim of demographic specificity.
- The rest of the E0 prose may be adapted for the paper, always keeping our position stated (raw magnitude as influence criterion).
- Exact percentages to be recomputed with the new models before being cited.
