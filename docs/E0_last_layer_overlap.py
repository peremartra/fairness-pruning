"""
Cross-category overlap of the most differential neurons, last layer vs other layers.

If the top last-layer neurons are the same across categories, they are likely
generic high-variance neurons (respond to any prompt change), not
category-specific. Read-only over stored BiasScores (down_proj_input).

Usage: python3 -I last_layer_overlap.py <results/neuron_analysis> <out_csv>
"""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = ["llama-3.2-1B", "llama-3.2-3B", "salamandra-2B"]
LANGS = ["en", "es"]
CATS = ["Age", "Gender", "PhysicalAppearance", "RaceEthnicity", "Religion"]
PROJ = "down_proj_input_layer_"
NS = (40, 80)  # top-N neurons per category within the layer


def load(model, lang):
    out = {}
    for c in CATS:
        raw = json.load(open(ROOT / model / lang / f"{c}_bias_scores.json"))
        L = len({k for k in raw if k.startswith(PROJ)})
        out[c] = np.stack([np.asarray(raw[f"{PROJ}{i}"]["values"], dtype=np.float64) for i in range(L)])
    return out  # {cat: [L, N]}


def ranks(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def layer_stats(vecs, n_top):
    """vecs: {cat: [N]} for one layer."""
    N = len(next(iter(vecs.values())))
    tops = {c: set(np.argsort(v)[-n_top:].tolist()) for c, v in vecs.items()}
    jac = [len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for a, b in combinations(CATS, 2)]
    # expected Jaccard of two random n-subsets of N (hypergeometric mean overlap n^2/N)
    e_int = n_top * n_top / N
    exp_j = e_int / (2 * n_top - e_int)
    counts = {}
    for s in tops.values():
        for i in s:
            counts[i] = counts.get(i, 0) + 1
    in3 = sum(1 for v in counts.values() if v >= 3)
    in5 = sum(1 for v in counts.values() if v == 5)
    rho = np.mean([np.corrcoef(ranks(vecs[a]), ranks(vecs[b]))[0, 1] for a, b in combinations(CATS, 2)])
    return float(np.mean(jac)), exp_j, in3, in5, float(rho)


rows = []
for model in MODELS:
    for lang in LANGS:
        D = load(model, lang)
        L = D[CATS[0]].shape[0]
        mid = L // 2
        for n_top in NS:
            for tag, li in (("last", L - 1), ("mid", mid), ("first", 0)):
                vecs = {c: D[c][li] for c in CATS}
                j, ej, in3, in5, rho = layer_stats(vecs, n_top)
                rows.append({"model": model, "lang": lang, "topN": n_top, "layer": f"{tag}(L{li})",
                             "mean_pairwise_jaccard": j, "expected_random_jaccard": ej,
                             "n_in_>=3_cats": in3, "n_in_5_cats": in5,
                             "mean_spearman_full_layer": rho})

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

print("{:<13}{:<4}{:<5}{:<11}{:>9}{:>9}{:>8}{:>8}{:>9}".format(
    "model", "lang", "topN", "layer", "Jacc", "Jrand", ">=3cat", "5cat", "Spearman"))
for r in rows:
    print("{:<13}{:<4}{:<5}{:<11}{:>9.3f}{:>9.4f}{:>8}{:>8}{:>9.3f}".format(
        r["model"], r["lang"], r["topN"], r["layer"], r["mean_pairwise_jaccard"],
        r["expected_random_jaccard"], r["n_in_>=3_cats"], r["n_in_5_cats"], r["mean_spearman_full_layer"]))
