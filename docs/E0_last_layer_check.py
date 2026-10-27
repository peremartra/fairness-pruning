"""
Last-layer artifact check (read-only analysis over stored BiasScores).

Question: is the concentration of bias signal in the final layer of
down_proj_input a property of specific neurons, or a layer-scale effect
(raw, unnormalized |a1-a2| magnitudes are larger in the last layer)?

We only have the differential scores, not baseline activation scale, so the
per-layer median score is used as a scale proxy (most neurons are not
demographically differential, so their score tracks activation scale).
This is an assumption and is reported as such.

Usage: python3 -I last_layer_check.py <path to results/neuron_analysis> <out_csv>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODELS = ["llama-3.2-1B", "llama-3.2-3B", "salamandra-2B"]
LANGS = ["en", "es"]
CATS = ["Age", "Gender", "PhysicalAppearance", "RaceEthnicity", "Religion"]
PROJ = "down_proj_input_layer_"
TOP_PCT = 0.01
KS = (20, 40)


def load(model, lang, cat):
    raw = json.load(open(ROOT / model / lang / f"{cat}_bias_scores.json"))
    layers = sorted(int(k.split("_layer_")[-1]) for k in raw if k.startswith(PROJ))
    return np.stack([np.asarray(raw[f"{PROJ}{i}"]["values"], dtype=np.float64) for i in layers])  # [L, N]


def topk_mask(score, k=None, pct=None):
    flat = score.ravel()
    if pct is not None:
        k = int(round(flat.size * pct))
    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros(flat.size, dtype=bool)
    m[idx] = True
    return m.reshape(score.shape)


def share_last(mask):
    return mask[-1].sum() / mask.sum()


rows = []
for model in MODELS:
    for lang in LANGS:
        for cat in CATS:
            S = load(model, lang, cat)  # [L, N]
            L, N = S.shape
            med = np.median(S, axis=1)
            mad = np.median(np.abs(S - med[:, None]), axis=1) * 1.4826
            mad = np.where(mad > 0, mad, np.nan)
            p99 = np.quantile(S, 0.99, axis=1)

            S_med = S / med[:, None]                # scale-free: score / layer median
            S_rz = (S - med[:, None]) / mad[:, None]  # robust z within layer

            # paper's criterion: layer with max mean (min-max is monotone -> same argmax)
            argmax_raw = int(np.argmax(S.mean(axis=1)))
            argmax_med = int(np.argmax(S_med.mean(axis=1)))
            argmax_p99r = int(np.argmax(p99 / med))  # tail heaviness, scale-free

            r = dict(
                model=model, lang=lang, cat=cat, L=L,
                last_median_over_others=med[-1] / np.median(med[:-1]),
                last_p99_over_median=p99[-1] / med[-1],
                others_p99_over_median=float(np.median(p99[:-1] / med[:-1])),
                argmax_mean_raw=argmax_raw,
                argmax_mean_medNorm=argmax_med,
                argmax_tail_ratio=argmax_p99r,
            )
            sels = {"raw": S, "medNorm": S_med, "robustZ": S_rz}
            masks1 = {}
            for name, sc in sels.items():
                m = topk_mask(sc, pct=TOP_PCT)
                masks1[name] = m
                r[f"top1pct_share_last_{name}"] = share_last(m)
                r[f"top1pct_nlayers_{name}"] = int((m.sum(axis=1) > 0).sum())
                for k in KS:
                    mk = topk_mask(sc, k=k)
                    r[f"top{k}_share_last_{name}"] = share_last(mk)
            r["frac_last_layer_selected_raw"] = masks1["raw"][-1].mean()
            inter = (masks1["raw"] & masks1["medNorm"]).sum()
            union = (masks1["raw"] | masks1["medNorm"]).sum()
            r["jaccard_raw_vs_medNorm_top1pct"] = inter / union
            rows.append(r)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

# ---- console summary ----
def last_is(r, key):
    return r[key] == r["L"] - 1

n = len(rows)
print(f"combinations: {n}")
print("last layer == argmax per-layer mean, RAW (paper's criterion):      ",
      sum(last_is(r, "argmax_mean_raw") for r in rows), "/", n)
print("last layer == argmax per-layer mean of score/median (scale-free):  ",
      sum(last_is(r, "argmax_mean_medNorm") for r in rows), "/", n)
print("last layer == argmax of tail ratio p99/median (scale-free):        ",
      sum(last_is(r, "argmax_tail_ratio") for r in rows), "/", n)
print()
hdr = ("model", "lang", "cat", "L", "medLast/med", "p99/med last", "p99/med oth",
       "t1% raw", "t1% med", "t1% rz", "t40 raw", "t40 med", "t40 rz", "Jacc")
print(("{:<13}{:<4}{:<19}{:<3}" + "{:>12}" * 10).format(*hdr))
for r in rows:
    print(("{:<13}{:<4}{:<19}{:<3}" + "{:>12.2f}" * 10).format(
        r["model"], r["lang"], r["cat"], r["L"],
        r["last_median_over_others"], r["last_p99_over_median"], r["others_p99_over_median"],
        r["top1pct_share_last_raw"], r["top1pct_share_last_medNorm"], r["top1pct_share_last_robustZ"],
        r["top40_share_last_raw"], r["top40_share_last_medNorm"], r["top40_share_last_robustZ"],
        r["jaccard_raw_vs_medNorm_top1pct"]))
print()
print("uniform expectation of last-layer share = 1/L (1B: %.3f, 3B: %.3f, Salamandra: %.3f)" % (1/16, 1/28, 1/24))
