"""
E3 follow-up: how much does a down_proj neuron weigh in the output? (Llama-3.1-8B, last layer, L31)

Neuron i contributes act_i * down_proj.weight[:, i] to the residual stream. The BiasScore is measured on act_i
(the input of down_proj), so it does not include the norm of that column. This script reads ONLY down_proj of one
layer (no forward pass, no data, CPU is enough) and reports, for a list of neurons, the percentile within the layer of:
  - the L2 norm of the down_proj column
  - the BiasScore of each category (from <category>_bias_scores.json, key down_proj_input_layer_<L>)
  - the product BiasScore x column norm (proxy of the effect in the output space, NOT the effect itself)

Usage (Colab, CPU; the model is gated: log in to Hugging Face first):
  !python fairness-pruning/analysis/e3/l31_weight_check.py \
      --scores-dir /content/drive/MyDrive/fairness-pruning/results/llama-3.1-8b/fairness-pruning-pairs-en \
      --out-dir /content/drive/MyDrive/fair_pruning/results/e3/llama-3.1-8b_en/analysis
Options: --model-id, --layer 31, --neurons 6950 1495 ..., --weights-file (a local .safetensors shard; skips the download)
Output: <out-dir>/<timestamp>_l31_weights/ (neurons.csv, layer_summary.json, report.md)
Note: the download is the whole safetensors shard that holds the tensor (about 5 GB); only the one tensor is read.
Percentile = % of the layer's neurons with a strictly lower value (100 = highest). Rank 1 = highest.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = {"Religion": "Religion", "Race": "RaceEthnicity", "Gender": "Gender"}   # label -> scores file prefix
# 6950 = shared by real_race5 and real_religion5; the next four are abl_race4, the last four abl_religion4 (E3 ablation)
DEFAULT_NEURONS = {6950: "abl_shared1", 1495: "abl_race4", 2100: "abl_race4", 6466: "abl_race4", 6532: "abl_race4",
                   33: "abl_religion4", 311: "abl_religion4", 2398: "abl_religion4", 2564: "abl_religion4"}


def load_down_proj(model_id, layer, weights_file):
    from safetensors import safe_open
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    if weights_file:
        path = weights_file
    else:
        from huggingface_hub import hf_hub_download
        idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))
        path = hf_hub_download(model_id, idx["weight_map"][name])
    with safe_open(path, framework="pt") as f:
        w = f.get_tensor(name).float().numpy()            # (hidden, intermediate)
    return w


def pct(values, x):
    return 100.0 * float((values < x).sum()) / len(values)


def rank(values, x):
    return int((values > x).sum()) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--neurons", type=int, nargs="*", default=None, help="default: the nine neurons of the E3 ablation")
    ap.add_argument("--weights-file")
    a = ap.parse_args()

    w = load_down_proj(a.model_id, a.layer, a.weights_file)
    hidden, width = w.shape
    assert np.isfinite(w).all(), "non-finite weights"
    col_norm = np.linalg.norm(w, axis=0)                  # (width,)
    assert col_norm.shape == (width,)

    scores = {}
    for label, prefix in CATEGORIES.items():
        d = json.load(open(Path(a.scores_dir) / f"{prefix}_bias_scores.json"))
        v = np.asarray(d[f"down_proj_input_layer_{a.layer}"]["values"], dtype=np.float64)
        assert v.shape == (width,) and np.isfinite(v).all(), f"{prefix}: unexpected scores"
        scores[label] = v

    neurons = {n: DEFAULT_NEURONS.get(n, "") for n in (a.neurons if a.neurons else DEFAULT_NEURONS)}
    rows = []
    for n, tag in neurons.items():
        assert 0 <= n < width, n
        r = {"neuron": n, "set": tag, "col_norm": float(col_norm[n]), "col_norm_pct": pct(col_norm, col_norm[n]),
             "col_norm_rank": rank(col_norm, col_norm[n])}
        for label, v in scores.items():
            prod = v * col_norm
            r[f"{label}_score"] = float(v[n]); r[f"{label}_score_pct"] = pct(v, v[n]); r[f"{label}_score_rank"] = rank(v, v[n])
            r[f"{label}_prod"] = float(prod[n]); r[f"{label}_prod_pct"] = pct(prod, prod[n]); r[f"{label}_prod_rank"] = rank(prod, prod[n])
        rows.append(r)
    df = pd.DataFrame(rows)

    summary = {"model": a.model_id, "layer": a.layer, "width": width, "hidden": hidden,
               "col_norm": {"min": float(col_norm.min()), "median": float(np.median(col_norm)), "mean": float(col_norm.mean()),
                            "p99": float(np.percentile(col_norm, 99)), "max": float(col_norm.max())},
               "top5_by_product": {}}
    for label, v in scores.items():
        prod = v * col_norm
        top = np.argsort(-prod)[:5]
        summary["top5_by_product"][label] = [{"neuron": int(i), "score": float(v[i]), "col_norm": float(col_norm[i]), "prod": float(prod[i])} for i in top]

    out = Path(a.out_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_l31_weights"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "neurons.csv", index=False)
    json.dump(summary, open(out / "layer_summary.json", "w"), indent=1)

    cols = ["neuron", "set", "col_norm", "col_norm_pct"] + [f"{l}_{m}_pct" for l in scores for m in ("score", "prod")]
    R = [f"# down_proj column weight check — {a.model_id}, layer {a.layer}", "",
         f"Layer: {width} neurons, hidden size {hidden}. Column norm: min {summary['col_norm']['min']:.3f}, median {summary['col_norm']['median']:.3f}, "
         f"p99 {summary['col_norm']['p99']:.3f}, max {summary['col_norm']['max']:.3f}.", "",
         "Percentile = % of the layer's neurons with a strictly lower value. prod = BiasScore x column norm (proxy, not the effect).", "",
         df[cols].round(3).to_markdown(index=False), "", "## Top 5 of the layer by BiasScore x column norm", ""]
    for label, t in summary["top5_by_product"].items():
        R += [f"**{label}**", "", pd.DataFrame(t).round(4).to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"report: {out / 'report.md'}")
    print(df[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
