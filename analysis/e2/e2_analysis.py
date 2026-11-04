#!/usr/bin/env python3
"""
E2 analysis (Llama-3.2-3B): random vs real neuron sets.

Reads the small per-set summaries written by notebook 09 (`*_summary.json`, a few KB each) plus the baseline
BBQ table (`baseline_bbq_by_category.json`, made once by `make_baseline_table.py`) and writes tables,
a JSON with every number computed, and a figure. It runs on a laptop: no GPU, no per-item logs.

The script describes, it does not judge: no thresholds, no verdicts. Which numbers go to R1_exp.md or to
the paper is decided afterwards.

Usage
    python analysis/e2/e2_analysis.py --input-dir <folder with the *_summary.json files>
Options
    --out-dir DIR       default: <input-dir>/../analysis/<timestamp>
    --baseline-bbq F    default: <input-dir>/baseline_bbq_by_category.json
    --csv F             consolidated CSV of notebook 09 for a cross-check (default: newest
                        e2_bbq_by_category_*.csv in <input-dir>/..; the check is skipped if there is none)
    --l4-seeds 0 1      seeds whose BBQ ran on an L4 (R1_exp.md, E2/S5); used for the sensitivity table
    --no-figures        do not draw the figure
Requires: python >= 3.9, numpy, pandas; matplotlib only for the figure.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance", "Race_ethnicity",
        "Race_x_SES", "Race_x_gender", "Religion", "SES", "Sexual_orientation"]
ITEMS_PER_CONTEXT = {"Age": 1840, "Disability_status": 778, "Gender_identity": 2836, "Nationality": 1540,
                     "Physical_appearance": 788, "Race_ethnicity": 3440, "Race_x_SES": 5580,
                     "Race_x_gender": 7980, "Religion": 600, "SES": 3432, "Sexual_orientation": 432}
TARGET, CROSS = "Religion", "Race_ethnicity"
BASELINE, REAL_T, REAL_C = "baseline", "real_religion20", "real_race20"
N_RANDOM = 10
BBQ_METRICS = ["amb_acc", "disamb_acc", "disamb_unknown", "disamb_bias", "amb_bias", "flips_pct"]
EPS = 1e-9

# GPU that ran the BBQ of each set. NOT stored in the data: taken from R1_exp.md (E2, S1/S3/S5).
# (The `env.gpu` field of a summary describes the Colab session that wrote the summary, not the one that evaluated.)
def bbq_gpu_note(set_id, l4_seeds):
    if set_id == BASELINE:
        return "A100-SXM4-40GB (baseline rerun, S1)"
    if set_id.startswith("real_"):
        return "A100-SXM4-80GB (S3)"
    m = re.search(r"seed(\d+)", set_id)
    if m:
        return "L4" if int(m.group(1)) in l4_seeds else "A100-SXM4-40GB"
    return "unknown"

# Measures: (name, value column, delta-against-baseline column)
MEASURES = [("disamb_bias", "disamb_bias", "d_disamb_bias"),
            ("abs_disamb_bias", "abs_disamb_bias", "d_abs_disamb_bias"),
            ("amb_bias", "amb_bias", "d_amb_bias"),
            ("abs_amb_bias", "abs_amb_bias", "d_abs_amb_bias"),
            ("disamb_acc", "disamb_acc", "d_disamb_acc"),
            ("amb_acc", "amb_acc", "d_amb_acc"),
            ("disamb_unknown", "disamb_unknown", "d_disamb_unknown"),
            ("flips_pct", "flips_pct", "d_flips_pct")]

CAP_FIELDS = {  # column -> (task, key)
    "wikitext_word_ppl": ("wikitext", "word_perplexity,none"),
    "wikitext_byte_ppl": ("wikitext", "byte_perplexity,none"),
    "wikitext_bits_per_byte": ("wikitext", "bits_per_byte,none"),
    "mmlu_acc": ("mmlu", "accuracy"), "mmlu_stem": ("mmlu", "category_STEM"),
    "mmlu_humanities": ("mmlu", "category_Humanities"),
    "mmlu_social_sciences": ("mmlu", "category_Social_Sciences"), "mmlu_other": ("mmlu", "category_Other"),
    "arc_challenge_acc": ("arc_challenge", "accuracy"), "arc_challenge_acc_norm": ("arc_challenge", "acc_norm"),
    "hellaswag_acc": ("hellaswag", "accuracy"), "hellaswag_acc_norm": ("hellaswag", "acc_norm"),
    "hellaswag_es_acc": ("hellaswag_es", "accuracy"), "hellaswag_es_acc_norm": ("hellaswag_es", "acc_norm"),
}


# ----------------------------------------------------------------------------- helpers
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(x):
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, min=np.nan, max=np.nan)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                median=float(np.median(x)), min=x.min(), max=x.max())


def fmt(v, nd=2, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
        return s
    return str(v)


def md_table(df, nd=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v, nd) for v in r.tolist()) + " |")
    return "\n".join(lines)


def records(df):
    return json.loads(df.to_json(orient="records"))


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


# ----------------------------------------------------------------------------- loading
def load_sets(input_dir, baseline_bbq_path):
    sets = {}
    for p in sorted(Path(input_dir).glob("*_summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_path"] = p
        sets[d["set"]] = d
    if BASELINE not in sets:
        sys.exit(f"ERROR: baseline_summary.json not found in {input_dir}")
    b = Path(baseline_bbq_path)
    if not b.exists():
        sys.exit(f"ERROR: {b} not found. Run make_baseline_table.py once (it reads the baseline per-item log) "
                 f"and put its output next to the summaries.")
    bb = json.loads(b.read_text(encoding="utf-8"))
    sets[BASELINE]["bbq_by_category"] = bb["bbq_by_category"]
    sets[BASELINE]["_baseline_bbq_file"] = b
    for sid, d in sets.items():
        rows = d.get("bbq_by_category")
        if rows:
            df = pd.DataFrame(rows).set_index("category").reindex(CATS)
            if sid == BASELINE and "flips_pct" not in df:
                df["flips_pct"] = 0.0
            d["_bbq"] = df
        else:
            d["_bbq"] = None
        m = re.search(r"seed(\d+)", sid)
        d["_seed"] = int(m.group(1)) if m else None
    return sets


# ----------------------------------------------------------------------------- A. inventory and checks
def neuron_set(d):
    return {(int(l), int(i)) for l, idxs in d["neuron_indices"].items() for i in idxs}


def inventory(sets, l4_seeds):
    rows = []
    for sid, d in sets.items():
        per_layer = {int(l): len(v) for l, v in d["neuron_indices"].items()}
        env = d.get("env", {})
        rows.append({
            "set": sid, "kind": d["kind"], "n_neurons": d["n_neurons"],
            "neurons_per_layer": ", ".join(f"L{l}:{n}" for l, n in sorted(per_layer.items())),
            "saved_at": d.get("saved_at"), "summary_session_gpu": env.get("gpu"),
            "bbq_gpu_from_R1_exp": bbq_gpu_note(sid, l4_seeds),
            "has_bbq": d["_bbq"] is not None if sid != BASELINE else True,
            "caps_tasks": ", ".join(sorted((d.get("capabilities_raw") or {}).keys())),
            "caps_failed": ", ".join(d.get("capabilities_failed_tasks") or []),
            "batch_size": d.get("batch_size"), "wikitext_batch_size": d.get("wikitext_batch_size"),
            "eval_limit": d.get("eval_limit"), "manifest_sha256": d.get("manifest_sha256"),
            "torch": env.get("torch"), "transformers": env.get("transformers"), "lm_eval": env.get("lm_eval"),
            "optipfair": env.get("optipfair"),
        })
    return pd.DataFrame(rows)


def overlap_matrix(sets):
    ids = [s for s in sets if sets[s]["n_neurons"]]
    ns = {s: neuron_set(sets[s]) for s in ids}
    return pd.DataFrame([[len(ns[a] & ns[b]) for b in ids] for a in ids], index=ids, columns=ids)


def run_checks(sets, inv, ov, csv_path):
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    expected = [BASELINE, REAL_T, REAL_C] + [f"rand_religion_seed{i}" for i in range(N_RANDOM)]
    missing = [s for s in expected if s not in sets]
    extra = [s for s in sets if s not in expected]
    add("expected sets present (baseline, 2 real, 10 random)", not missing, f"missing: {missing}" if missing else "13 of 13")
    if extra:
        add("no unexpected sets", False, f"extra: {extra}")
    add("same manifest sha256 in all summaries", inv["manifest_sha256"].nunique() == 1,
        ", ".join(inv["manifest_sha256"].dropna().unique()))
    add("eval_limit is null in all summaries (no smoke-test runs)", inv["eval_limit"].isna().all(),
        str(inv["eval_limit"].unique().tolist()))
    for col in ["torch", "transformers", "lm_eval", "optipfair"]:
        add(f"same {col} version in all summaries", inv[col].nunique() == 1, str(inv[col].unique().tolist()))
    add("BBQ with 11 categories in every set with BBQ",
        all(d["_bbq"] is None or (d["_bbq"].index.tolist() == CATS and not d["_bbq"].isna().all(axis=1).any())
            for d in sets.values()), "")
    if REAL_T in sets:
        ref = sets[REAL_T]
        ref_layers = {int(l): len(v) for l, v in ref["neuron_indices"].items()}
        rnd = [s for s in sets if sets[s]["kind"] == "random"]
        same = [s for s in rnd if {int(l): len(v) for l, v in sets[s]["neuron_indices"].items()} == ref_layers]
        add("random sets have the same neurons per layer as Religion-20", len(same) == len(rnd),
            f"{len(same)} of {len(rnd)}")
        no_ov = [s for s in rnd if ov.loc[s, REAL_T] == 0]
        add("random sets do not overlap with Religion-20", len(no_ov) == len(rnd), f"{len(no_ov)} of {len(rnd)}")
        mx = max([ov.loc[a, b] for a in rnd for b in rnd if a < b] or [0])
        add("maximum overlap between two random sets (neurons)", True, str(int(mx)))
        for s in rnd:
            if len(neuron_set(sets[s])) != sets[s]["n_neurons"]:
                add(f"{s}: neurons are distinct", False, "")
    if REAL_C in sets and REAL_T in sets:
        add("overlap Religion-20 / Race-20 (neurons)", True, str(int(ov.loc[REAL_T, REAL_C])))
    if csv_path:
        csv = pd.read_csv(csv_path)
        worst, n = 0.0, 0
        for sid, d in sets.items():
            if d["_bbq"] is None or sid == BASELINE:
                continue
            c = csv[csv.set == sid].set_index("category").reindex(CATS)
            if c.empty:
                continue
            for m in ["amb_acc", "disamb_acc", "disamb_bias", "amb_bias", "flips_pct"]:
                worst = max(worst, float(np.nanmax(np.abs(c[m].to_numpy(float) - d["_bbq"][m].to_numpy(float)))))
            n += 1
        add(f"summaries agree with {Path(csv_path).name} (max abs difference over {n} sets)", worst < 1e-6, f"{worst:.2e}")
    else:
        add("cross-check with the consolidated CSV", True, "skipped: no CSV found")
    return pd.DataFrame(checks)


# ----------------------------------------------------------------------------- B. long table with deltas
def long_table(sets):
    base = sets[BASELINE]["_bbq"]
    rows = []
    for sid, d in sets.items():
        if d["_bbq"] is None or sid == BASELINE:
            continue
        for cat in CATS:
            r, b = d["_bbq"].loc[cat], base.loc[cat]
            row = {"set": sid, "kind": d["kind"], "seed": d["_seed"], "category": cat}
            row.update({m: r[m] for m in BBQ_METRICS})
            row["abs_disamb_bias"], row["abs_amb_bias"] = abs(r.disamb_bias), abs(r.amb_bias)
            row["d_amb_acc"], row["d_disamb_acc"] = r.amb_acc - b.amb_acc, r.disamb_acc - b.disamb_acc
            row["d_disamb_unknown"] = r.disamb_unknown - b.disamb_unknown
            row["d_disamb_bias"], row["d_amb_bias"] = r.disamb_bias - b.disamb_bias, r.amb_bias - b.amb_bias
            row["d_abs_disamb_bias"] = abs(r.disamb_bias) - abs(b.disamb_bias)
            row["d_abs_amb_bias"] = abs(r.amb_bias) - abs(b.amb_bias)
            row["d_flips_pct"] = r.flips_pct
            rows.append(row)
    df = pd.DataFrame(rows)
    brows = base.reset_index().rename(columns={"index": "category"})
    brows.insert(0, "set", BASELINE)
    return df, brows


# ----------------------------------------------------------------------------- C. panel of a category
def panel(long, cat, exclude_seeds=()):
    """Random distribution (value and delta) and position of each real set, for one category."""
    sub = long[long.category == cat]
    rnd = sub[(sub.kind == "random") & ~sub.seed.isin(list(exclude_seeds))]
    stat_rows, cmp_rows = [], []
    for name, vcol, dcol in MEASURES:
        sv, sd_ = describe(rnd[vcol]), describe(rnd[dcol])
        stat_rows.append({"measure": name, **{f"value_{k}": v for k, v in sv.items()},
                          **{f"delta_{k}": v for k, v in sd_.items() if k != "n"}})
        for rs in (REAL_T, REAL_C):
            rv = sub[sub.set == rs]
            if rv.empty:
                continue
            d = float(rv[dcol].iloc[0])
            ad = np.abs(rnd[dcol].to_numpy(float))
            n_ge = int((ad >= abs(d) - EPS).sum())
            cmp_rows.append({
                "measure": name, "real_set": rs, "value": float(rv[vcol].iloc[0]), "delta": d,
                "random_n": len(rnd), "random_delta_min": sd_["min"], "random_delta_max": sd_["max"],
                "random_delta_mean": sd_["mean"], "random_delta_sd": sd_["sd"],
                "rank_abs_delta_of_%d" % (len(rnd) + 1): 1 + int((ad > abs(d) + EPS).sum()),
                "n_random_abs_delta_ge_real": n_ge, "p_empirical": (n_ge + 1) / (len(rnd) + 1),
                "z_signed_delta": (d - sd_["mean"]) / sd_["sd"] if sd_["sd"] and not np.isnan(sd_["sd"]) else np.nan,
            })
    return pd.DataFrame(stat_rows), pd.DataFrame(cmp_rows)


# ----------------------------------------------------------------------------- F. collateral and pooled
def collateral(long):
    rows = []
    for cat in CATS:
        sub = long[long.category == cat]
        rnd = sub[sub.kind == "random"]
        row = {"category": cat}
        for rs, tag in ((REAL_T, "rel20"), (REAL_C, "race20")):
            r = sub[sub.set == rs]
            row[f"flips_{tag}"] = float(r.flips_pct.iloc[0]) if len(r) else np.nan
        row.update(flips_rand_mean=rnd.flips_pct.mean(), flips_rand_min=rnd.flips_pct.min(),
                   flips_rand_max=rnd.flips_pct.max())
        row["ratio_rel20_over_rand_mean"] = row["flips_rel20"] / row["flips_rand_mean"]
        row["ratio_race20_over_rand_mean"] = row["flips_race20"] / row["flips_rand_mean"]
        for col, tag in (("d_disamb_acc", "dacc"), ("d_disamb_bias", "dbias")):
            for rs, t2 in ((REAL_T, "rel20"), (REAL_C, "race20")):
                r = sub[sub.set == rs]
                row[f"{tag}_{t2}"] = float(r[col].iloc[0]) if len(r) else np.nan
            row[f"{tag}_rand_mean"], row[f"{tag}_rand_min"], row[f"{tag}_rand_max"] = (
                rnd[col].mean(), rnd[col].min(), rnd[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(sets):
    w = pd.Series(ITEMS_PER_CONTEXT)[CATS]
    base = sets[BASELINE]["_bbq"]
    rows = []
    for sid, d in sets.items():
        t = d["_bbq"]
        if t is None:
            continue
        row = {"set": sid, "kind": d["kind"],
               "pooled_amb_acc": float((t.amb_acc * w).sum() / w.sum()),
               "pooled_disamb_acc": float((t.disamb_acc * w).sum() / w.sum()),
               "pooled_flips_pct": float((t.flips_pct * w).sum() / w.sum())}
        rows.append(row)
    df = pd.DataFrame(rows)
    b = df[df.set == BASELINE].iloc[0]
    df["d_pooled_amb_acc"] = df.pooled_amb_acc - b.pooled_amb_acc
    df["d_pooled_disamb_acc"] = df.pooled_disamb_acc - b.pooled_disamb_acc
    return df


# ----------------------------------------------------------------------------- G. capabilities
def capabilities(sets):
    rows = []
    for sid, d in sets.items():
        raw = d.get("capabilities_raw") or {}
        row = {"set": sid, "kind": d["kind"], "seed": d["_seed"]}
        for col, (task, key) in CAP_FIELDS.items():
            row[col] = num((raw.get(task) or {}).get(key))
        rows.append(row)
    df = pd.DataFrame(rows)
    b = df[df.set == BASELINE].iloc[0]
    for col in CAP_FIELDS:
        df["d_" + col] = df[col] - b[col]
    return df


def cap_random_stats(cap, exclude_seeds=()):
    rnd = cap[(cap.kind == "random") & ~cap.seed.isin(list(exclude_seeds))]
    rows = []
    for col in CAP_FIELDS:
        if rnd[col].notna().any():
            rows.append({"measure": col, **describe(rnd[col].dropna()),
                         **{"delta_" + k: v for k, v in describe(rnd["d_" + col].dropna()).items() if k != "n"}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main tables for R1 / paper
def main_tables(long, cap, sets):
    base = sets[BASELINE]["_bbq"].loc[TARGET]
    sub = long[long.category == TARGET]
    rnd = sub[sub.kind == "random"]
    rt = sub[sub.set == REAL_T].iloc[0]
    rc = sub[sub.set == REAL_C].iloc[0]
    rng = lambda s, sign=False: f"{fmt(s.min(), 2, sign)} to {fmt(s.max(), 2, sign)}"
    simple = [
        "| | Religion bias | Change | Changed answers |",
        "|---|---|---|---|",
        f"| Unpruned | {fmt(base.disamb_bias)} | - | - |",
        f"| 20 selected neurons | {fmt(rt.disamb_bias)} | {fmt(rt.d_disamb_bias, 2, True)} | {fmt(rt.flips_pct, 1)} % |",
        f"| 20 random neurons ({len(rnd)} sets) | {rng(rnd.disamb_bias)} | {rng(rnd.d_disamb_bias, True)} | "
        f"{fmt(rnd.flips_pct.min(), 1)} to {fmt(rnd.flips_pct.max(), 1)} % |",
    ]
    cb, cr, ct = (cap[cap.set == s].iloc[0] for s in (BASELINE, REAL_T, REAL_C))
    crn = cap[cap.kind == "random"]
    ext = [
        "| Set | Religion disamb. bias (change) | Religion disamb. acc. % (change) | Religion changed % | WikiText ppl (change) |",
        "|---|---|---|---|---|",
        f"| Unpruned | {fmt(base.disamb_bias)} | {fmt(base.disamb_acc)} | - | {fmt(cb.wikitext_word_ppl, 4)} |",
        f"| Religion-20 | {fmt(rt.disamb_bias)} ({fmt(rt.d_disamb_bias, 2, True)}) | {fmt(rt.disamb_acc)} ({fmt(rt.d_disamb_acc, 2, True)}) | "
        f"{fmt(rt.flips_pct)} | {fmt(cr.wikitext_word_ppl, 4)} ({fmt(cr.d_wikitext_word_ppl, 2, True)}) |",
        f"| Race-20 (cross control) | {fmt(rc.disamb_bias)} ({fmt(rc.d_disamb_bias, 2, True)}) | {fmt(rc.disamb_acc)} ({fmt(rc.d_disamb_acc, 2, True)}) | "
        f"{fmt(rc.flips_pct)} | {fmt(ct.wikitext_word_ppl, 4)} ({fmt(ct.d_wikitext_word_ppl, 2, True)}) |",
        f"| Random, mean (range), n={len(rnd)} | {fmt(rnd.disamb_bias.mean())} ({rng(rnd.disamb_bias)}) | "
        f"{fmt(rnd.disamb_acc.mean())} ({rng(rnd.disamb_acc)}) | {fmt(rnd.flips_pct.mean())} ({rng(rnd.flips_pct)}) | "
        f"{fmt(crn.wikitext_word_ppl.mean(), 4)} ({fmt(crn.wikitext_word_ppl.min(), 4)} to {fmt(crn.wikitext_word_ppl.max(), 4)}) |",
    ]
    return "\n".join(simple), "\n".join(ext)


# ----------------------------------------------------------------------------- H. figure
def figure(long, cap, out_dir, l4_seeds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: figure skipped")
        return []
    sub = long[long.category == TARGET]
    rnd = sub[sub.kind == "random"].sort_values("seed")
    cr = cap[cap.kind == "random"].sort_values("seed")
    panels = [
        ("Change in Religion disambiguated bias (pp)", rnd.d_disamb_bias.to_numpy(), rnd.seed.to_numpy(),
         {REAL_T: float(sub[sub.set == REAL_T].d_disamb_bias.iloc[0]), REAL_C: float(sub[sub.set == REAL_C].d_disamb_bias.iloc[0])}, 0.0),
        ("Religion answers changed vs. unpruned (%)", rnd.flips_pct.to_numpy(), rnd.seed.to_numpy(),
         {REAL_T: float(sub[sub.set == REAL_T].flips_pct.iloc[0]), REAL_C: float(sub[sub.set == REAL_C].flips_pct.iloc[0])}, 0.0),
        ("Change in WikiText word perplexity", cr.d_wikitext_word_ppl.to_numpy(), cr.seed.to_numpy(),
         {REAL_T: float(cap[cap.set == REAL_T].d_wikitext_word_ppl.iloc[0]), REAL_C: float(cap[cap.set == REAL_C].d_wikitext_word_ppl.iloc[0])}, 0.0),
    ]
    style = {REAL_T: dict(color="#b2182b", ls="-", label="Religion-20 (selected)"),
             REAL_C: dict(color="#2166ac", ls="--", label="Race-20 (selected, cross control)")}
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 6.4))
    for ax, (title, vals, seeds, reals, ref) in zip(axes, panels):
        ys = np.linspace(-0.25, 0.25, len(vals))
        is_l4 = np.array([int(s) in l4_seeds for s in seeds])
        ax.scatter(vals[~is_l4], ys[~is_l4], s=34, color="#555555", label="random set (A100)", zorder=3)
        if is_l4.any():
            ax.scatter(vals[is_l4], ys[is_l4], s=40, facecolors="none", edgecolors="#555555", linewidths=1.4,
                       label="random set (BBQ on L4)", zorder=3)
        ax.axvline(ref, color="#999999", ls=":", lw=1, label="unpruned")
        for rs, v in reals.items():
            ax.axvline(v, lw=2, zorder=2, **style[rs])
        ax.set_yticks([]); ax.set_ylim(-0.6, 0.6); ax.set_title(title, fontsize=9, loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    seen, H, L = set(), [], []
    for hh, ll in list(zip(h, l)) + list(zip(h2, l2)):
        if ll not in seen:
            seen.add(ll); H.append(hh); L.append(ll)
    fig.legend(H, L, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    paths = []
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"e2_religion_strip.{ext}"
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--baseline-bbq")
    ap.add_argument("--csv")
    ap.add_argument("--l4-seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir).expanduser() if a.out_dir else inp.parent / "analysis" / stamp
    out.mkdir(parents=True, exist_ok=True)
    base_path = Path(a.baseline_bbq).expanduser() if a.baseline_bbq else inp / "baseline_bbq_by_category.json"
    csv_path = Path(a.csv).expanduser() if a.csv else None
    if csv_path is None:
        cands = sorted(inp.parent.glob("e2_bbq_by_category_*.csv"))
        csv_path = cands[-1] if cands else None

    sets = load_sets(inp, base_path)
    l4 = set(a.l4_seeds)

    # A
    inv = inventory(sets, l4)
    ov = overlap_matrix(sets)
    chk = run_checks(sets, inv, ov, csv_path)
    # B
    long, base_rows = long_table(sets)
    # C, D, E
    tabs = {}
    for cat in (TARGET, CROSS):
        st, cm = panel(long, cat)
        tabs[f"random_stats_{cat}"], tabs[f"real_vs_random_{cat}"] = st, cm
        st2, cm2 = panel(long, cat, exclude_seeds=l4)
        tabs[f"random_stats_{cat}_without_L4_seeds"], tabs[f"real_vs_random_{cat}_without_L4_seeds"] = st2, cm2
    # F
    coll, pool = collateral(long), pooled(sets)
    # G
    cap = capabilities(sets)
    cap_stats = cap_random_stats(cap)
    cap_stats_noL4 = cap_random_stats(cap, exclude_seeds=l4)
    simple_md, ext_md = main_tables(long, cap, sets)

    # write
    def w(name, df):
        df.to_csv(out / f"{name}.csv", index=False)

    w("inventory", inv); w("checks", chk); ov.reset_index(names="set").to_csv(out / "overlap_matrix.csv", index=False)
    w("bbq_long", long); w("bbq_baseline", base_rows)
    for k, v in tabs.items():
        w(k, v)
    w("collateral_by_category", coll); w("pooled", pool); w("capabilities", cap)
    w("capabilities_random_stats", cap_stats); w("capabilities_random_stats_without_L4_seeds", cap_stats_noL4)
    (out / "table_main.md").write_text(simple_md + "\n", encoding="utf-8")
    (out / "table_main_extended.md").write_text(ext_md + "\n", encoding="utf-8")

    figs = [] if a.no_figures else figure(long, cap, out, l4)

    # inputs record
    files = sorted(inp.glob("*_summary.json")) + [base_path] + ([csv_path] if csv_path else [])
    inputs = {"input_dir": str(inp), "files": [{"file": str(f), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files]}
    meta = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(__file__), "repo_commit": git_commit(Path(__file__).resolve().parent),
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "l4_seeds": sorted(l4), "args": vars(a)}
    (out / "inputs.json").write_text(json.dumps({"meta": meta, **inputs}, indent=2), encoding="utf-8")

    full = {"meta": meta, "inventory": records(inv), "checks": records(chk), "overlap_matrix": json.loads(ov.to_json()),
            "bbq_long": records(long), "bbq_baseline": records(base_rows),
            **{k: records(v) for k, v in tabs.items()}, "collateral_by_category": records(coll), "pooled": records(pool),
            "capabilities": records(cap), "capabilities_random_stats": records(cap_stats),
            "capabilities_random_stats_without_L4_seeds": records(cap_stats_noL4)}
    (out / "results_full.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    # report
    n_bbq = sum(1 for s, d in sets.items() if d["_bbq"] is not None and s != BASELINE)
    rep = [f"# E2 analysis report ({meta['generated_at']})", "",
           f"Input: `{inp}`. Output: `{out}`. Sets: {len(sets)} summaries (baseline + {n_bbq} with BBQ). "
           "Descriptive only: no thresholds, no verdicts.", "",
           "## Main table (as agreed)", "", simple_md, "", "## Extended table", "", ext_md, "",
           "## Checks", "", md_table(chk), "",
           f"## Religion: random sets vs real sets", "", md_table(tabs[f'random_stats_{TARGET}']), "",
           md_table(tabs[f'real_vs_random_{TARGET}'], 3), "",
           f"## Religion without the L4 seeds {sorted(l4)}", "", md_table(tabs[f'random_stats_{TARGET}_without_L4_seeds']), "",
           md_table(tabs[f'real_vs_random_{TARGET}_without_L4_seeds'], 3), "",
           f"## Race/ethnicity: random sets vs real sets", "", md_table(tabs[f'random_stats_{CROSS}']), "",
           md_table(tabs[f'real_vs_random_{CROSS}'], 3), "",
           "## Effect on every category (changed answers %, change in disambiguated accuracy and bias)", "", md_table(coll), "",
           "## Pooled over categories (weighted by items)", "", md_table(pool), "",
           "## Capabilities", "", md_table(cap[["set", "wikitext_word_ppl", "d_wikitext_word_ppl", "mmlu_acc", "arc_challenge_acc_norm",
                                                "hellaswag_acc_norm", "hellaswag_es_acc_norm"]], 4), "",
           md_table(cap_stats, 4), "", "## Inventory", "", md_table(inv[["set", "kind", "n_neurons", "neurons_per_layer", "saved_at",
                                                                         "summary_session_gpu", "bbq_gpu_from_R1_exp"]]), ""]
    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")

    print(f"Output folder: {out}")
    print(f"Checks: {int(chk.ok.sum())} of {len(chk)} passed" + ("" if chk.ok.all() else "  <-- LOOK AT checks.csv"))
    for _, r in chk[~chk.ok].iterrows():
        print(f"  FAILED: {r['check']}  {r['detail']}")
    print("\n" + simple_md + "\n")
    print("Files:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
