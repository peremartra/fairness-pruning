#!/usr/bin/env python3
"""
E3 per-item analysis for Llama-3.2-3B EN including random sets (T3).

Combines:
  - E3 baseline and 6 real sets from `<e3_unit_dir>/items/`
  - E3 Gender-20 random sets (`rand_gender20_seed*`) from `<random_items_dir>` (default `<e3_unit_dir>/items_random/`)
  - E2 Religion-20 random sets (`rand_religion_seed0..4`, optionally `5..9`) from `<e2_items_dir>`, mapped to E3 names
    (`rand_religion20_seed0..4` / `5..9`) after re-asserting exact neuron-index identity against the E3 manifest and
    exact baseline source_log_sha256 identity.

Changed-answer definition (primary, consistent with E2 and `e3_item_analysis.py` bootstrap/groups/overlap/margin):
  pred != base_pred over the 3-way prediction {0: person0, 1: person1, 2: unknown}, pooling ambig + disambig items
  in each BBQ category (denominator = all items of the category).

Outputs to a NEW timestamped folder `<e3_unit_dir>/analysis/<timestamp>_items_3b_random/`.
Does not modify any existing script or Drive file.
"""
import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse unmodified analysis/e3/e3_item_analysis.py helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_item_analysis as base_mod

CATS = base_mod.CATS
SET_TO_BBQ_CAT = {
    "Religion": "Religion",
    "Race": "Race_ethnicity",
    "Gender": "Gender_identity",
}

# Verified BBQ GPU hardware (from pretty_env_info in the per-item lm_eval BBQ logs)
KNOWN_BBQ_GPU = {
    "baseline": "A100-SXM4-40GB",
    "real_religion20": "A100-SXM4-80GB",
    "real_race20": "A100-SXM4-80GB",
    "real_gender20": "A100-SXM4-80GB",
    "real_religion5": "A100-SXM4-80GB",
    "real_race5": "A100-SXM4-80GB",
    "real_gender5": "A100-SXM4-80GB",
    "rand_gender20_seed0": "A100-SXM4-80GB",
    "rand_gender20_seed0_L4": "L4",
    "rand_gender20_seed1": "A100-SXM4-80GB",
    "rand_gender20_seed2": "A100-SXM4-80GB",
    "rand_gender20_seed3": "A100-SXM4-80GB",
    "rand_gender20_seed4": "A100-SXM4-80GB",
    "rand_religion20_seed0": "L4",
    "rand_religion20_seed1": "L4",
    "rand_religion20_seed2": "A100-SXM4-40GB",
    "rand_religion20_seed3": "A100-SXM4-40GB",
    "rand_religion20_seed4": "A100-SXM4-40GB",
    "rand_religion20_seed5": "A100-SXM4-40GB",
    "rand_religion20_seed6": "A100-SXM4-40GB",
    "rand_religion20_seed7": "A100-SXM4-40GB",
    "rand_religion20_seed8": "A100-SXM4-40GB",
    "rand_religion20_seed9": "A100-SXM4-40GB",
}


def norm_neurons(by_layer):
    return {int(k): sorted(int(x) for x in v) for k, v in by_layer.items()}


def load_table(csv_path):
    t = pd.read_csv(csv_path, keep_default_na=False, na_values=[""]).set_index("doc_id").sort_index()
    for c in ("group0", "group1"):
        t[c] = t[c].fillna(base_mod.EMPTY_LABEL).astype(str)
    t["state"] = base_mod.state(t)
    lls = t[["ll0", "ll1", "ll_unk"]].to_numpy()
    srt = np.sort(lls, axis=1)
    t["margin"] = srt[:, -1] - srt[:, -2]
    return t


def load_combined(e3_items_dir, rand_items_dir, e2_items_dir, manifest_path, include_e2_5_9=True):
    e3_meta = json.load(open(e3_items_dir / "sets.json", encoding="utf-8"))["sets"]
    e2_meta = json.load(open(e2_items_dir / "sets.json", encoding="utf-8"))["sets"]
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    # Assert baseline source_log_sha256 matches between E2 and E3
    assert e3_meta["baseline"]["source_log_sha256"] == e2_meta["baseline"]["source_log_sha256"], (
        "Baseline source_log_sha256 mismatch between E3 and E2!"
    )

    meta = {}
    tables = {}
    source_files = [e3_items_dir / "sets.json", e2_items_dir / "sets.json", manifest_path]

    # 1. Load E3 baseline + 6 real sets
    for sid, m in e3_meta.items():
        p = e3_items_dir / m["table"]
        meta[sid] = dict(m)
        meta[sid]["bbq_gpu"] = KNOWN_BBQ_GPU.get(sid, m.get("summary_session_gpu") or "unknown")
        tables[sid] = load_table(p)
        source_files.append(p)

    # 2. Load E3 random sets: A100 seed0 from items_random_seed0_a100/ (if present), seeds 1..4 from items_random/,
    #    plus L4 seed0 as a labelled sensitivity row (`rand_gender20_seed0_L4`)
    seed0_a100_dir = rand_items_dir.parent / "items_random_seed0_a100"
    if seed0_a100_dir.exists() and (seed0_a100_dir / "sets.json").exists():
        s0_meta = json.load(open(seed0_a100_dir / "sets.json", encoding="utf-8"))["sets"]
        source_files.append(seed0_a100_dir / "sets.json")
        if "rand_gender20_seed0" in s0_meta:
            m = dict(s0_meta["rand_gender20_seed0"])
            assert norm_neurons(m["neuron_indices"]) == norm_neurons(manifest["random_sets"]["rand_gender20_seed0"]["by_layer"]), (
                "rand_gender20_seed0 (A100): neuron indices differ from E3 manifest!"
            )
            p = seed0_a100_dir / m["table"]
            m["bbq_gpu"] = "A100-SXM4-80GB"
            meta["rand_gender20_seed0"] = m
            tables["rand_gender20_seed0"] = load_table(p)
            source_files.append(p)

    if rand_items_dir.exists() and (rand_items_dir / "sets.json").exists():
        r_meta = json.load(open(rand_items_dir / "sets.json", encoding="utf-8"))["sets"]
        source_files.append(rand_items_dir / "sets.json")
        for sid, m in r_meta.items():
            if sid == "rand_gender20_seed0" and "rand_gender20_seed0" in meta:
                # Keep the L4 seed0 as a labelled sensitivity row
                l4_csv = rand_items_dir / "rand_gender20_seed0_L4.csv.gz"
                if l4_csv.exists():
                    m_l4 = dict(m)
                    m_l4["set"] = "rand_gender20_seed0_L4"
                    m_l4["kind"] = "random_sensitivity"
                    m_l4["seed"] = 0
                    m_l4["bbq_gpu"] = "L4"
                    m_l4["table"] = l4_csv.name
                    meta["rand_gender20_seed0_L4"] = m_l4
                    tables["rand_gender20_seed0_L4"] = load_table(l4_csv)
                    source_files.append(l4_csv)
                continue
            if sid in meta:
                continue
            # Assert neuron indices match manifest
            if sid in manifest["random_sets"]:
                assert norm_neurons(m["neuron_indices"]) == norm_neurons(manifest["random_sets"][sid]["by_layer"]), (
                    f"{sid}: neuron indices differ from E3 manifest!"
                )
            p = rand_items_dir / m["table"]
            meta[sid] = dict(m)
            meta[sid]["bbq_gpu"] = KNOWN_BBQ_GPU.get(sid, m.get("summary_session_gpu") or "unknown")
            tables[sid] = load_table(p)
            source_files.append(p)

    # 3. Load E2 Religion random sets (0..4 required; 5..9 optional) mapped to E3 names
    e2_seeds = list(range(10)) if include_e2_5_9 else list(range(5))
    for s in e2_seeds:
        e2_sid = f"rand_religion_seed{s}"
        e3_sid = f"rand_religion20_seed{s}"
        assert e2_sid in e2_meta, f"Missing {e2_sid} in E2 items!"
        m = dict(e2_meta[e2_sid])
        # Re-assert exact neuron identity E2 vs E3 manifest for seeds 0..4
        if s < 5:
            e3_by = norm_neurons(manifest["random_sets"][e3_sid]["by_layer"])
            e2_by = norm_neurons(m["neuron_indices"])
            assert e2_by == e3_by, f"Neuron mismatch between E2 {e2_sid} and E3 {e3_sid}!"
        m["set"] = e3_sid
        m["e2_original_set"] = e2_sid
        m["category"] = "Religion"
        m["k"] = 20
        m["mimics"] = "real_religion20"
        m["seed"] = s
        m["bbq_gpu"] = KNOWN_BBQ_GPU.get(e3_sid, "unknown")
        p = e2_items_dir / m["table"]
        meta[e3_sid] = m
        tables[e3_sid] = load_table(p)
        source_files.append(p)

    # Assert exact doc_id and metadata alignment across all sets
    base = "baseline"
    for sid, t in tables.items():
        assert t.index.equals(tables[base].index), f"{sid}: items differ from baseline"
        cols = ["category", "context", "group0", "group1", "target"]
        bad = [c for c in cols if not (t[c] == tables[base][c]).all()]
        assert not bad, f"{sid}: item metadata differs from baseline in {bad}"

    order = (
        [base]
        + sorted([s for s in meta if meta[s]["kind"] == "real"], key=lambda s: (-int(meta[s]["k"] or 0), s))
        + sorted(
            [s for s in meta if meta[s]["kind"] == "random"],
            key=lambda s: (s.rsplit("_seed", 1)[0], int(meta[s]["seed"])),
        )
        + sorted([s for s in meta if meta[s]["kind"] == "random_sensitivity"])
    )
    return meta, tables, base, order, source_files


def flips_and_specificity(meta, tables, base, order):
    b = tables[base]
    base_bias = {c: base_mod.disamb_bias(b[b["category"] == c]) for c in CATS}
    base_dacc = {
        c: 100.0 * b[(b["category"] == c) & (b["context"] == "disambig")]["correct"].mean() for c in CATS
    }

    long_rows = []
    spec_rows = []
    for sid in order:
        if sid == base:
            continue
        t = tables[sid]
        cat_flips = {}
        for c in CATS:
            mc = b["category"] == c
            ch = t.loc[mc, "pred"] != b.loc[mc, "pred"]
            fp = 100.0 * float(ch.mean())
            cat_flips[c] = fp
            db = base_mod.disamb_bias(t[mc])
            da = 100.0 * float(t[mc & (t["context"] == "disambig")]["correct"].mean())
            long_rows.append({
                "set": sid,
                "kind": meta[sid]["kind"],
                "set_category": meta[sid].get("category"),
                "k": meta[sid].get("k"),
                "bbq_gpu": meta[sid].get("bbq_gpu"),
                "category": c,
                "n_items": int(mc.sum()),
                "changed": int(ch.sum()),
                "flips_pct": fp,
                "disamb_bias": db,
                "d_disamb_bias": db - base_bias[c],
                "disamb_acc": da,
                "d_disamb_acc": da - base_dacc[c],
            })
        set_cat = meta[sid].get("category")
        tgt_cat = SET_TO_BBQ_CAT.get(set_cat, set_cat)
        tgt_flips = cat_flips[tgt_cat]
        others = [v for k, v in cat_flips.items() if k != tgt_cat]
        other_mean = float(np.mean(others))
        spec = tgt_flips / other_mean if other_mean > 0 else np.nan
        rank = 1 + sum(1 for v in cat_flips.values() if v > tgt_flips + 1e-12)
        spec_rows.append({
            "set": sid,
            "kind": meta[sid]["kind"],
            "set_category": set_cat,
            "target_category": tgt_cat,
            "k": meta[sid].get("k"),
            "bbq_gpu": meta[sid].get("bbq_gpu"),
            "target_flips_pct": tgt_flips,
            "other_10_mean_flips_pct": other_mean,
            "specificity": spec,
            "target_rank_of_11": rank,
        })
    return pd.DataFrame(long_rows), pd.DataFrame(spec_rows)


def paired_comparisons(spec_df):
    rows = []
    specs = [
        ("Religion·20 (E2 seeds 0-4, primary)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(5)]),
        ("Religion·20 (E2 seeds 0-9, all)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(10)]),
        ("Religion·20 (E2 seeds 2-4, A100-only)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(2, 5)]),
        ("Religion·20 (E2 seeds 2-9, A100-only)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(2, 10)]),
        ("Gender·20 (seeds 0-4, A100-only, primary)", "real_gender20", [f"rand_gender20_seed{i}" for i in range(5)]),
        ("Gender·20 (L4 seed0 + A100 seeds 1-4, sensitivity)", "real_gender20", ["rand_gender20_seed0_L4"] + [f"rand_gender20_seed{i}" for i in range(1, 5)]),
        ("Gender·20 (L4 seed0 alone, sensitivity)", "real_gender20", ["rand_gender20_seed0_L4"]),
    ]
    for label, real_sid, rand_sids in specs:
        r_row = spec_df[spec_df["set"] == real_sid]
        rnd = spec_df[spec_df["set"].isin(rand_sids)]
        if r_row.empty or rnd.empty:
            continue
        r = r_row.iloc[0]
        gpus = sorted(set(rnd["bbq_gpu"].tolist()))
        mixes_gpu = any(g != r["bbq_gpu"] for g in gpus)
        rows.append({
            "comparison": label,
            "real_set": real_sid,
            "real_gpu": r["bbq_gpu"],
            "n_random": len(rnd),
            "random_gpus": ", ".join(gpus),
            "mixes_gpu_l4_vs_a100": any("L4" in g for g in gpus) and ("A100" in r["bbq_gpu"]),
            "real_target_flips_pct": r["target_flips_pct"],
            "rand_target_flips_mean": float(rnd["target_flips_pct"].mean()),
            "rand_target_flips_min": float(rnd["target_flips_pct"].min()),
            "rand_target_flips_max": float(rnd["target_flips_pct"].max()),
            "ratio_real_over_rand_mean": float(r["target_flips_pct"] / rnd["target_flips_pct"].mean()),
            "real_other10_mean_pct": r["other_10_mean_flips_pct"],
            "rand_other10_mean_pct": float(rnd["other_10_mean_flips_pct"].mean()),
            "real_specificity": r["specificity"],
            "rand_specificity_mean": float(rnd["specificity"].mean()),
            "rand_specificity_min": float(rnd["specificity"].min()),
            "rand_specificity_max": float(rnd["specificity"].max()),
        })
    return pd.DataFrame(rows)


def md_table(df, nd=2):
    d = df.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        vals = []
        for v in row.tolist():
            if isinstance(v, (float, np.floating)):
                vals.append("-" if np.isnan(v) else f"{v:.{nd}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", required=True, help="Path to MyDrive/fair_pruning")
    ap.add_argument("--out-dir", help="Output directory (default: results/e3/llama-3.2-3b_en/analysis/<timestamp>_items_3b_random)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-e2-seeds-5-9", action="store_true", help="Exclude E2 Religion random seeds 5-9")
    a = ap.parse_args()

    drive = Path(a.drive_root)
    unit_dir = drive / "results/e3/llama-3.2-3b_en"
    e3_items_dir = unit_dir / "items"
    rand_items_dir = unit_dir / "items_random"
    e2_items_dir = drive / "results/e2/items"
    manifest_path = drive / "results/e3/llama-3.2-3b_en_E3_manifest.json"

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) if a.out_dir else unit_dir / "analysis" / f"{stamp}_items_3b_random"
    assert not out.exists(), f"Refusing to overwrite existing output dir: {out}"
    out.mkdir(parents=True, exist_ok=False)

    meta, tables, base, order, source_files = load_combined(
        e3_items_dir, rand_items_dir, e2_items_dir, manifest_path, include_e2_5_9=not a.no_e2_seeds_5_9
    )
    print(f"Loaded {len(tables)} sets ({base} + {len(order) - 1}); writing to {out}")

    flips_df, spec_df = flips_and_specificity(meta, tables, base, order)
    paired_df = paired_comparisons(spec_df)

    # Standard e3_item_analysis outputs
    dir_df, trans_df = base_mod.direction(meta, tables, base, order)
    grp_df = base_mod.groups(meta, tables, base, order)
    ov_all = base_mod.overlap(tables, base, order)
    target_cats = ["Religion", "Race_ethnicity", "Gender_identity"]
    ov_cat = {c: base_mod.overlap(tables, base, order, c) for c in target_cats}
    med_df, dec_df = base_mod.margin(meta, tables, base, order)
    bs_df = base_mod.bootstrap(meta, tables, base, order, a.bootstrap, a.seed)

    flips_df.to_csv(out / "flips_by_category.csv", index=False)
    spec_df.to_csv(out / "specificity_summary.csv", index=False)
    paired_df.to_csv(out / "paired_real_vs_random.csv", index=False)
    dir_df.to_csv(out / "direction.csv", index=False)
    trans_df.to_csv(out / "transitions.csv", index=False)
    grp_df.to_csv(out / "groups.csv", index=False)
    ov_all.to_csv(out / "overlap_changed_items_all.csv")
    for c, m in ov_cat.items():
        m.to_csv(out / f"overlap_changed_items_{c}.csv")
    med_df.to_csv(out / "margin_changed_vs_unchanged.csv", index=False)
    dec_df.to_csv(out / "margin_deciles.csv", index=False)
    bs_df.to_csv(out / "bootstrap.csv", index=False)

    # Compact markdown report
    R = [
        "# E3 Per-Item Analysis — `llama-3.2-3b_en` (Real + Random Sets)",
        "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')}. Baseline: `baseline` (`A100-SXM4-40GB`).",
        "Changed-answer definition: `pred != base_pred` over 3-way prediction `{0: person0, 1: person1, 2: unknown}`, pooling `ambig` + `disambig` items per BBQ category (exact match with E2 `flips_pct` and `e3_item_analysis.py` `bootstrap`/`groups`/`overlap`/`margin`).",
        "",
        "## 1. Specificity Summary (Target Category vs Mean of Other 10 Categories)",
        "",
        md_table(spec_df.set_index("set")),
        "",
        "## 2. Paired Comparisons (Real vs Random of Same Spec)",
        "",
        md_table(paired_df.set_index("comparison")),
        "",
        "## 3. Flips (%) by BBQ Category (All Sets)",
        "",
        md_table(flips_df.pivot(index="set", columns="category", values="flips_pct").reindex([s for s in order if s != base])[CATS]),
        "",
    ]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")

    json.dump(
        {
            "specificity_summary": spec_df.to_dict("records"),
            "paired_real_vs_random": paired_df.to_dict("records"),
            "flips_by_category": flips_df.to_dict("records"),
            "bootstrap": bs_df.to_dict("records"),
        },
        open(out / "results_full.json", "w", encoding="utf-8"),
        indent=1,
        default=float,
    )
    json.dump(
        {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "args": vars(a),
            "script_sha256": base_mod.sha256(__file__),
            "repo_commit": base_mod.git_commit(Path(__file__).parent),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sets": meta,
        },
        open(out / "inputs.json", "w", encoding="utf-8"),
        indent=2,
    )
    print(f"✅ Wrote T3 analysis to {out}")


if __name__ == "__main__":
    main()
