#!/usr/bin/env python3
"""
Extract A100 reruns of Llama-3.2-3B EN `rand_religion20_seed0` and `rand_religion20_seed1`
and run the full all-A100 E3 per-item analysis for `llama-3.2-3b_en`.

Strictly non-destructive on Google Drive:
  - Writes extracted item tables to a NEW folder:
      `<drive-root>/results/e3/llama-3.2-3b_en/items_religion_a100_seed01/`
    (skips any already-extracted complete file; never overwrites).
  - Writes analysis outputs to a NEW timestamped folder:
      `<drive-root>/results/e3/llama-3.2-3b_en/analysis/<timestamp>_items_3b_random_a100/`
  - Reuses unmodified `extract_items.py`, `e3_item_analysis.py`, and `e3_item_analysis_3b_random.py`.
"""
import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_item_analysis as base_mod
import e3_item_analysis_3b_random as r3b_mod
import extract_items as ext_mod

CATS = base_mod.CATS

# All primary sets are now 100% A100; historical L4 runs are preserved only as `*_L4` sensitivity rows.
KNOWN_BBQ_GPU_A100 = {
    **r3b_mod.KNOWN_BBQ_GPU,
    "rand_religion20_seed0": "A100-SXM4-80GB",
    "rand_religion20_seed1": "A100-SXM4-80GB",
    "rand_religion20_seed0_L4": "L4",
    "rand_religion20_seed1_L4": "L4",
}


def extract_religion_a100_seed01(drive: Path, manifest: dict) -> Path:
    unit_dir = drive / "results/e3/llama-3.2-3b_en"
    out_dir = unit_dir / "items_religion_a100_seed01"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = drive / "checkpoints_e2_a100_rerun/results/lm_evals"
    meta_run_path = drive / "checkpoints_e2_a100_rerun/run_metadata.json"
    run_meta = json.load(open(meta_run_path, encoding="utf-8")) if meta_run_path.exists() else {}
    gpu_name = (run_meta.get("gpu") or "NVIDIA A100-SXM4-80GB").replace("NVIDIA ", "").strip()

    sets_path = out_dir / "sets.json"
    meta = json.load(open(sets_path, encoding="utf-8")) if sets_path.exists() else {"unit": "llama-3.2-3b_en", "sets": {}}

    e2_meta = json.load(open(drive / "results/e2/items/sets.json", encoding="utf-8"))["sets"]

    for seed in (0, 1):
        sid = f"rand_religion20_seed{seed}"
        e2_sid = f"rand_religion_seed{seed}"
        log_path = logs_dir / f"llama_3.2_3b_en__{sid}_bbq.json"
        assert log_path.exists(), f"Missing A100 BBQ log: {log_path}"

        # Assert exact neuron-index identity between E3 manifest and E2 sets.json
        e3_by = r3b_mod.norm_neurons(manifest["random_sets"][sid]["by_layer"])
        e2_by = r3b_mod.norm_neurons(e2_meta[e2_sid]["neuron_indices"])
        assert e3_by == e2_by, f"Neuron mismatch between E3 manifest and E2 for {sid}!"

        out_csv = out_dir / f"{sid}.csv.gz"
        if out_csv.exists() and sid in meta["sets"] and meta["sets"][sid].get("n_items") == ext_mod.N_ITEMS:
            print(f"[{sid}] already extracted in {out_dir} — skipped (no overwrite)", flush=True)
            continue

        t0 = dt.datetime.now()
        print(f"[{sid}] extracting {log_path} ({log_path.stat().st_size / 1e6:.1f} MB)...", flush=True)
        n, mism = ext_mod.extract(log_path, out_csv)
        assert n == ext_mod.N_ITEMS, f"{sid}: expected {ext_mod.N_ITEMS} items, got {n}"
        assert mism == 0, f"{sid}: got {mism} pred mismatches vs lm-eval acc"

        meta["sets"][sid] = {
            "set": sid,
            "kind": "random",
            "category": "Religion",
            "k": 20,
            "mimics": "real_religion20",
            "seed": seed,
            "n_neurons": 20,
            "neuron_indices": manifest["random_sets"][sid]["by_layer"],
            "source_log": str(log_path),
            "source_log_bytes": log_path.stat().st_size,
            "source_log_sha256": ext_mod.sha256(log_path),
            "table": out_csv.name,
            "n_items": n,
            "pred_mismatches_vs_lm_eval_acc": mism,
            "manifest_sha256": e2_meta[e2_sid].get("manifest_sha256"),
            "summary_session_gpu": gpu_name,
            "bbq_gpu": gpu_name,
            "extracted_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        meta["python"] = platform.python_version()
        json.dump(meta, open(sets_path, "w", encoding="utf-8"), indent=2)
        print(f"  -> {n} items, {mism} mismatches, {dt.datetime.now() - t0} -> {out_csv}", flush=True)

    return out_dir


def load_combined_all_a100(e3_items_dir: Path, rand_items_dir: Path, rel_a100_dir: Path, e2_items_dir: Path, manifest_path: Path):
    e3_meta = json.load(open(e3_items_dir / "sets.json", encoding="utf-8"))["sets"]
    e2_meta = json.load(open(e2_items_dir / "sets.json", encoding="utf-8"))["sets"]
    rel_a100_meta = json.load(open(rel_a100_dir / "sets.json", encoding="utf-8"))["sets"]
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    assert e3_meta["baseline"]["source_log_sha256"] == e2_meta["baseline"]["source_log_sha256"], (
        "Baseline source_log_sha256 mismatch between E3 and E2!"
    )

    meta = {}
    tables = {}
    source_files = [e3_items_dir / "sets.json", e2_items_dir / "sets.json", rel_a100_dir / "sets.json", manifest_path]

    # 1. E3 baseline + 6 real sets
    for sid, m in e3_meta.items():
        p = e3_items_dir / m["table"]
        meta[sid] = dict(m)
        meta[sid]["bbq_gpu"] = KNOWN_BBQ_GPU_A100.get(sid, m.get("summary_session_gpu") or "unknown")
        tables[sid] = r3b_mod.load_table(p)
        source_files.append(p)

    # 2. E3 Gender random sets: A100 seed0 + A100 seeds 1..4 + L4 seed0 sensitivity
    seed0_a100_dir = rand_items_dir.parent / "items_random_seed0_a100"
    assert (seed0_a100_dir / "sets.json").exists(), f"Missing {seed0_a100_dir / 'sets.json'}"
    s0_meta = json.load(open(seed0_a100_dir / "sets.json", encoding="utf-8"))["sets"]
    source_files.append(seed0_a100_dir / "sets.json")
    m = dict(s0_meta["rand_gender20_seed0"])
    assert r3b_mod.norm_neurons(m["neuron_indices"]) == r3b_mod.norm_neurons(manifest["random_sets"]["rand_gender20_seed0"]["by_layer"])
    p = seed0_a100_dir / m["table"]
    m["bbq_gpu"] = "A100-SXM4-80GB"
    meta["rand_gender20_seed0"] = m
    tables["rand_gender20_seed0"] = r3b_mod.load_table(p)
    source_files.append(p)

    r_meta = json.load(open(rand_items_dir / "sets.json", encoding="utf-8"))["sets"]
    source_files.append(rand_items_dir / "sets.json")
    for sid, m in r_meta.items():
        if sid == "rand_gender20_seed0":
            l4_csv = rand_items_dir / "rand_gender20_seed0_L4.csv.gz"
            if l4_csv.exists():
                m_l4 = dict(m)
                m_l4["set"] = "rand_gender20_seed0_L4"
                m_l4["kind"] = "random_sensitivity"
                m_l4["seed"] = 0
                m_l4["bbq_gpu"] = "L4"
                m_l4["table"] = l4_csv.name
                meta["rand_gender20_seed0_L4"] = m_l4
                tables["rand_gender20_seed0_L4"] = r3b_mod.load_table(l4_csv)
                source_files.append(l4_csv)
            continue
        if sid in manifest["random_sets"]:
            assert r3b_mod.norm_neurons(m["neuron_indices"]) == r3b_mod.norm_neurons(manifest["random_sets"][sid]["by_layer"])
        p = rand_items_dir / m["table"]
        meta[sid] = dict(m)
        meta[sid]["bbq_gpu"] = KNOWN_BBQ_GPU_A100.get(sid, m.get("summary_session_gpu") or "unknown")
        tables[sid] = r3b_mod.load_table(p)
        source_files.append(p)

    # 3. Religion random sets: NEW A100 seeds 0..1 + E2 A100 seeds 2..9 + L4 seeds 0..1 sensitivity
    for s in (0, 1):
        e3_sid = f"rand_religion20_seed{s}"
        e2_sid = f"rand_religion_seed{s}"
        m_a100 = dict(rel_a100_meta[e3_sid])
        assert r3b_mod.norm_neurons(m_a100["neuron_indices"]) == r3b_mod.norm_neurons(manifest["random_sets"][e3_sid]["by_layer"])
        m_a100["bbq_gpu"] = KNOWN_BBQ_GPU_A100[e3_sid]
        p_a100 = rel_a100_dir / m_a100["table"]
        meta[e3_sid] = m_a100
        tables[e3_sid] = r3b_mod.load_table(p_a100)
        source_files.append(p_a100)

        # Keep historical L4 seed0/seed1 as labelled sensitivity rows
        m_l4 = dict(e2_meta[e2_sid])
        l4_sid = f"rand_religion20_seed{s}_L4"
        m_l4["set"] = l4_sid
        m_l4["e2_original_set"] = e2_sid
        m_l4["kind"] = "random_sensitivity"
        m_l4["category"] = "Religion"
        m_l4["k"] = 20
        m_l4["mimics"] = "real_religion20"
        m_l4["seed"] = s
        m_l4["bbq_gpu"] = "L4"
        p_l4 = e2_items_dir / m_l4["table"]
        meta[l4_sid] = m_l4
        tables[l4_sid] = r3b_mod.load_table(p_l4)
        source_files.append(p_l4)

    for s in range(2, 10):
        e2_sid = f"rand_religion_seed{s}"
        e3_sid = f"rand_religion20_seed{s}"
        m = dict(e2_meta[e2_sid])
        if s < 5:
            assert r3b_mod.norm_neurons(m["neuron_indices"]) == r3b_mod.norm_neurons(manifest["random_sets"][e3_sid]["by_layer"])
        m["set"] = e3_sid
        m["e2_original_set"] = e2_sid
        m["category"] = "Religion"
        m["k"] = 20
        m["mimics"] = "real_religion20"
        m["seed"] = s
        m["bbq_gpu"] = KNOWN_BBQ_GPU_A100[e3_sid]
        p = e2_items_dir / m["table"]
        meta[e3_sid] = m
        tables[e3_sid] = r3b_mod.load_table(p)
        source_files.append(p)

    # Assert exact doc_id and item metadata alignment across all sets
    base = "baseline"
    for sid, t in tables.items():
        assert t.index.equals(tables[base].index), f"{sid}: items differ from baseline"
        cols = ["category", "context", "group0", "group1", "target"]
        bad = [c for c in cols if not (t[c] == tables[base][c]).all()]
        assert not bad, f"{sid}: item metadata differs from baseline in {bad}"

    order = (
        [base]
        + sorted([s for s in meta if meta[s]["kind"] == "real"], key=lambda s: (-int(meta[s]["k"] or 0), s))
        + sorted([s for s in meta if meta[s]["kind"] == "random"], key=lambda s: (s.rsplit("_seed", 1)[0], int(meta[s]["seed"])))
        + sorted([s for s in meta if meta[s]["kind"] == "random_sensitivity"])
    )
    return meta, tables, base, order, source_files


def paired_comparisons_a100(spec_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Religion·20 (seeds 0-4, A100-only)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(5)]),
        ("Religion·20 (seeds 0-9, A100-only, primary)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(10)]),
        ("Religion·20 (L4 seeds 0-1 + A100 seeds 2-9, sensitivity)", "real_religion20", ["rand_religion20_seed0_L4", "rand_religion20_seed1_L4"] + [f"rand_religion20_seed{i}" for i in range(2, 10)]),
        ("Gender·20 (seeds 0-4, A100-only, primary)", "real_gender20", [f"rand_gender20_seed{i}" for i in range(5)]),
        ("Gender·20 (L4 seed0 + A100 seeds 1-4, sensitivity)", "real_gender20", ["rand_gender20_seed0_L4"] + [f"rand_gender20_seed{i}" for i in range(1, 5)]),
    ]
    for label, real_sid, rand_sids in specs:
        r_row = spec_df[spec_df["set"] == real_sid]
        rnd = spec_df[spec_df["set"].isin(rand_sids)]
        if r_row.empty or rnd.empty:
            continue
        r = r_row.iloc[0]
        gpus = sorted(set(rnd["bbq_gpu"].tolist()))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--out-dir", help="Optional explicit output directory (must not exist)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    drive = Path(a.drive_root)
    unit_dir = drive / "results/e3/llama-3.2-3b_en"
    manifest_path = drive / "results/e3/llama-3.2-3b_en_E3_manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    rel_a100_dir = extract_religion_a100_seed01(drive, manifest)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) if a.out_dir else unit_dir / "analysis" / f"{stamp}_items_3b_random_a100"
    assert not out.exists(), f"Refusing to overwrite existing output dir: {out}"
    out.mkdir(parents=True, exist_ok=False)

    meta, tables, base, order, source_files = load_combined_all_a100(
        unit_dir / "items", unit_dir / "items_random", rel_a100_dir, drive / "results/e2/items", manifest_path
    )
    print(f"Loaded {len(tables)} sets ({base} + {len(order) - 1}); writing to {out}", flush=True)

    flips_df, spec_df = r3b_mod.flips_and_specificity(meta, tables, base, order)
    paired_df = paired_comparisons_a100(spec_df)

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

    R = [
        "# E3 Per-Item Analysis — `llama-3.2-3b_en` (All-A100 Real + Random Sets)",
        "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')}. Baseline: `baseline` (`A100-SXM4-40GB`).",
        "All primary random sets (`Gender·20 seeds 0-4`, `Religion·20 seeds 0-9`) run on A100; historical L4 runs are retained only as `*_L4` sensitivity rows.",
        "",
        "## 1. Specificity Summary",
        "",
        r3b_mod.md_table(spec_df.set_index("set")),
        "",
        "## 2. Paired Comparisons",
        "",
        r3b_mod.md_table(paired_df.set_index("comparison")),
        "",
        "## 3. Flips (%) by BBQ Category",
        "",
        r3b_mod.md_table(flips_df.pivot(index="set", columns="category", values="flips_pct").reindex([s for s in order if s != base])[CATS]),
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
    print(f"ANALYSIS_DIR={out}", flush=True)


if __name__ == "__main__":
    main()
