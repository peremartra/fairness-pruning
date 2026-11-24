#!/usr/bin/env python3
"""
Consolidate E3 experiment outputs from Google Drive into compact, verified CSV tables.

Read-only with respect to Google Drive (never modifies or overwrites anything on Drive).
Reads the latest verified per-item analysis directories of all E3 units (BBQ EN and EsBBQ ES),
plus the Llama-3.2-1B EN summary-only unit, the Llama-3.1-8B L31 weight check, and the K3 GPU
noise comparison, and writes compact CSVs + an inputs_manifest.json to --out-dir.

Primary measure: changed answers against the unpruned baseline (`changed`, `n_items`, `flips_pct`
over 3-way predictions {person0, person1, unknown}, pooled across ambig + disambig items per
category; ambig and disambig breakdowns are also preserved). BBQ/EsBBQ bias scores and accuracy
transitions are exported separately as secondary metrics.

Usage (Colab CPU with Drive mounted):
    python analysis/e3/build_e3_tables.py \
        --drive-root /content/drive/MyDrive/fair_pruning \
        --out-dir /content/e3_tables
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BBQ_CATS = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]

BBQ_ITEM_COUNTS = {
    "Age": 3680,
    "Disability_status": 1556,
    "Gender_identity": 5672,
    "Nationality": 3080,
    "Physical_appearance": 1576,
    "Race_ethnicity": 6880,
    "Race_x_SES": 11160,
    "Race_x_gender": 15960,
    "Religion": 1200,
    "SES": 6864,
    "Sexual_orientation": 864,
}

SET_CAT_TO_BBQ = {
    "Religion": "Religion",
    "Race": "Race_ethnicity",
    "RaceEthnicity": "Race_ethnicity",
    "Gender": "Gender_identity",
}

SET_CAT_TO_ESBBQ = {
    "Religion": "Religion",
    "Race": "RaceEthnicity",
    "RaceEthnicity": "RaceEthnicity",
    "Gender": "Gender",
    "Age": "Age",
}

# Pinned latest verified analysis directories on Drive (relative to --drive-root)
UNITS_CONFIG = [
    {
        "unit": "llama-3.1-8b_en",
        "model": "Llama-3.1-8B",
        "language": "EN",
        "benchmark": "BBQ",
        "mode": "bbq_items",
        "analysis_dir": "results/e3/llama-3.1-8b_en/analysis/20261108_144541_items",
        "target_categories": ["Religion", "Race_ethnicity", "Gender_identity"],
        "last_layer": 31,
    },
    {
        "unit": "salamandra-7b_en",
        "model": "Salamandra-7B",
        "language": "EN",
        "benchmark": "BBQ",
        "mode": "bbq_items",
        "analysis_dir": "results/e3/salamandra-7b_en/analysis/20261107_153940_items",
        "target_categories": ["Religion", "Race_ethnicity", "Gender_identity"],
        "last_layer": 31,
    },
    {
        "unit": "qwen3-4b-base_en",
        "model": "Qwen3-4B-Base",
        "language": "EN",
        "benchmark": "BBQ",
        "mode": "bbq_items",
        "analysis_dir": "results/e3/qwen3-4b-base_en/analysis/20261107_153055_items",
        "target_categories": ["Religion", "Race_ethnicity", "Gender_identity"],
        "last_layer": 35,
    },
    {
        "unit": "llama-3.2-3b_en",
        "model": "Llama-3.2-3B",
        "language": "EN",
        "benchmark": "BBQ",
        "mode": "bbq_items_3b_random",
        "analysis_dir": "results/e3/llama-3.2-3b_en/analysis/20261124_a100_items_3b_random",
        "target_categories": ["Religion", "Race_ethnicity", "Gender_identity"],
        "last_layer": 27,
    },
    {
        "unit": "llama-3.2-1b_en",
        "model": "Llama-3.2-1B",
        "language": "EN",
        "benchmark": "BBQ",
        "mode": "bbq_per_set_summary",
        "per_set_dir": "results/e3/llama-3.2-1b_en/per_set",
        "target_categories": ["Religion", "Race_ethnicity", "Gender_identity"],
        "last_layer": 15,
    },
    {
        "unit": "salamandra-7b_es",
        "model": "Salamandra-7B",
        "language": "ES",
        "benchmark": "EsBBQ",
        "mode": "esbbq_items",
        "analysis_dir": "results/esbbq_zeroed_7b/analysis/20261111_095116_items",
        "runs_jsonl": "results/esbbq_zeroed_7b/runs.jsonl",
        "target_categories": ["Religion", "Age", "Gender", "RaceEthnicity"],
        "last_layer": 31,
    },
    {
        "unit": "llama-3.1-8b_es",
        "model": "Llama-3.1-8B",
        "language": "ES",
        "benchmark": "EsBBQ",
        "mode": "esbbq_items",
        "analysis_dir": "results/esbbq_zeroed_llama8b/analysis/20261124_141737_items_a100base",
        "runs_jsonl": "results/esbbq_zeroed_llama8b/runs.jsonl",
        "target_categories": ["Religion", "Gender", "RaceEthnicity"],
        "last_layer": 31,
    },
    {
        "unit": "llama-3.2-3b_es",
        "model": "Llama-3.2-3B",
        "language": "ES",
        "benchmark": "EsBBQ",
        "mode": "esbbq_items",
        "analysis_dir": "results/esbbq_zeroed_llama3b/analysis/20261112_081707_items",
        "runs_jsonl": "results/esbbq_zeroed_llama3b/runs.jsonl",
        "target_categories": ["Religion", "Gender", "RaceEthnicity"],
        "last_layer": 27,
    },
]

L31_WEIGHTS_REL = "results/e3/llama-3.1-8b_en/analysis/20261108_151339_l31_weights/neurons.csv"
K3_GPU_NOISE_REL = "checkpoints_mbbq_a100check/k3_gpu_noise_l4_vs_a100.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path: Path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def track_file(tracked: list, path: Path, drive_root: Path):
    rel = str(path.relative_to(drive_root)) if path.is_relative_to(drive_root) else str(path)
    tracked.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})


def neuron_set_from_dict(neuron_indices) -> set:
    out = set()
    if not neuron_indices:
        return out
    for layer, idxs in neuron_indices.items():
        for idx in idxs:
            out.add((int(layer), int(idx)))
    return out


def layer_profile_str(neuron_indices) -> str:
    if not neuron_indices:
        return ""
    parts = [f"L{int(k)}:{len(v)}" for k, v in sorted(neuron_indices.items(), key=lambda kv: int(kv[0]))]
    return ", ".join(parts)


def load_runs_gpu(path: Path, tracked: list, drive_root: Path) -> dict:
    gpus = {}
    if not path.exists():
        return gpus
    track_file(tracked, path, drive_root)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        eid = d.get("exp_id")
        gpu = (d.get("env") or {}).get("gpu")
        if eid and gpu:
            gpus[eid] = gpu
    return gpus


def normalize_gpu(g):
    if not g:
        return "unknown"
    return str(g).replace("NVIDIA ", "").strip()


def map_target_category(benchmark: str, set_category: str, set_id: str):
    if set_category:
        if benchmark == "BBQ":
            return SET_CAT_TO_BBQ.get(set_category, set_category)
        return SET_CAT_TO_ESBBQ.get(set_category, set_category)
    # Fallback for ablation sets in llama-3.1-8b_en
    if "race" in set_id:
        return "Race_ethnicity" if benchmark == "BBQ" else "RaceEthnicity"
    if "religion" in set_id:
        return "Religion"
    if "shared" in set_id:
        return "Religion"
    return None


def parse_ablation_set_category(set_id: str, raw_cat):
    if raw_cat:
        return raw_cat
    if set_id == "abl_shared1":
        return "Race+Religion (shared L31.6950)"
    if set_id == "abl_race4":
        return "Race (without L31.6950)"
    if set_id == "abl_religion4":
        return "Religion (without L31.6950)"
    return "Ablation"


def compute_rank_and_ratio(cat_flips: dict, target_cat: str):
    if not target_cat or target_cat not in cat_flips:
        return np.nan, np.nan, np.nan
    tgt = float(cat_flips[target_cat])
    others = [float(v) for k, v in cat_flips.items() if k != target_cat]
    other_mean = float(np.mean(others)) if others else np.nan
    ratio = tgt / other_mean if (other_mean and other_mean > 0) else np.nan
    rank = 1 + sum(1 for v in cat_flips.values() if float(v) > tgt + 1e-12)
    return other_mean, ratio, int(rank)


def process_items_unit(cfg: dict, drive: Path, tracked: list, accum: dict):
    unit = cfg["unit"]
    model = cfg["model"]
    lang = cfg["language"]
    bench = cfg["benchmark"]
    adir = drive / cfg["analysis_dir"]
    assert adir.exists(), f"Missing analysis dir: {adir}"

    inp_path = adir / "inputs.json"
    dir_path = adir / "direction.csv"
    bs_path = adir / "bootstrap.csv"
    med_path = adir / "margin_changed_vs_unchanged.csv"
    dec_path = adir / "margin_deciles.csv"
    grp_path = adir / "groups.csv"
    ov_all_path = adir / "overlap_changed_items_all.csv"
    top_path = adir / "topics.csv"

    for p in (inp_path, dir_path, bs_path, med_path, dec_path, grp_path, ov_all_path):
        assert p.exists(), f"Missing required file: {p}"
        track_file(tracked, p, drive)

    inp = json.load(open(inp_path, encoding="utf-8"))
    sets_meta = inp["sets"]
    dir_df = pd.read_csv(dir_path)
    bs_df = pd.read_csv(bs_path)
    med_df = pd.read_csv(med_path)
    dec_df = pd.read_csv(dec_path)
    grp_df = pd.read_csv(grp_path)
    ov_all_df = pd.read_csv(ov_all_path, index_col=0)

    ov_cat_dfs = {}
    for tc in cfg["target_categories"]:
        p = adir / f"overlap_changed_items_{tc}.csv"
        if p.exists():
            track_file(tracked, p, drive)
            ov_cat_dfs[tc] = pd.read_csv(p, index_col=0)

    top_df = None
    if top_path.exists():
        track_file(tracked, top_path, drive)
        top_df = pd.read_csv(top_path)

    es_runs_gpu = {}
    if cfg.get("runs_jsonl"):
        es_runs_gpu = load_runs_gpu(drive / cfg["runs_jsonl"], tracked, drive)

    # Helper to get set GPU
    def get_gpu(sid):
        m = sets_meta[sid]
        g = m.get("bbq_gpu") or m.get("summary_session_gpu") or es_runs_gpu.get(sid)
        if not g and sid == "baseline" and bench == "EsBBQ":
            # Documented in R1_exp.md: Llama-3.1-8B ES baseline = L4 (E1 run 2), Salamandra-7B ES = A100-80GB (E1 run 3), Llama-3.2-3B ES = L4
            if unit == "salamandra-7b_es":
                return "A100-SXM4-80GB (E1)"
            if unit == "llama-3.1-8b_es":
                return "A100-SXM4-40GB (K3)"
            if unit == "llama-3.2-3b_es":
                return "L4 (rerun)"
        return normalize_gpu(g)

    # All categories present
    all_cats = sorted(dir_df["category"].unique().tolist())
    n_cats = len(all_cats)
    n_items_total = int(dir_df[(dir_df["context"] == "all")].drop_duplicates("category")["n_items"].sum())

    # Inventory row
    real_sids = [s for s, m in sets_meta.items() if m.get("kind") == "real"]
    abl_sids = [s for s, m in sets_meta.items() if m.get("kind") == "ablation"]
    rnd_sids = [s for s, m in sets_meta.items() if m.get("kind") == "random"]
    sens_sids = [s for s, m in sets_meta.items() if m.get("kind") == "random_sensitivity"]
    ks = sorted({int(sets_meta[s]["k"]) for s in real_sids if sets_meta[s].get("k") is not None})
    real_cats = sorted({str(sets_meta[s].get("category")) for s in real_sids if sets_meta[s].get("category")})
    gpus_used = sorted({get_gpu(s) for s, m in sets_meta.items() if s != "baseline" and m.get("kind") != "random_sensitivity"})

    accum["inventory"].append({
        "unit": unit,
        "model": model,
        "language": lang,
        "benchmark": bench,
        "n_items_total": n_items_total,
        "n_categories": n_cats,
        "n_real_sets": len(real_sids),
        "n_ablation_sets": len(abl_sids),
        "n_random_sets": len(rnd_sids),
        "n_random_sensitivity_sets": len(sens_sids),
        "k_values": ", ".join(str(k) for k in ks),
        "target_categories_tested": ", ".join(real_cats),
        "has_per_item_analysis": True,
        "analysis_dir": cfg["analysis_dir"],
        "gpus_intervened": "; ".join(gpus_used),
        "baseline_gpu": get_gpu("baseline"),
    })

    # Provenance rows
    for sid, m in sets_meta.items():
        n_idx = m.get("neuron_indices") or {}
        n_neurons = int(m.get("n_neurons") if m.get("n_neurons") is not None else sum(len(v) for v in n_idx.values()))
        last_cnt = len(n_idx.get(str(cfg["last_layer"]), n_idx.get(cfg["last_layer"], [])))
        accum["provenance"].append({
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "set": sid,
            "kind": m.get("kind"),
            "set_category": m.get("category"),
            "k": m.get("k"),
            "seed": m.get("seed"),
            "mimics_or_matched_to": m.get("mimics") or m.get("matched_to"),
            "n_neurons": n_neurons,
            "n_layers": len(n_idx),
            "last_layer": cfg["last_layer"],
            "last_layer_neurons": last_cnt,
            "layer_profile": layer_profile_str(n_idx),
            "gpu": get_gpu(sid),
            "manifest_sha256": m.get("manifest_sha256"),
            "analysis_dir": cfg["analysis_dir"],
        })

    # Canonical 3-way pred != base_pred counts for context="all" (consistent with E2,
    # e3_item_analysis_3b_random.py, bootstrap.csv, margin_changed_vs_unchanged.csv, and EsBBQ changed_by_category.csv).
    # Note: direction.csv counts S/A/U state changes, which undercounts person0<->person1 flips when both are non-stereotyped.
    med_cat_idx = med_df[med_df["category"] != "ALL"].set_index(["set", "category"])["n_changed"]
    med_all_idx = med_df[med_df["category"] == "ALL"].set_index("set")["n_changed"]
    bs_idx = bs_df.set_index(["set", "category"])

    # Cross-check pred != base_pred consistency against bootstrap.csv and changed_by_category.csv / flips_by_category.csv
    chg_cat_df = pd.read_csv(adir / "changed_by_category.csv") if (adir / "changed_by_category.csv").exists() else None
    flips_3b_df = pd.read_csv(adir / "flips_by_category.csv") if (adir / "flips_by_category.csv").exists() else None
    if chg_cat_df is not None:
        track_file(tracked, adir / "changed_by_category.csv", drive)
    if flips_3b_df is not None:
        track_file(tracked, adir / "flips_by_category.csv", drive)

    cat_n_map = dir_df[dir_df["context"] == "all"].drop_duplicates("category").set_index("category")["n_items"].to_dict()

    # Full flips_by_category: context="all" uses canonical pred != base_pred; ambig/disambig come from direction.csv (S/A/U)
    for _, row in dir_df.iterrows():
        sid = row["set"]
        cat = row["category"]
        ctx = row["context"]
        m = sets_meta[sid]
        n_items = int(row["n_items"])
        if ctx == "all":
            ch_pred = int(med_cat_idx.loc[(sid, cat)])
            fp_pred = 100.0 * ch_pred / n_items if n_items else np.nan
            if (sid, cat) in bs_idx.index:
                bs_fp = float(bs_idx.loc[(sid, cat), "changed_pct"])
                assert abs(fp_pred - bs_fp) < 1e-6, f"{unit} {sid} {cat}: margin vs bootstrap mismatch ({fp_pred} vs {bs_fp})"
            changed_val, flips_val = ch_pred, fp_pred
        else:
            changed_val, flips_val = int(row["changed"]), float(row["changed_pct"])
        accum["flips_by_category"].append({
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "set": sid,
            "kind": row["kind"],
            "set_category": parse_ablation_set_category(sid, m.get("category")) if row["kind"] == "ablation" else m.get("category"),
            "k": m.get("k"),
            "gpu": get_gpu(sid),
            "category": cat,
            "context": ctx,
            "n_items": n_items,
            "changed": changed_val,
            "flips_pct": flips_val,
        })

    # Pre-index direction (disambig) for secondary disambig-only breakdown
    dir_dis = dir_df[dir_df["context"] == "disambig"]

    # Compute per-set localization metrics using canonical pred != base_pred for context="all"
    per_set_spec = {}
    for sid, m in sets_meta.items():
        if sid == "baseline":
            continue
        kind = m.get("kind")
        raw_cat = m.get("category")
        tgt_cat = map_target_category(bench, raw_cat, sid)
        sub_dis = dir_dis[dir_dis["set"] == sid]

        cat_changed_all = {c: int(med_cat_idx.loc[(sid, c)]) for c in all_cats}
        cat_n_all = {c: int(cat_n_map[c]) for c in all_cats}
        cat_flips_all = {c: 100.0 * cat_changed_all[c] / cat_n_all[c] for c in all_cats}
        cat_flips_dis = dict(zip(sub_dis["category"], sub_dis["changed_pct"]))

        # Verify sum of per-category pred flips equals ALL pred flips in margin_changed_vs_unchanged.csv
        all_changed = int(sum(cat_changed_all.values()))
        assert all_changed == int(med_all_idx.loc[sid]), f"{unit} {sid}: sum(cat) {all_changed} != ALL {med_all_idx.loc[sid]}"

        other_mean_all, ratio_all, rank_all = compute_rank_and_ratio(cat_flips_all, tgt_cat)
        other_mean_dis, ratio_dis, rank_dis = compute_rank_and_ratio(cat_flips_dis, tgt_cat)

        tgt_n = int(cat_n_all[tgt_cat]) if tgt_cat in cat_n_all else np.nan
        tgt_ch = int(cat_changed_all[tgt_cat]) if tgt_cat in cat_changed_all else np.nan
        tgt_pct = float(cat_flips_all[tgt_cat]) if tgt_cat in cat_flips_all else np.nan
        tgt_dis_pct = float(cat_flips_dis[tgt_cat]) if tgt_cat in cat_flips_dis else np.nan

        ci_lo = ci_hi = np.nan
        if (sid, tgt_cat) in bs_idx.index:
            brow = bs_idx.loc[(sid, tgt_cat)]
            ci_lo = float(brow["changed_pct_ci_low"])
            ci_hi = float(brow["changed_pct_ci_high"])

        all_total = int(sum(cat_n_all.values()))
        all_pct = 100.0 * all_changed / all_total if all_total else np.nan

        n_idx = m.get("neuron_indices") or {}
        n_neurons = int(m.get("n_neurons") if m.get("n_neurons") is not None else sum(len(v) for v in n_idx.values()))
        last_cnt = len(n_idx.get(str(cfg["last_layer"]), n_idx.get(cfg["last_layer"], [])))

        rec = {
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "set": sid,
            "kind": kind,
            "set_category": parse_ablation_set_category(sid, raw_cat) if kind == "ablation" else raw_cat,
            "target_category": tgt_cat,
            "k": m.get("k") if m.get("k") is not None else n_neurons,
            "n_neurons": n_neurons,
            "n_layers": len(n_idx),
            "last_layer_neurons": last_cnt,
            "gpu": get_gpu(sid),
            "target_n_items": tgt_n,
            "target_changed": tgt_ch,
            "target_flips_pct": tgt_pct,
            "target_flips_ci_low": ci_lo,
            "target_flips_ci_high": ci_hi,
            "other_mean_flips_pct": other_mean_all,
            "specificity_ratio": ratio_all,
            "target_rank": rank_all,
            "n_categories": n_cats,
            "all_items_changed": all_changed,
            "all_items_total": all_total,
            "all_items_flips_pct": all_pct,
            "disambig_target_flips_pct": tgt_dis_pct,
            "disambig_other_mean_flips_pct": other_mean_dis,
            "disambig_specificity_ratio": ratio_dis,
            "disambig_target_rank": rank_dis,
        }
        per_set_spec[sid] = rec
        if kind in ("real", "ablation"):
            accum["localization"].append(rec)

    # Paired Real vs Random comparisons
    def add_rvr(label: str, real_sid: str, rand_list: list, is_primary: bool):
        if real_sid not in per_set_spec or not rand_list:
            return
        r = per_set_spec[real_sid]
        rs = [per_set_spec[s] for s in rand_list if s in per_set_spec]
        if not rs:
            return
        r_flips = np.array([x["target_flips_pct"] for x in rs], dtype=float)
        r_ch = np.array([x["target_changed"] for x in rs], dtype=float)
        r_oth = np.array([x["other_mean_flips_pct"] for x in rs], dtype=float)
        r_spec = np.array([x["specificity_ratio"] for x in rs], dtype=float)
        r_all = np.array([x["all_items_flips_pct"] for x in rs], dtype=float)
        rank_vs_rand = 1 + int((r_flips >= r["target_flips_pct"] - 1e-12).sum())
        rand_gpus = sorted({x["gpu"] for x in rs})
        accum["real_vs_random"].append({
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "comparison": label,
            "is_primary": is_primary,
            "real_set": real_sid,
            "target_category": r["target_category"],
            "k": r["k"],
            "target_n_items": r["target_n_items"],
            "real_target_changed": r["target_changed"],
            "real_target_flips_pct": r["target_flips_pct"],
            "n_random": len(rs),
            "rand_target_changed_min": int(r_ch.min()),
            "rand_target_changed_max": int(r_ch.max()),
            "rand_target_changed_mean": float(r_ch.mean()),
            "rand_target_flips_min": float(r_flips.min()),
            "rand_target_flips_max": float(r_flips.max()),
            "rand_target_flips_mean": float(r_flips.mean()),
            "ratio_real_over_rand_mean": float(r["target_flips_pct"] / r_flips.mean()) if r_flips.mean() > 0 else np.nan,
            "rank_of_real": f"{rank_vs_rand} of {len(rs) + 1}",
            "real_other_mean_flips_pct": r["other_mean_flips_pct"],
            "rand_other_mean_flips_pct": float(r_oth.mean()),
            "real_specificity_ratio": r["specificity_ratio"],
            "rand_specificity_ratio_mean": float(r_spec.mean()),
            "real_all_items_flips_pct": r["all_items_flips_pct"],
            "rand_all_items_flips_mean": float(r_all.mean()),
            "real_gpu": r["gpu"],
            "random_gpus": "; ".join(rand_gpus),
        })

    if rnd_sids or sens_sids:
        if unit == "llama-3.2-3b_en":
            add_rvr("Gender·20 (seeds 0-4, A100-only)", "real_gender20", [f"rand_gender20_seed{i}" for i in range(5)], True)
            add_rvr("Religion·20 (seeds 0-9, A100-only)", "real_religion20", [f"rand_religion20_seed{i}" for i in range(10)], True)
            add_rvr("Religion·20 (L4 seeds 0-1 + A100 seeds 2-9, sensitivity)", "real_religion20", ["rand_religion20_seed0_L4", "rand_religion20_seed1_L4"] + [f"rand_religion20_seed{i}" for i in range(2, 10)], False)
            add_rvr("Gender·20 (L4 seed0 + A100 seeds 1-4, sensitivity)", "real_gender20", ["rand_gender20_seed0_L4"] + [f"rand_gender20_seed{i}" for i in range(1, 5)], False)
        elif bench == "EsBBQ":
            by_matched = {}
            for s in rnd_sids:
                mt = sets_meta[s].get("matched_to")
                if mt:
                    by_matched.setdefault(mt, []).append(s)
            for real_sid, rs in sorted(by_matched.items()):
                rs_sorted = sorted(rs, key=lambda x: int(sets_meta[x].get("seed", 0)))
                cat = sets_meta[real_sid].get("category")
                k = sets_meta[real_sid].get("k")
                add_rvr(f"{cat}·{k} (seeds 0-{len(rs_sorted)-1})", real_sid, rs_sorted, True)
        else:
            by_mimics = {}
            for s in rnd_sids:
                mt = sets_meta[s].get("mimics")
                if not mt:
                    fam = s.rsplit("_seed", 1)[0]
                    mt = fam.replace("rand_", "real_", 1)
                by_mimics.setdefault(mt, []).append(s)
            for real_sid, rs in sorted(by_mimics.items()):
                rs_sorted = sorted(rs, key=lambda x: int(sets_meta[x].get("seed", 0)))
                cat = sets_meta[real_sid].get("category")
                k = sets_meta[real_sid].get("k")
                add_rvr(f"{cat}·{k} (seeds 0-{len(rs_sorted)-1})", real_sid, rs_sorted, True)

    # Neuron + changed-item overlap among non-random sets
    nonrand = [s for s in sets_meta if sets_meta[s].get("kind") in ("real", "ablation")]
    nonrand = sorted(nonrand, key=lambda s: (0 if sets_meta[s]["kind"] == "real" else 1, -(sets_meta[s].get("k") or 0), s))
    nsets = {s: neuron_set_from_dict(sets_meta[s].get("neuron_indices")) for s in nonrand}
    for i, a in enumerate(nonrand):
        for b in nonrand[i + 1:]:
            na, nb = nsets[a], nsets[b]
            inter = sorted(na & nb)
            union = na | nb
            n_jacc = len(inter) / len(union) if union else np.nan
            ij_all = float(ov_all_df.loc[a, b]) if (a in ov_all_df.index and b in ov_all_df.columns) else np.nan
            tc_a = per_set_spec[a]["target_category"]
            tc_b = per_set_spec[b]["target_category"]
            ij_ta = float(ov_cat_dfs[tc_a].loc[a, b]) if (tc_a in ov_cat_dfs and a in ov_cat_dfs[tc_a].index and b in ov_cat_dfs[tc_a].columns) else np.nan
            ij_tb = float(ov_cat_dfs[tc_b].loc[a, b]) if (tc_b in ov_cat_dfs and a in ov_cat_dfs[tc_b].index and b in ov_cat_dfs[tc_b].columns) else np.nan
            accum["overlap"].append({
                "unit": unit,
                "model": model,
                "language": lang,
                "benchmark": bench,
                "set_a": a,
                "set_b": b,
                "n_neurons_a": len(na),
                "n_neurons_b": len(nb),
                "n_shared_neurons": len(inter),
                "neuron_jaccard": n_jacc,
                "items_jaccard_all": ij_all,
                "target_cat_a": tc_a,
                "items_jaccard_target_a": ij_ta,
                "target_cat_b": tc_b,
                "items_jaccard_target_b": ij_tb,
                "shared_neurons": ", ".join(f"L{l}.{idx}" for l, idx in inter[:15]) + (" ..." if len(inter) > 15 else ""),
            })

    # Secondary metrics (for target categories, non-random sets + random-family summaries)
    t_cats = set(cfg["target_categories"])
    dir_t = dir_df[dir_df["category"].isin(t_cats)]
    for _, row in dir_t[dir_t["kind"].isin(["real", "ablation"])].iterrows():
        sid, cat, ctx = row["set"], row["category"], row["context"]
        brow = bs_idx.loc[(sid, cat)] if (sid, cat) in bs_idx.index else None
        rec = {
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "set": sid,
            "kind": row["kind"],
            "category": cat,
            "context": ctx,
            "n_items": int(row["n_items"]),
            "changed": int(row["changed"]),
            "changed_pct": float(row["changed_pct"]),
            "to_stereotyped": int(row["to_stereotyped"]),
            "to_other_person": int(row["to_other_person"]),
            "to_unknown": int(row["to_unknown"]),
            "S_to_A": int(row["S_to_A"]),
            "A_to_S": int(row["A_to_S"]),
            "net_towards_stereotype_pct": float(row["net_towards_stereotype_pct"]),
            "correct_to_incorrect": int(row["correct_to_incorrect"]),
            "incorrect_to_correct": int(row["incorrect_to_correct"]),
            "disamb_bias_delta": float(brow["disamb_bias_delta"]) if (brow is not None and ctx == "disambig") else np.nan,
            "disamb_bias_delta_ci_low": float(brow["disamb_bias_delta_ci_low"]) if (brow is not None and ctx == "disambig") else np.nan,
            "disamb_bias_delta_ci_high": float(brow["disamb_bias_delta_ci_high"]) if (brow is not None and ctx == "disambig") else np.nan,
            "esbbq_disamb_bias_delta": float(brow["esbbq_disamb_bias_delta"]) if (brow is not None and ctx == "disambig" and "esbbq_disamb_bias_delta" in brow) else np.nan,
            "esbbq_disamb_bias_delta_ci_low": float(brow["esbbq_disamb_bias_delta_ci_low"]) if (brow is not None and ctx == "disambig" and "esbbq_disamb_bias_delta_ci_low" in brow) else np.nan,
            "esbbq_disamb_bias_delta_ci_high": float(brow["esbbq_disamb_bias_delta_ci_high"]) if (brow is not None and ctx == "disambig" and "esbbq_disamb_bias_delta_ci_high" in brow) else np.nan,
            "esbbq_amb_bias_delta": float(brow["esbbq_amb_bias_delta"]) if (brow is not None and ctx == "ambig" and "esbbq_amb_bias_delta" in brow) else np.nan,
            "esbbq_amb_bias_delta_ci_low": float(brow["esbbq_amb_bias_delta_ci_low"]) if (brow is not None and ctx == "ambig" and "esbbq_amb_bias_delta_ci_low" in brow) else np.nan,
            "esbbq_amb_bias_delta_ci_high": float(brow["esbbq_amb_bias_delta_ci_high"]) if (brow is not None and ctx == "ambig" and "esbbq_amb_bias_delta_ci_high" in brow) else np.nan,
        }
        accum["secondary"].append(rec)

    # Margin medians (ALL + target categories, non-random sets + random-family mean)
    med_keep = med_df[med_df["category"].isin(t_cats | {"ALL"})]
    for _, row in med_keep[med_keep["kind"].isin(["real", "ablation"])].iterrows():
        accum["margin_summary"].append({
            "unit": unit,
            "set": row["set"],
            "kind": row["kind"],
            "category": row["category"],
            "n_changed": int(row["n_changed"]),
            "median_margin_changed": float(row["median_margin_changed"]) if pd.notna(row["median_margin_changed"]) else np.nan,
            "median_margin_unchanged": float(row["median_margin_unchanged"]) if pd.notna(row["median_margin_unchanged"]) else np.nan,
        })
    med_rnd = med_keep[med_keep["kind"] == "random"].copy()
    if not med_rnd.empty:
        med_rnd["family"] = med_rnd["set"].str.rsplit("_seed", n=1).str[0]
        for (fam, cat), g in med_rnd.groupby(["family", "category"]):
            accum["margin_summary"].append({
                "unit": unit,
                "set": f"{fam} (mean n={len(g)})",
                "kind": "random_family_mean",
                "category": cat,
                "n_changed": float(g["n_changed"].mean()),
                "median_margin_changed": float(g["median_margin_changed"].mean()),
                "median_margin_unchanged": float(g["median_margin_unchanged"].mean()),
            })

    # Margin deciles (non-random sets + random-family mean & max)
    for _, row in dec_df[dec_df["kind"].isin(["real", "ablation"])].iterrows():
        accum["margin_deciles"].append({
            "unit": unit,
            "set": row["set"],
            "kind": row["kind"],
            "decile": int(row["decile"]),
            "margin_min": float(row["margin_min"]),
            "margin_max": float(row["margin_max"]),
            "n_items": int(row["n_items"]),
            "changed_pct": float(row["changed_pct"]),
        })
    dec_rnd = dec_df[dec_df["kind"] == "random"].copy()
    if not dec_rnd.empty:
        dec_rnd["family"] = dec_rnd["set"].str.rsplit("_seed", n=1).str[0]
        for (fam, dec), g in dec_rnd.groupby(["family", "decile"]):
            accum["margin_deciles"].append({
                "unit": unit,
                "set": f"{fam} (mean n={len(g)})",
                "kind": "random_family_mean",
                "decile": int(dec),
                "margin_min": float(g["margin_min"].iloc[0]),
                "margin_max": float(g["margin_max"].iloc[0]),
                "n_items": int(g["n_items"].iloc[0]),
                "changed_pct": float(g["changed_pct"].mean()),
            })

    # Groups (target categories, non-random sets + random-family min/max)
    grp_t = grp_df[grp_df["category"].isin(t_cats)]
    for _, row in grp_t[grp_t["kind"].isin(["real", "ablation"])].iterrows():
        accum["groups"].append({
            "unit": unit,
            "set": row["set"],
            "kind": row["kind"],
            "category": row["category"],
            "group": str(row["group"]),
            "n_items": int(row["n_items"]),
            "changed": int(row["changed"]),
            "changed_pct": float(row["changed_pct"]),
        })

    # EsBBQ Topics (target categories, non-random sets)
    if top_df is not None:
        top_t = top_df[top_df["category"].isin(t_cats)]
        for _, row in top_t[top_t["kind"].isin(["real", "ablation"])].iterrows():
            accum["topics"].append({
                "unit": unit,
                "set": row["set"],
                "kind": row["kind"],
                "category": row["category"],
                "topic": str(row["topic"]),
                "n_items": int(row["n_items"]),
                "changed": int(row["changed"]),
                "changed_pct": float(row["changed_pct"]),
            })

    # Special: Llama-3.1-8B EN single-neuron ablation table
    if unit == "llama-3.1-8b_en":
        abl_sets = ["real_race5", "real_religion5", "abl_shared1", "abl_race4", "abl_religion4"]
        for sid in abl_sets:
            if sid not in sets_meta:
                continue
            for cat in ("Religion", "Race_ethnicity", "Gender_identity"):
                for ctx in ("ambig", "disambig", "all"):
                    sub = dir_df[(dir_df["set"] == sid) & (dir_df["category"] == cat) & (dir_df["context"] == ctx)]
                    assert len(sub) == 1
                    row = sub.iloc[0]
                    n_items = int(row["n_items"])
                    if ctx == "all":
                        ch_val = int(med_cat_idx.loc[(sid, cat)])
                        fp_val = 100.0 * ch_val / n_items
                    else:
                        ch_val = int(row["changed"])
                        fp_val = float(row["changed_pct"])
                    j_race5 = float(ov_cat_dfs[cat].loc[sid, "real_race5"]) if cat in ov_cat_dfs else np.nan
                    j_rel5 = float(ov_cat_dfs[cat].loc[sid, "real_religion5"]) if cat in ov_cat_dfs else np.nan
                    accum["ablation"].append({
                        "set": sid,
                        "n_neurons": int(sets_meta[sid]["n_neurons"]),
                        "neurons": ", ".join(f"L{l}.{idx}" for l, idx in sorted(neuron_set_from_dict(sets_meta[sid]["neuron_indices"]))),
                        "category": cat,
                        "context": ctx,
                        "n_items": n_items,
                        "changed": ch_val,
                        "flips_pct": fp_val,
                        "correct_to_incorrect": int(row["correct_to_incorrect"]),
                        "incorrect_to_correct": int(row["incorrect_to_correct"]),
                        "jaccard_vs_real_race5_allctx": j_race5,
                        "jaccard_vs_real_religion5_allctx": j_rel5,
                    })

    # Special: Llama-3.2-3B EN L4 vs A100 rand_gender20_seed0 hardware comparison (canonical pred != base_pred)
    if unit == "llama-3.2-3b_en" and "rand_gender20_seed0_L4" in sets_meta and "rand_gender20_seed0" in sets_meta:
        for cat in all_cats:
            n_items = int(cat_n_map[cat])
            a100_ch = int(med_cat_idx.loc[("rand_gender20_seed0", cat)])
            l4_ch = int(med_cat_idx.loc[("rand_gender20_seed0_L4", cat)])
            a100_fp = 100.0 * a100_ch / n_items
            l4_fp = 100.0 * l4_ch / n_items
            accum["gpu_noise_3b"].append({
                "unit": unit,
                "set_a100": "rand_gender20_seed0 (A100-SXM4-80GB)",
                "set_l4": "rand_gender20_seed0_L4 (L4)",
                "category": cat,
                "n_items": n_items,
                "a100_changed": a100_ch,
                "a100_flips_pct": a100_fp,
                "l4_changed": l4_ch,
                "l4_flips_pct": l4_fp,
                "diff_l4_minus_a100_pp": float(l4_fp - a100_fp),
            })


def process_summary_only_unit(cfg: dict, drive: Path, tracked: list, accum: dict):
    """Handle Llama-3.2-1B EN which has per_set/*_summary.json but no per-item analysis dir."""
    unit = cfg["unit"]
    model = cfg["model"]
    lang = cfg["language"]
    bench = cfg["benchmark"]
    pdir = drive / cfg["per_set_dir"]
    assert pdir.exists(), f"Missing per_set dir: {pdir}"

    summaries = {}
    for f in sorted(pdir.glob("*_summary.json")):
        track_file(tracked, f, drive)
        d = json.load(open(f, encoding="utf-8"))
        summaries[d["set"]] = d

    assert "baseline" in summaries, f"Missing baseline summary in {pdir}"
    base_by_cat = {r["category"]: r for r in summaries["baseline"]["bbq_by_category"]}
    n_items_total = sum(BBQ_ITEM_COUNTS.values())

    real_sids = sorted([s for s, d in summaries.items() if d.get("kind") == "real"], key=lambda s: (-(summaries[s].get("k") or 0), s))
    ks = sorted({int(summaries[s]["k"]) for s in real_sids if summaries[s].get("k") is not None})
    real_cats = sorted({str(summaries[s]["category"]) for s in real_sids if summaries[s].get("category")})
    gpus_used = sorted({normalize_gpu((summaries[s].get("env") or {}).get("gpu")) for s in real_sids})

    accum["inventory"].append({
        "unit": unit,
        "model": model,
        "language": lang,
        "benchmark": bench,
        "n_items_total": n_items_total,
        "n_categories": len(BBQ_CATS),
        "n_real_sets": len(real_sids),
        "n_ablation_sets": 0,
        "n_random_sets": 0,
        "n_random_sensitivity_sets": 0,
        "k_values": ", ".join(str(k) for k in ks),
        "target_categories_tested": ", ".join(real_cats),
        "has_per_item_analysis": False,
        "analysis_dir": cfg["per_set_dir"] + " (summary-only)",
        "gpus_intervened": "; ".join(gpus_used),
        "baseline_gpu": normalize_gpu((summaries["baseline"].get("env") or {}).get("gpu")),
    })

    for sid, d in summaries.items():
        n_idx = d.get("neuron_indices") or {}
        n_neurons = int(d.get("n_neurons") or 0)
        last_cnt = len(n_idx.get(str(cfg["last_layer"]), n_idx.get(cfg["last_layer"], [])))
        accum["provenance"].append({
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "set": sid,
            "kind": d.get("kind"),
            "set_category": d.get("category"),
            "k": d.get("k"),
            "seed": None,
            "mimics_or_matched_to": d.get("mimics"),
            "n_neurons": n_neurons,
            "n_layers": len(n_idx),
            "last_layer": cfg["last_layer"],
            "last_layer_neurons": last_cnt,
            "layer_profile": layer_profile_str(n_idx),
            "gpu": normalize_gpu((d.get("env") or {}).get("gpu")),
            "manifest_sha256": d.get("manifest_sha256"),
            "analysis_dir": cfg["per_set_dir"],
        })

    for sid in real_sids:
        d = summaries[sid]
        gpu = normalize_gpu((d.get("env") or {}).get("gpu"))
        cat_flips = {}
        total_changed = 0
        for row in d["bbq_by_category"]:
            c = row["category"]
            n = BBQ_ITEM_COUNTS[c]
            fp = float(row["flips_pct"])
            ch = int(round(fp * n / 100.0))
            cat_flips[c] = fp
            total_changed += ch
            accum["flips_by_category"].append({
                "unit": unit,
                "model": model,
                "language": lang,
                "benchmark": bench,
                "set": sid,
                "kind": "real",
                "set_category": d.get("category"),
                "k": d.get("k"),
                "gpu": gpu,
                "category": c,
                "context": "all",
                "n_items": n,
                "changed": ch,
                "flips_pct": fp,
            })
            if c in set(cfg["target_categories"]):
                b_row = base_by_cat[c]
                accum["secondary"].append({
                    "unit": unit,
                    "model": model,
                    "language": lang,
                    "benchmark": bench,
                    "set": sid,
                    "kind": "real",
                    "category": c,
                    "context": "disambig",
                    "n_items": n // 2,
                    "changed": np.nan,
                    "changed_pct": np.nan,
                    "to_stereotyped": np.nan,
                    "to_other_person": np.nan,
                    "to_unknown": np.nan,
                    "S_to_A": np.nan,
                    "A_to_S": np.nan,
                    "net_towards_stereotype_pct": np.nan,
                    "correct_to_incorrect": np.nan,
                    "incorrect_to_correct": np.nan,
                    "disamb_bias_delta": float(row["disamb_bias"] - b_row["disamb_bias"]),
                    "disamb_bias_delta_ci_low": np.nan,
                    "disamb_bias_delta_ci_high": np.nan,
                    "esbbq_disamb_bias_delta": np.nan,
                    "esbbq_disamb_bias_delta_ci_low": np.nan,
                    "esbbq_disamb_bias_delta_ci_high": np.nan,
                    "esbbq_amb_bias_delta": np.nan,
                    "esbbq_amb_bias_delta_ci_low": np.nan,
                    "esbbq_amb_bias_delta_ci_high": np.nan,
                })

        tgt_cat = map_target_category(bench, d.get("category"), sid)
        other_mean, ratio, rank = compute_rank_and_ratio(cat_flips, tgt_cat)
        tgt_n = BBQ_ITEM_COUNTS[tgt_cat]
        tgt_pct = cat_flips[tgt_cat]
        tgt_ch = int(round(tgt_pct * tgt_n / 100.0))
        n_idx = d.get("neuron_indices") or {}
        last_cnt = len(n_idx.get(str(cfg["last_layer"]), n_idx.get(cfg["last_layer"], [])))
        accum["localization"].append({
            "unit": unit,
            "model": model,
            "language": lang,
            "benchmark": bench,
            "set": sid,
            "kind": "real",
            "set_category": d.get("category"),
            "target_category": tgt_cat,
            "k": d.get("k"),
            "n_neurons": int(d.get("n_neurons") or 0),
            "n_layers": len(n_idx),
            "last_layer_neurons": last_cnt,
            "gpu": gpu,
            "target_n_items": tgt_n,
            "target_changed": tgt_ch,
            "target_flips_pct": tgt_pct,
            "target_flips_ci_low": np.nan,
            "target_flips_ci_high": np.nan,
            "other_mean_flips_pct": other_mean,
            "specificity_ratio": ratio,
            "target_rank": rank,
            "n_categories": len(BBQ_CATS),
            "all_items_changed": total_changed,
            "all_items_total": n_items_total,
            "all_items_flips_pct": 100.0 * total_changed / n_items_total,
            "disambig_target_flips_pct": np.nan,
            "disambig_other_mean_flips_pct": np.nan,
            "disambig_specificity_ratio": np.nan,
            "disambig_target_rank": np.nan,
        })

    # Neuron overlap among 1B real sets
    nsets = {s: neuron_set_from_dict(summaries[s].get("neuron_indices")) for s in real_sids}
    for i, a in enumerate(real_sids):
        for b in real_sids[i + 1:]:
            na, nb = nsets[a], nsets[b]
            inter = sorted(na & nb)
            union = na | nb
            accum["overlap"].append({
                "unit": unit,
                "model": model,
                "language": lang,
                "benchmark": bench,
                "set_a": a,
                "set_b": b,
                "n_neurons_a": len(na),
                "n_neurons_b": len(nb),
                "n_shared_neurons": len(inter),
                "neuron_jaccard": len(inter) / len(union) if union else np.nan,
                "items_jaccard_all": np.nan,
                "target_cat_a": map_target_category(bench, summaries[a].get("category"), a),
                "items_jaccard_target_a": np.nan,
                "target_cat_b": map_target_category(bench, summaries[b].get("category"), b),
                "items_jaccard_target_b": np.nan,
                "shared_neurons": ", ".join(f"L{l}.{idx}" for l, idx in inter[:15]) + (" ..." if len(inter) > 15 else ""),
            })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", required=True, help="Path to MyDrive/fair_pruning")
    ap.add_argument("--out-dir", required=True, help="Output directory for consolidated CSV tables")
    args = ap.parse_args()

    drive = Path(args.drive_root)
    assert drive.exists(), f"Drive root not found: {drive}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracked = []
    accum = {
        "inventory": [],
        "localization": [],
        "real_vs_random": [],
        "flips_by_category": [],
        "overlap": [],
        "secondary": [],
        "ablation": [],
        "margin_summary": [],
        "margin_deciles": [],
        "groups": [],
        "topics": [],
        "gpu_noise_3b": [],
        "provenance": [],
    }

    for cfg in UNITS_CONFIG:
        print(f"Processing {cfg['unit']} ({cfg['mode']})...")
        if cfg["mode"] == "bbq_per_set_summary":
            process_summary_only_unit(cfg, drive, tracked, accum)
        else:
            process_items_unit(cfg, drive, tracked, accum)

    # L31 weight check
    l31_path = drive / L31_WEIGHTS_REL
    assert l31_path.exists(), f"Missing L31 weights file: {l31_path}"
    track_file(tracked, l31_path, drive)
    l31_df = pd.read_csv(l31_path)

    # K3 GPU noise
    k3_path = drive / K3_GPU_NOISE_REL
    assert k3_path.exists(), f"Missing K3 GPU noise file: {k3_path}"
    track_file(tracked, k3_path, drive)
    k3_df = pd.read_csv(k3_path)

    # Write CSVs
    outputs = {
        "e3_inventory.csv": pd.DataFrame(accum["inventory"]),
        "e3_localization_summary.csv": pd.DataFrame(accum["localization"]),
        "e3_real_vs_random.csv": pd.DataFrame(accum["real_vs_random"]),
        "e3_flips_by_category.csv": pd.DataFrame(accum["flips_by_category"]),
        "e3_neuron_overlap.csv": pd.DataFrame(accum["overlap"]),
        "e3_secondary_metrics.csv": pd.DataFrame(accum["secondary"]),
        "e3_ablation_l31_6950.csv": pd.DataFrame(accum["ablation"]),
        "e3_l31_weights.csv": l31_df,
        "e3_margin_summary.csv": pd.DataFrame(accum["margin_summary"]),
        "e3_margin_deciles.csv": pd.DataFrame(accum["margin_deciles"]),
        "e3_groups.csv": pd.DataFrame(accum["groups"]),
        "e3_esbbq_topics.csv": pd.DataFrame(accum["topics"]),
        "e3_gpu_noise_k3.csv": k3_df,
        "e3_gpu_noise_3b.csv": pd.DataFrame(accum["gpu_noise_3b"]),
        "e3_provenance.csv": pd.DataFrame(accum["provenance"]),
    }

    written_summary = {}
    for fname, df in outputs.items():
        p = out / fname
        df.to_csv(p, index=False)
        written_summary[fname] = {"rows": int(len(df)), "cols": int(len(df.columns)), "bytes": p.stat().st_size}
        print(f"  wrote {fname}: {len(df)} rows x {len(df.columns)} cols ({p.stat().st_size:,} B)")

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "script": "analysis/e3/build_e3_tables.py",
        "script_sha256": sha256(Path(__file__)),
        "repo_commit": git_commit(Path(__file__).parent),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "units": [c["unit"] for c in UNITS_CONFIG],
        "outputs": written_summary,
        "source_files": tracked,
    }
    with open(out / "inputs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Manifest records {len(tracked)} verified Drive files.")


if __name__ == "__main__":
    main()
