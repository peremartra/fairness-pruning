#!/usr/bin/env python3
"""
Build up-to-date, verified E4 capability tables from Google Drive (strictly read-only on Drive).

What this script does:
  1. Copies existing per-item tables from `<drive-root>/results/e4/<unit>/items/` into a local
     temporary working directory (`--work-dir`) so Google Drive is never written or modified.
  2. Compares every `<unit>__<set>_<task>.json` log in `<drive-root>/checkpoints_e3_cap/results/lm_evals/`
     against `sets.json` (by set presence and `log_sha256` / file size + mtime). Any new or updated
     log (for example the 2026-11-23 A100 reruns of `llama_3.2_3b_en`) is extracted into `--work-dir`
     using the repository's `extract_items_cap.py`.
  3. Runs the repository's `e4_cap_analysis.py` (`B=10000`, `seed=0`, `margin=1.0 pp`, `chance-excl=0.35`,
     using the repository's `analysis/e4/provenance.csv`) on `--work-dir` and writes the full table set
     into `--out-dir` (`raw_values.csv`, `deltas.csv`, `e4_capabilities_table.csv`, `mmlu_micro_macro.csv`,
     `checks.csv`, `provenance.csv`, `inputs.json`) plus two compact helper tables (`e4_inventory.csv`,
     `e4_wikitext_random_summary.csv`).
  4. Never overwrites anything on Google Drive.

Usage (Colab CPU runtime with Drive mounted):
    python analysis/e4/build_e4_tables.py \
        --drive-root /content/drive/MyDrive/fair_pruning \
        --provenance analysis/e4/provenance.csv \
        --out-dir /content/e4_tables
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Match e4_cap_analysis.py UNITS order exactly so RNG seed [seed, ui, ti] is bit-for-bit reproducible
UNITS = ["llama_3.1_8b_en", "qwen3_4b_base_en", "salamandra_7b_en", "llama_3.2_3b_en", "llama_3.2_1b_en"]
MODEL_NAMES = {
    "llama_3.1_8b_en": "Llama-3.1-8B",
    "salamandra_7b_en": "Salamandra-7B",
    "qwen3_4b_base_en": "Qwen3-4B-Base",
    "llama_3.2_3b_en": "Llama-3.2-3B",
    "llama_3.2_1b_en": "Llama-3.2-1B",
}
TASKS = ["wikitext", "arc_challenge", "hellaswag", "hellaswag_es", "mmlu"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def find_stale_or_new(work_unit_dir: Path, logs_dir: Path, unit: str, extract_mod) -> list:
    """Return list of (set_id, task, reason) whose Drive log is missing from sets.json or differs in sha256."""
    sj = work_unit_dir / "items" / "sets.json"
    info = json.load(open(sj, encoding="utf-8")) if sj.exists() else {"sets": {}}
    sets = info.get("sets", {})
    stale = []
    for task in TASKS:
        found = extract_mod.discover(logs_dir, unit, task)
        for sid, lp in sorted(found.items()):
            entry = sets.get(sid, {}).get("tasks", {}).get(task)
            if entry is None:
                stale.append((sid, task, "new_log"))
            else:
                old_sz = entry.get("log_bytes")
                cur_sz = lp.stat().st_size
                if old_sz != cur_sz:
                    stale.append((sid, task, f"size_changed({old_sz}->{cur_sz})"))
                else:
                    cur_sha = sha256(lp)
                    if cur_sha != entry.get("log_sha256"):
                        stale.append((sid, task, "sha256_changed"))
    return stale


def build_helper_tables(out_dir: Path):
    raw = pd.read_csv(out_dir / "raw_values.csv")
    deltas = pd.read_csv(out_dir / "deltas.csv")
    prov = pd.read_csv(out_dir / "provenance.csv")
    checks = pd.read_csv(out_dir / "checks.csv")

    # 1. Inventory table
    inv_rows = []
    for u in UNITS:
        ru = raw[raw["unit"] == u]
        if ru.empty:
            continue
        sets_u = ru[["set", "kind", "category", "k"]].drop_duplicates()
        reals = sets_u[sets_u["kind"] == "real"]
        rands = sets_u[sets_u["kind"] == "random"]
        rands_full = ru[(ru["kind"] == "random") & (ru["task"] == "mmlu")]["set"].unique().tolist()
        rands_wt = ru[(ru["kind"] == "random") & (ru["task"] == "wikitext")]["set"].unique().tolist()
        pu = prov[prov["unit"] == u]
        cu = checks[checks["unit"] == u]
        n_bad = int(((~cu["agg_ok"]) | (cu["cap_ok"] == False) | (cu["extraction_problems"].fillna("") != "")).sum())
        real_gpus = sorted({str(g) for g in pu[pu["set"].str.startswith("real_") | (pu["set"] == "baseline")]["gpu"].dropna().unique()})
        rand_gpus = sorted({str(g) for g in pu[pu["set"].str.startswith("rand_")]["gpu"].dropna().unique()})
        inv_rows.append({
            "unit": u,
            "model": MODEL_NAMES.get(u, u),
            "language": "EN",
            "n_tasks": int(ru["task"].nunique()),
            "n_real_sets": int(len(reals)),
            "n_random_full_battery": len(rands_full),
            "n_random_wikitext_only": max(0, len(rands_wt) - len(rands_full)),
            "n_random_total": int(len(rands)),
            "k_values": ", ".join(str(int(k)) for k in sorted(reals["k"].dropna().unique())),
            "mmlu_excluded_chance": bool((deltas[(deltas["unit"] == u) & (deltas["task"] == "mmlu")]["verdict"].str.startswith("excluded")).any()),
            "checks_total": int(len(cu)),
            "checks_problems": n_bad,
            "gpus_baseline_real": "; ".join(real_gpus),
            "gpus_random": "; ".join(rand_gpus) if rand_gpus else "-",
        })
    pd.DataFrame(inv_rows).to_csv(out_dir / "e4_inventory.csv", index=False)

    # 2. WikiText multi-seed random summary
    wt = deltas[deltas["task"] == "wikitext"].copy()
    wt_rows = []
    for u in UNITS:
        wu = wt[wt["unit"] == u]
        if wu.empty:
            continue
        rnd_vs_base = wu[(wu["set_b"] == "baseline") & (wu["set_a"].str.startswith("rand_"))].copy()
        if rnd_vs_base.empty:
            continue
        rnd_vs_base["family"] = rnd_vs_base["set_a"].str.rsplit("_seed", n=1).str[0]
        for fam, g in sorted(rnd_vs_base.groupby("family")):
            real_sid = fam.replace("rand_", "real_", 1)
            r_row = wu[(wu["set_a"] == real_sid) & (wu["set_b"] == "baseline")]
            if r_row.empty:
                continue
            r = r_row.iloc[0]
            rvr = wu[(wu["set_a"] == real_sid) & (wu["set_b"].isin(g["set_a"]))]
            wt_rows.append({
                "unit": u,
                "model": MODEL_NAMES.get(u, u),
                "real_set": real_sid,
                "random_family": fam,
                "n_seeds": int(len(g)),
                "baseline_ppl": float(r["baseline_value"]),
                "real_delta_pct": float(r["delta"]),
                "real_ci95_lo": float(r["ci95_lo"]),
                "real_ci95_hi": float(r["ci95_hi"]),
                "rand_delta_min_pct": float(g["delta"].min()),
                "rand_delta_mean_pct": float(g["delta"].mean()),
                "rand_delta_max_pct": float(g["delta"].max()),
                "real_vs_rand_mean_pct": float(rvr["delta"].mean()) if not rvr.empty else np.nan,
                "real_vs_rand_min_pct": float(rvr["delta"].min()) if not rvr.empty else np.nan,
                "real_vs_rand_max_pct": float(rvr["delta"].max()) if not rvr.empty else np.nan,
            })
    pd.DataFrame(wt_rows).to_csv(out_dir / "e4_wikitext_random_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", required=True, help="Path to MyDrive/fair_pruning (read-only)")
    ap.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent), help="Directory containing extract_items_cap.py and e4_cap_analysis.py")
    ap.add_argument("--provenance", default=str(Path(__file__).with_name("provenance.csv")), help="Path to provenance.csv")
    ap.add_argument("--work-dir", default="/content/e4_work_items", help="Local temp directory for working copy of items/")
    ap.add_argument("--out-dir", required=True, help="Output directory for consolidated E4 CSV tables")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--chance-excl", type=float, default=0.35)
    args = ap.parse_args()

    drive = Path(args.drive_root)
    assert drive.exists(), f"Drive root not found: {drive}"
    scripts_dir = Path(args.scripts_dir)
    extract_script = scripts_dir / "extract_items_cap.py"
    analysis_script = scripts_dir / "e4_cap_analysis.py"
    assert extract_script.exists() and analysis_script.exists(), f"Missing E4 scripts in {scripts_dir}"

    sys.path.insert(0, str(scripts_dir))
    import extract_items_cap as extract_mod

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    logs_dir = drive / "checkpoints_e3_cap" / "results" / "lm_evals"
    a100_rerun_logs_dir = drive / "checkpoints_e4_a100_rerun" / "results" / "lm_evals"
    refreshed_log = []

    for unit in UNITS:
        src_items = drive / "results" / "e4" / unit / "items"
        dst_items = work / unit / "items"
        assert src_items.exists(), f"Missing Drive items dir: {src_items}"
        shutil.copytree(src_items, dst_items)
        stale = find_stale_or_new(work / unit, logs_dir, unit, extract_mod)
        print(f"[{unit}] copied {len(list(dst_items.iterdir()))} item files; stale/new in checkpoints_e3_cap: {len(stale)}", flush=True)
        if stale:
            for sid, task, reason in stale:
                print(f"  -> refreshing {sid} / {task} ({reason})", flush=True)
                refreshed_log.append({"unit": unit, "set": sid, "task": task, "source": "checkpoints_e3_cap", "reason": reason})
            sets_to_run = sorted({sid for sid, _, _ in stale})
            tasks_to_run = sorted({t for _, t, _ in stale})
            cmd = [
                sys.executable, "-u", str(extract_script),
                "--fair-pruning-dir", str(drive),
                "--units", unit,
                "--out-root", str(work),
                "--sets", *sets_to_run,
                "--tasks", *tasks_to_run,
                "--overwrite",
            ]
            r = subprocess.run(cmd, check=False)
            assert r.returncode == 0, f"extract_items_cap.py failed for {unit} (rc={r.returncode})"

        # Overlay any A100 reruns in checkpoints_e4_a100_rerun/ (takes precedence over earlier L4 logs)
        if a100_rerun_logs_dir.exists():
            a100_stale = find_stale_or_new(work / unit, a100_rerun_logs_dir, unit, extract_mod)
            if a100_stale:
                print(f"[{unit}] overlaying {len(a100_stale)} A100 rerun logs from checkpoints_e4_a100_rerun", flush=True)
                for sid, task, reason in a100_stale:
                    print(f"  -> overlaying {sid} / {task} (A100 rerun)", flush=True)
                    refreshed_log.append({"unit": unit, "set": sid, "task": task, "source": "checkpoints_e4_a100_rerun", "reason": "A100 rerun"})
                sets_to_run = sorted({sid for sid, _, _ in a100_stale})
                tasks_to_run = sorted({t for _, t, _ in a100_stale})
                empty_fp = work / "_no_old_cap_dir"
                empty_fp.mkdir(exist_ok=True)
                cmd = [
                    sys.executable, "-u", str(extract_script),
                    "--fair-pruning-dir", str(empty_fp),
                    "--logs-dir", str(a100_rerun_logs_dir),
                    "--units", unit,
                    "--out-root", str(work),
                    "--sets", *sets_to_run,
                    "--tasks", *tasks_to_run,
                    "--overwrite",
                ]
                r = subprocess.run(cmd, check=False)
                assert r.returncode == 0, f"extract_items_cap.py overlay failed for {unit} (rc={r.returncode})"

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    # Run unmodified e4_cap_analysis.py (same B, seed, margin, chance-excl)
    cmd = [
        sys.executable, "-u", str(analysis_script),
        "--items-root", str(work),
        "--units", *UNITS,
        "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
        "--margin", str(args.margin),
        "--chance-excl", str(args.chance_excl),
        "--provenance", str(Path(args.provenance).resolve()),
        "--out-dir", str(out),
    ]
    print("\nRunning e4_cap_analysis.py...", flush=True)
    r = subprocess.run(cmd, check=False)
    assert r.returncode == 0, f"e4_cap_analysis.py failed (rc={r.returncode})"

    # Drop large JSON + intermediate report.md from tables dir (keep compact CSVs + inputs.json)
    for drop in ("results_full.json", "report.md"):
        p = out / drop
        if p.exists():
            p.unlink()

    build_helper_tables(out)

    # Enrich inputs.json with refresh record
    inp_path = out / "inputs.json"
    inp = json.load(open(inp_path, encoding="utf-8"))
    inp["build_e4_tables_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    inp["refreshed_from_drive_logs"] = refreshed_log
    json.dump(inp, open(inp_path, "w", encoding="utf-8"), indent=2)

    print("\nConsolidated E4 tables ready in", out, flush=True)
    for p in sorted(out.iterdir()):
        print(f"  {p.name:35s} {p.stat().st_size:>10,d} B", flush=True)


if __name__ == "__main__":
    main()
