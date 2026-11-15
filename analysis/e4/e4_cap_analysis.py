#!/usr/bin/env python3
"""
E4 step 2: capability changes (paired bootstrap over items) from the per-item tables written by `extract_items_cap.py`.
CPU only. Reads `<items-root>/<unit>/items/*.csv.gz` and `sets.json`; writes a NEW timestamped folder, never overwrites.

Per unit and task:
  - every real set (Gender / Race / Religion, K=5 and K=20) against the baseline;
  - where a random control exists (`rand_*_seed*`): random against baseline, and the real set of the same category and K
    against that random set (real minus random).
  - same resampled item indices for every set of a unit-task (paired); B replicates; percentile 95 % interval.
Accuracy tasks: delta in percentage points (pp). WikiText: relative change (%) of word perplexity, resampling documents.
The script describes; the verdict column only applies the margin fixed beforehand (see README, "Definitions").

Usage (Colab CPU runtime, Drive mounted; step 1 done):
    python analysis/e4/e4_cap_analysis.py --items-root /content/drive/MyDrive/fair_pruning/results/e4
Options: --units, --bootstrap (10000), --seed (0), --margin (1.0 pp), --chance-excl (0.35),
         --provenance (default: provenance.csv next to this script),
         --compare-old PATH (the old `e4_bootstrap_deltas.csv`; prints OK/WARN, never stops).
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

UNITS = ["llama_3.1_8b_en", "qwen3_4b_base_en", "salamandra_7b_en", "llama_3.2_3b_en", "llama_3.2_1b_en"]
TASKS = ["wikitext", "arc_challenge", "hellaswag", "hellaswag_es", "mmlu"]
ACC_COL = {"arc_challenge": "acc_norm", "hellaswag": "acc_norm", "hellaswag_es": "acc_norm", "mmlu": "acc"}
CHUNK = 250


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def read_table(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load(items_dir, sid, task):
    """-> dict(keys, hashes, vals, groups). vals: float acc per item, or (n,2) [loglik, n_words] for WikiText. Sorted by key."""
    rows = read_table(items_dir / f"{sid}__{task}.csv.gz")
    if task == "wikitext":
        keys = [(0, int(r["doc_id"])) for r in rows]
        vals = np.array([[float(r["loglik"]), float(r["n_words"])] for r in rows])
        groups = None
    else:
        keys = [(r["group"], int(r["doc_id"])) for r in rows]
        vals = np.array([float(r[ACC_COL[task]]) for r in rows])
        groups = [r["group"] for r in rows]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return dict(keys=[keys[i] for i in order], hashes=[rows[i]["doc_hash"] for i in order], vals=vals[order],
                groups=None if groups is None else [groups[i] for i in order])


def pct(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def verdict(lo, hi, m):
    if hi < -m:
        return "worse"
    if lo > m:
        return "better"
    if lo >= -m and hi <= m:
        return "equivalent"
    return "inconclusive"


def boot_acc(D, B, rng):
    """D: (ncomp, n) float32 per-item differences. Returns replicates (ncomp, B) in pp."""
    n = D.shape[1]
    out = np.empty((D.shape[0], B), dtype=np.float32)
    done = 0
    while done < B:
        c = min(CHUNK, B - done)
        idx = rng.integers(0, n, size=(c, n))
        out[:, done:done + c] = D[:, idx].mean(axis=2)
        done += c
    return out * 100.0


def boot_wiki(LL, NW, pairs, B, rng):
    """LL: dict set -> (n,) loglik; NW (n,) words. pairs: list of (a, b) -> relative change of ppl a vs b in %."""
    n = len(NW)
    idx = rng.integers(0, n, size=(B, n))
    nw = NW[idx].sum(axis=1)
    ppl = {s: np.exp(-LL[s][idx].sum(axis=1) / nw) for s in LL}
    return np.stack([(ppl[a] / ppl[b] - 1) * 100.0 for a, b in pairs])


def word_ppl(ll, nw):
    return float(np.exp(-ll.sum() / nw.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-root", default="/content/drive/MyDrive/fair_pruning/results/e4")
    ap.add_argument("--units", nargs="*", default=UNITS)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--chance-excl", type=float, default=0.35)
    ap.add_argument("--provenance", default=str(Path(__file__).with_name("provenance.csv")))
    ap.add_argument("--compare-old")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    root = Path(a.items_root)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) if a.out_dir else root / "analysis" / f"{stamp}_caps"
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite")
    out.mkdir(parents=True)

    prov = {}
    if Path(a.provenance).exists():
        for r in csv.DictReader(open(a.provenance, encoding="utf-8")):
            prov[(r["unit"], r["set"], r["task"])] = r
    def prov_of(u, s, t):
        r = prov.get((u, s, t)) or prov.get((u, s, "*")) or {}
        return r.get("gpu", "unknown"), r.get("batch_size", "unknown"), r.get("source", "")

    raw, deltas, checks, mmlu_rows, prov_rows, inputs = [], [], [], [], [], {"units": {}}
    for ui, unit in enumerate(a.units):
        idir = root / unit / "items"
        sj = idir / "sets.json"
        if not sj.exists():
            print(f"[{unit}] no sets.json in {idir}: skipped")
            continue
        info = json.load(open(sj, encoding="utf-8"))
        sets = info["sets"]
        inputs["units"][unit] = {"sets_json_sha256": sha256(sj), "tables": {}}
        print(f"\n===== {unit} =====")
        for ti, task in enumerate(TASKS):
            have = {s: e for s, e in sets.items() if task in e["tasks"]}
            if "baseline" not in have:
                print(f"  {task}: no baseline, skipped")
                continue
            data = {s: load(idir, s, task) for s in have}
            for s in have:
                inputs["units"][unit]["tables"][f"{s}__{task}"] = have[s]["tasks"][task]["table_sha256"]
            base = data["baseline"]
            # alignment checks
            for s in have:
                aligned = data[s]["keys"] == base["keys"] and data[s]["hashes"] == base["hashes"]
                m = have[s]["tasks"][task]
                checks.append(dict(unit=unit, set=s, task=task, n_items=m["n_items"], aligned_with_baseline=aligned,
                                   agg_items=m["agg_items"], agg_lm_eval=m["agg_lm_eval"], agg_ok=m["agg_ok"],
                                   cap_json_value=m["cap_value"], cap_ok=m["cap_ok"], extraction_problems="; ".join(m["problems"])))
                g, b, src = prov_of(unit, s, task)
                prov_rows.append(dict(unit=unit, set=s, task=task, gpu=g, batch_size=b, source=src,
                                      log_date=m["log_date"], lm_eval_version=m["lm_eval_version"]))
                if not aligned:
                    sys.exit(f"{unit} {task} {s}: items not aligned with baseline (keys or doc_hash); stopping")
            real = sorted(s for s in have if have[s]["kind"] == "real")
            rand = sorted(s for s in have if have[s]["kind"] == "random")
            # comparisons: (name, kind, set_a, set_b) with delta = a - b
            comps = [(f"{s}_vs_baseline", s, "baseline") for s in real]
            for r in rand:
                comps.append((f"{r}_vs_baseline", r, "baseline"))
                match = [s for s in real if have[s]["category"] == have[r]["category"] and have[s]["k"] == have[r]["k"]]
                comps += [(f"{s}_vs_{r}", s, r) for s in match]
            rng = np.random.default_rng([a.seed, ui, ti])
            n = len(base["keys"])
            if task == "wikitext":
                NW = base["vals"][:, 1]
                for s in have:
                    assert np.array_equal(data[s]["vals"][:, 1], NW), f"{unit} {s}: word counts differ"
                    raw.append(dict(unit=unit, set=s, kind=have[s]["kind"], category=have[s]["category"], k=have[s]["k"],
                                    task=task, metric="word_perplexity", value=word_ppl(data[s]["vals"][:, 0], NW), n=n))
                LL = {s: data[s]["vals"][:, 0] for s in have}
                rep = boot_wiki(LL, NW, [(x, y) for _, x, y in comps], a.bootstrap, rng)
                pts = [(word_ppl(LL[x], NW) / word_ppl(LL[y], NW) - 1) * 100 for _, x, y in comps]
                unit_lbl, excluded = "relative % change of word perplexity", False
            else:
                for s in have:
                    raw.append(dict(unit=unit, set=s, kind=have[s]["kind"], category=have[s]["category"], k=have[s]["k"],
                                    task=task, metric=ACC_COL[task] + (" (micro)" if task == "mmlu" else ""),
                                    value=float(data[s]["vals"].mean() * 100), n=n))
                D = np.stack([data[x]["vals"] - data[y]["vals"] for _, x, y in comps]).astype(np.float32)
                rep = boot_acc(D, a.bootstrap, rng)
                pts = list(D.mean(axis=1) * 100)
                unit_lbl = "pp"
                excluded = bool(base["vals"].mean() < a.chance_excl)
            for (name, x, y), r, p in zip(comps, rep, pts):
                lo, hi = pct(r)
                v = "n/a (no margin for perplexity)" if task == "wikitext" else (
                    "excluded (baseline near chance)" if excluded else verdict(lo, hi, a.margin))
                deltas.append(dict(unit=unit, task=task, comparison=name, set_a=x, set_b=y,
                                   kind=have[x]["kind"], category=have[x]["category"], k=have[x]["k"], n_items=n,
                                   unit_of_delta=unit_lbl, delta=float(p), ci95_lo=lo, ci95_hi=hi,
                                   ci_excludes_zero=bool(lo > 0 or hi < 0), verdict=v,
                                   baseline_value=raw[-len(have) + list(have).index("baseline")]["value"],
                                   bootstrap=a.bootstrap, seed=a.seed, margin_pp=a.margin))
            # MMLU micro / macro
            if task == "mmlu":
                for s in have:
                    subj = {}
                    for gname, v in zip(data[s]["groups"], data[s]["vals"]):
                        subj.setdefault(gname, []).append(v)
                    macro = float(np.mean([np.mean(v) for v in subj.values()]))
                    m = have[s]["tasks"][task]
                    cap = m["cap_value"]
                    mmlu_rows.append(dict(unit=unit, set=s, n_items=n, n_subjects=len(subj),
                                          micro_lm_eval=m["agg_lm_eval"], micro_from_items=m["agg_items"], macro_57_subjects=macro,
                                          cap_json_value=cap, cap_minus_macro_pp=None if cap is None else (cap - macro) * 100))
            top = [d for d in deltas if d["unit"] == unit and d["task"] == task][:3]
            print(f"  {task:14s} n={n:6d} sets={len(have):2d} comparisons={len(comps):2d}  e.g. "
                  + " | ".join(f"{d['comparison']}: {d['delta']:+.2f} [{d['ci95_lo']:+.2f},{d['ci95_hi']:+.2f}] {d['verdict']}" for d in top[:2]))
            del data

    def write(name, rows):
        if not rows:
            return
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    write("raw_values.csv", raw)
    write("deltas.csv", deltas)
    write("mmlu_micro_macro.csv", mmlu_rows)
    write("provenance.csv", prov_rows)
    write("checks.csv", checks)
    # compact table for the repo
    lookup = {(r["unit"], r["set"], r["task"]): r["value"] for r in raw}
    small = []
    for d in deltas:
        small.append(dict(unit=d["unit"], task=d["task"], comparison=d["comparison"], value_a=lookup[(d["unit"], d["set_a"], d["task"])],
                          value_b=lookup[(d["unit"], d["set_b"], d["task"])], delta=round(d["delta"], 3),
                          ci95_lo=round(d["ci95_lo"], 3), ci95_hi=round(d["ci95_hi"], 3), unit_of_delta=d["unit_of_delta"], verdict=d["verdict"]))
    write("e4_capabilities_table.csv", small)
    json.dump({"deltas": deltas, "raw": raw}, open(out / "results_full.json", "w"), indent=1)

    # comparison with the old bootstrap (warning only)
    if a.compare_old and Path(a.compare_old).exists():
        old = list(csv.DictReader(open(a.compare_old, encoding="utf-8")))
        newd = {(d["unit"], d["task"], d["comparison"]): d for d in deltas}
        rows, nbad = [], 0
        for o in old:
            nm = {"real_vs_baseline": "real_gender20_vs_baseline", "random_vs_baseline": "rand_gender20_seed0_vs_baseline",
                  "real_vs_random": "real_gender20_vs_rand_gender20_seed0"}.get(o["comparison"])
            n_ = newd.get((o["model"], o["task"], nm))
            if not n_:
                continue
            dd = abs(float(o["delta"]) - n_["delta"])
            tol = 0.02
            bad = dd > tol
            nbad += bad
            rows.append(dict(unit=o["model"], task=o["task"], comparison=nm, old_delta=o["delta"], new_delta=n_["delta"],
                             abs_diff=dd, old_ci=f"[{float(o['ci95_lo']):.2f},{float(o['ci95_hi']):.2f}]",
                             new_ci=f"[{n_['ci95_lo']:.2f},{n_['ci95_hi']:.2f}]", status="WARN" if bad else "OK"))
        write("compare_old_bootstrap.csv", rows)
        print(f"\nComparison with the old bootstrap: {'OK, all point estimates within 0.02' if not nbad else f'WARN, {nbad} of {len(rows)} differ by more than 0.02'}")

    bad_checks = [c for c in checks if not c["agg_ok"] or c["cap_ok"] is False or c["extraction_problems"]]
    stat = {"n_deltas": len(deltas), "n_checks_with_problems": len(bad_checks)}
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# E4 capability analysis ({stamp} UTC)\n\nB={a.bootstrap}, seed={a.seed}, margin ±{a.margin} pp, near-chance exclusion < {a.chance_excl}.\n\n")
        f.write(f"Checks with problems: {len(bad_checks)} (see checks.csv).\n\n")
        for unit in a.units:
            rows = [d for d in deltas if d["unit"] == unit]
            if not rows:
                continue
            f.write(f"## {unit}\n\n| task | comparison | delta | 95 % CI | verdict |\n|---|---|---|---|---|\n")
            for d in rows:
                f.write(f"| {d['task']} | {d['comparison']} | {d['delta']:+.2f} {d['unit_of_delta'].split()[0]} | [{d['ci95_lo']:+.2f}, {d['ci95_hi']:+.2f}] | {d['verdict']} |\n")
            f.write("\n")
    inputs.update({"script": Path(__file__).name, "script_sha256": sha256(__file__), "repo_commit": git_commit(Path(__file__).parent),
                   "python": platform.python_version(), "numpy": np.__version__, "args": vars(a), "created_utc": stamp,
                   "provenance_file_sha256": sha256(a.provenance) if Path(a.provenance).exists() else None, **stat})
    json.dump(inputs, open(out / "inputs.json", "w"), indent=1)
    print(f"\nWritten to {out}\nChecks with problems: {len(bad_checks)}")


if __name__ == "__main__":
    main()
