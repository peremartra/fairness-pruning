#!/usr/bin/env python3
"""
E4 step 1: compact per-item tables of the capability logs (ARC-Challenge, HellaSwag, HellaSwag-es, MMLU, WikiText).
Run ONCE per unit; it is resumable. Step 2 (`e4_cap_analysis.py`) reads only these tables.

For every set of a unit (baseline, real sets, random sets) and every task with a per-item lm-eval log it streams the log and
writes `items/<set_id>__<task>.csv.gz` plus `items/sets.json` (set metadata, source log path, size, sha256, checks).

Logs read (written by notebook 10, `log_samples=True`):
    <fair_pruning>/checkpoints_e3_cap/results/lm_evals/<unit>__<set_id>_<task>.json
Files whose name does not match exactly (for example `..._mmlu_bs1.json`) are NOT read.
Per-set checkpoints read for a cross-check only:
    <fair_pruning>/checkpoints_e3_cap/<unit>/<set_id>_cap.json

Tables (one row per item; sorted by group, doc_id):
    ARC-C, HellaSwag, HellaSwag-es, MMLU: group (the subject in MMLU), doc_id, doc_hash, target, acc, acc_norm
        (acc_norm empty in MMLU). The metric of the paper is acc_norm in ARC-C / HellaSwag / HellaSwag-es and acc in MMLU.
    WikiText: doc_id, doc_hash, loglik, n_words (word perplexity = exp(-sum(loglik) / sum(n_words)))

Checks (stored in sets.json under `problems`; a file with problems prints "CHECK"; nothing stops the run):
    number of items (ARC-C 1,172; HellaSwag 10,042; HellaSwag-es 9,374; MMLU 14,042 in 57 subjects; WikiText 62 documents);
    aggregate recomputed from the items equals lm-eval's `results` (accuracy 1e-5; WikiText relative 5e-4);
    argmax of the logged loglikelihoods equals the logged `acc` in every item;
    finite values; no `limit` in the log; number of few-shot examples (MMLU 5, the rest 0);
    cross-check with the value stored in the `_cap.json` (informational; MMLU there is the macro average over subjects,
    so it is not compared).

Usage (Colab CPU runtime, Drive mounted):
    python analysis/e4/extract_items_cap.py --fair-pruning-dir /content/drive/MyDrive/fair_pruning
Options: --units (default: the five units below), --logs-dir, --out-root (default <fair_pruning>/results/e4),
         --sets (only these set ids), --tasks (only these tasks), --overwrite (recompute tables that already exist).
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import platform
import re
import subprocess
from pathlib import Path

UNITS = ["llama_3.1_8b_en", "qwen3_4b_base_en", "salamandra_7b_en", "llama_3.2_3b_en", "llama_3.2_1b_en"]
TASKS = ["arc_challenge", "hellaswag", "hellaswag_es", "mmlu", "wikitext"]
EXPECTED_ITEMS = {"arc_challenge": 1172, "hellaswag": 10042, "hellaswag_es": 9374, "mmlu": 14042, "wikitext": 62}
# task -> (metric of the paper, key of that metric in the item, key in results[task] of the log, key in the _cap.json)
METRIC = {
    "arc_challenge": ("acc_norm", "acc_norm,none", "acc_norm"),
    "hellaswag": ("acc_norm", "acc_norm,none", "acc_norm"),
    "hellaswag_es": ("acc_norm", "acc_norm,none", "acc_norm"),
    "mmlu": ("acc", "acc,none", "accuracy"),
}
EXPECTED_FEWSHOT = {"mmlu": 5}   # every other task: 0
SET_RE = re.compile(r"^(baseline|real_(?P<rc>gender|race|religion)(?P<rk>\d+)|rand_(?P<qc>gender|race|religion)(?P<qk>\d+)_seed(?P<seed>\d+))$")
CATEGORY = {"gender": "Gender_identity", "race": "Race_ethnicity", "religion": "Religion"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def _ll(r):
    while isinstance(r, (list, tuple)):
        r = r[0]
    return float(r)


def set_info(sid):
    m = SET_RE.match(sid)
    if not m:
        return None
    if sid == "baseline":
        return {"kind": "baseline", "category": None, "k": 0, "seed": None}
    if sid.startswith("real_"):
        return {"kind": "real", "category": CATEGORY[m["rc"]], "k": int(m["rk"]), "seed": None}
    return {"kind": "random", "category": CATEGORY[m["qc"]], "k": int(m["qk"]), "seed": int(m["seed"])}


def discover(logs_dir, unit, task):
    """set ids with a log `<unit>__<set_id>_<task>.json` (exact name, so `_mmlu_bs1` and similar never match)."""
    suffix = f"_{task}.json"
    out = {}
    for p in sorted(Path(logs_dir).glob(f"{unit}__*{suffix}")):
        sid = p.name[len(unit) + 2:-len(suffix)]
        if set_info(sid):
            out[sid] = p
    return out


def read_cap_value(cap_path, task):
    if not cap_path.exists():
        return None
    try:
        r = json.load(open(cap_path, encoding="utf-8"))["results"][task]
        key = "word_perplexity,none" if task == "wikitext" else METRIC[task][2]
        return float(r[key])
    except Exception:
        return None


def open_gz_text(path):
    raw = open(path, "wb")
    gz = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)      # mtime 0: the same table gives the same bytes and sha256
    return io.TextIOWrapper(gz, newline="", encoding="utf-8"), gz, raw


def process(log_path, task, table_path, cap_value):
    d = json.load(open(log_path, encoding="utf-8"))
    samples, res = d["samples"], d["results"]
    problems = []
    rows = []
    if task == "wikitext":
        for it in samples["wikitext"]:
            ll, nw = it["word_perplexity"]
            rows.append((it["doc_id"], it["doc_hash"], float(ll), float(nw)))
        rows.sort(key=lambda r: r[0])
        sum_ll, sum_nw = sum(r[2] for r in rows), sum(r[3] for r in rows)
        agg_items = math.exp(-sum_ll / sum_nw)
        agg_lm = float(res["wikitext"]["word_perplexity,none"])
        agg_ok = abs(agg_items - agg_lm) <= 5e-4 * abs(agg_lm)
        if not all(math.isfinite(x) for r in rows for x in r[2:]):
            problems.append("non-finite values")
        header = ["doc_id", "doc_hash", "loglik", "n_words"]
        n_groups, argmax_mismatch = 1, 0
    else:
        metric, agg_key, _ = METRIC[task]
        argmax_mismatch, comparable = 0, 0
        for g, items in samples.items():
            for it in items:
                acc = float(it["acc"])
                accn = it.get("acc_norm")
                target = it.get("target")
                try:
                    lls = [_ll(r) for r in it["resps"]]
                    pred = max(range(len(lls)), key=lambda i: lls[i])
                    comparable += 1
                    argmax_mismatch += int(int(pred == int(target)) != int(round(acc)))
                except Exception:
                    pass
                rows.append((g, it["doc_id"], it["doc_hash"], target, acc, "" if accn is None else float(accn)))
        rows.sort(key=lambda r: (r[0], r[1]))
        col = 5 if metric == "acc_norm" else 4
        vals = [float(r[col]) for r in rows]
        if any(not math.isfinite(v) for v in vals):
            problems.append("non-finite values")
        agg_items = sum(vals) / len(vals)
        agg_lm = float(res[task][agg_key])
        agg_ok = abs(agg_items - agg_lm) <= 1e-5
        header = ["group", "doc_id", "doc_hash", "target", "acc", "acc_norm"]
        n_groups = len(samples)
        if comparable == 0:
            problems.append("argmax check not possible (no resps/target)")
    n = len(rows)
    if n != EXPECTED_ITEMS[task]:
        problems.append(f"{n} items, expected {EXPECTED_ITEMS[task]}")
    if task == "mmlu" and n_groups != 57:
        problems.append(f"{n_groups} MMLU subjects, expected 57")
    if not agg_ok:
        problems.append(f"aggregate from items {agg_items:.6f} != lm-eval {agg_lm:.6f}")
    if argmax_mismatch:
        problems.append(f"{argmax_mismatch} items where argmax(resps) disagrees with the logged acc")
    cfg = d.get("config") or {}
    if cfg.get("limit") not in (None, 0):
        problems.append(f"log run with limit={cfg.get('limit')}")
    fewshot = sorted({str(c.get("num_fewshot")) for c in (d.get("configs") or {}).values()})
    want = str(EXPECTED_FEWSHOT.get(task, 0))
    if fewshot and fewshot != [want]:
        problems.append(f"num_fewshot {fewshot}, expected [{want}]")
    cap_ok = None
    if cap_value is not None and task != "mmlu":       # the MMLU value of the _cap.json is a macro average
        cap_ok = abs(cap_value - agg_items) <= 1.5e-4
        if not cap_ok:
            problems.append(f"_cap.json value {cap_value} != items {agg_items:.6f}")
    tmp = table_path.with_suffix(".tmp")
    f, gz, raw = open_gz_text(tmp)
    try:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([repr(x) if isinstance(x, float) else x for x in r])
    finally:
        f.close(); gz.close(); raw.close()
    tmp.replace(table_path)
    meta = {"table": table_path.name, "n_items": n, "n_groups": n_groups, "agg_lm_eval": agg_lm, "agg_items": agg_items,
            "agg_ok": bool(agg_ok), "cap_value": cap_value, "cap_ok": cap_ok, "argmax_mismatches": argmax_mismatch,
            "lm_eval_version": d.get("lm_eval_version"), "log_date": d.get("date"), "num_fewshot": fewshot,
            "problems": problems}
    del d, samples, res, rows
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fair-pruning-dir", default="/content/drive/MyDrive/fair_pruning")
    ap.add_argument("--units", nargs="*", default=UNITS)
    ap.add_argument("--logs-dir")
    ap.add_argument("--out-root")
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    fp = Path(a.fair_pruning_dir)
    logs_dir = Path(a.logs_dir) if a.logs_dir else fp / "checkpoints_e3_cap" / "results" / "lm_evals"
    cap_dir = fp / "checkpoints_e3_cap"
    out_root = Path(a.out_root) if a.out_root else fp / "results" / "e4"
    n_check = 0
    for unit in a.units:
        out = out_root / unit / "items"
        out.mkdir(parents=True, exist_ok=True)
        sj = out / "sets.json"
        info = json.load(open(sj, encoding="utf-8")) if sj.exists() else {"sets": {}}
        info.update({"unit": unit, "script_sha256": sha256(__file__), "repo_commit": git_commit(Path(__file__).parent),
                     "python": platform.python_version(), "logs_dir": str(logs_dir)})
        found = {t: discover(logs_dir, unit, t) for t in a.tasks}
        set_ids = sorted({s for t in found.values() for s in t}, key=lambda s: (set_info(s)["kind"] != "baseline", set_info(s)["kind"], s))
        if a.sets:
            set_ids = [s for s in set_ids if s in a.sets]
        print(f"\n===== {unit}: {len(set_ids)} sets =====")
        for sid in set_ids:
            entry = info["sets"].setdefault(sid, {**set_info(sid), "tasks": {}})
            for task in a.tasks:
                lp = found[task].get(sid)
                if lp is None:
                    continue
                tp = out / f"{sid}__{task}.csv.gz"
                if tp.exists() and task in entry["tasks"] and not a.overwrite:
                    print(f"  [{sid}] {task}: already extracted, skipped")
                    continue
                cap_value = read_cap_value(cap_dir / unit / f"{sid}_cap.json", task)
                meta = process(lp, task, tp, cap_value)
                meta.update({"log": str(lp), "log_bytes": lp.stat().st_size, "log_sha256": sha256(lp),
                             "table_sha256": sha256(tp)})
                entry["tasks"][task] = meta
                flag = "CHECK" if meta["problems"] else "ok"
                n_check += bool(meta["problems"])
                print(f"  [{sid}] {task}: n={meta['n_items']} agg={meta['agg_items']:.4f} lm-eval={meta['agg_lm_eval']:.4f} {flag}"
                      + ("" if not meta["problems"] else " " + "; ".join(meta["problems"])))
                info["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
                json.dump(info, open(sj, "w", encoding="utf-8"), indent=1)       # after every file: a crash loses nothing
        json.dump(info, open(sj, "w", encoding="utf-8"), indent=1)
        print(f"  tables and sets.json in {out}")
    print(f"\nDone. Files with problems: {n_check}" + (" (see `problems` in sets.json)" if n_check else ""))


if __name__ == "__main__":
    main()
