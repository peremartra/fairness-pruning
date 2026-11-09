#!/usr/bin/env python3
"""
E3 step 1: compact per-item tables from the per-item lm-eval BBQ logs (about 700 MB each). Run ONCE per unit.

For every set of a unit (baseline, real and random sets, found through the per-set summaries written by notebook 10)
it streams the BBQ log and writes one small table `items/<set_id>.csv.gz` (58,492 rows) plus `items/sets.json`
(set metadata, source log path, size, sha256, checks). Step 2 (`e3_item_analysis.py`) reads only these files.

Mapping (validated 2026-11-02, see claude/BBQ_log_format.md; same rule as notebooks 09 and 10): 12 scored options per
item; options 0 and 1 are the two non-"unknown" answers in document order, options 2 to 11 are phrasings of "unknown";
prediction = argmax of the 12 loglikelihoods (first index on ties); index >= 2 means "unknown".
Stored per item: the loglikelihood of person 0, of person 1 and the maximum over the ten "unknown" phrasings, so the
prediction is also argmax(ll0, ll1, ll_unk) with the same tie rule.

Usage (Colab CPU runtime, Drive mounted):
    pip install ijson
    python analysis/e3/extract_items.py --unit-dir /content/drive/MyDrive/fair_pruning/results/e3/llama-3.1-8b_en
Options: --logs-dir (default <fair_pruning>/checkpoints_e3/results/lm_evals), --out (default <unit-dir>/items),
         --sets (only these set ids), --overwrite (recompute tables that already exist).

E2 (Llama-3.2-3B, notebook 09): its per-set summaries lack the fields of notebook 10 (unit, language, category, k,
baseline_bbq_log) and its logs are named <model>__<set_id>_bbq.json. Use
    python analysis/e3/extract_items.py --unit-dir <fair_pruning>/results/e2 \
        --logs-dir <fair_pruning>/checkpoints_e2/results/lm_evals --log-prefix llama_3.2_3b \
        --baseline-log <fair_pruning>/checkpoints_bbq_rerun/results/lm_evals/meta_llama_llama_3.2_3b_bbq.json
  --log-prefix   name of the logs is <log-prefix>__<set_id>_bbq.json (instead of <unit>_<language>__<set_id>_bbq.json)
  --baseline-log baseline log when the baseline summary does not record one (or there is no baseline summary)
Category and K missing from a summary are taken from the set id (real_religion20 -> Religion, 20); a random set
"rand_<category>_seed<N>" is matched to the real set of that category with the largest K (recorded as `mimics`).
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import platform
import sys
from pathlib import Path

N_ITEMS = 58492
COLUMNS = ["doc_id", "category", "context", "polarity", "question_index", "example_id", "subcategory",
           "stereotyped_groups", "text0", "text1", "group0", "group1", "biased0", "biased1",
           "target", "pred", "correct", "biased", "ll0", "ll1", "ll_unk", "acc_lm_eval"]


def _ll(r):
    while isinstance(r, (list, tuple)):
        r = r[0]
    return float(r)


class _NonFiniteToNull:
    """File wrapper for ijson: the lm-eval logs contain bare NaN / Infinity (valid for Python's json, not for ijson).
    They are replaced by null; the chunk is cut after the last ',' or newline so no token is split."""
    def __init__(self, f, size=1 << 22):
        self.f, self.size, self.carry, self.out, self.eof = f, size, b"", b"", False

    def _fill(self):
        while not self.out and not self.eof:
            chunk = self.f.read(self.size)
            buf = self.carry + chunk
            if not chunk:                       # end of file
                self.eof, self.carry = True, b""
                self.out = self._fix(buf)
                return
            cut = max(buf.rfind(b","), buf.rfind(b"\n"))
            if cut < 0:
                self.carry = buf
                continue
            self.carry = buf[cut + 1:]
            self.out = self._fix(buf[:cut + 1])

    def read(self, n=-1):
        """Returns at most n bytes (ijson's C backend relies on it)."""
        self._fill()
        if n is None or n < 0:
            data, self.out = self.out, b""
            while not self.eof:
                self._fill()
                data, self.out = data + self.out, b""
            return data
        data, self.out = self.out[:n], self.out[n:]
        return data

    @staticmethod
    def _fix(b):
        return b.replace(b"-Infinity", b"null").replace(b"Infinity", b"null").replace(b"NaN", b"null")


def iter_items(path):
    try:
        import ijson
    except ImportError:
        print("   ijson not installed: loading the whole file (several GB of RAM). `pip install ijson` streams it.")
        yield from json.load(open(path, encoding="utf-8"))["samples"]["bbq"]
        return
    with open(path, "rb") as f:
        yield from ijson.items(_NonFiniteToNull(f), "samples.bbq.item")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


SET_CATEGORY = {"religion": "Religion", "race": "Race_ethnicity", "gender": "Gender_identity"}


def set_category_k(sm):
    """Category and K of a set: from the summary (notebook 10) or from the set id (notebook 09, E2)."""
    if sm.get("category") is not None or sm.get("k") is not None:
        return sm.get("category"), sm.get("k")
    parts = sm["set"].split("_")
    if sm["kind"] == "real" and len(parts) == 2:              # real_religion20
        name = parts[1].rstrip("0123456789")
        digits = parts[1][len(name):]
        return SET_CATEGORY.get(name), int(digits) if digits else None
    if sm["kind"] == "random" and len(parts) == 3:            # rand_religion_seed3
        return SET_CATEGORY.get(parts[1]), None
    return None, None


def safe_name(s):
    """Same sanitisation as utils._save_raw_result (name of the per-item log)."""
    return s.replace("/", "_").replace("-", "_").lower()


def item_row(s):
    d = s["doc"]
    info = d["answer_info"]
    persons = [i for i in range(3) if info[f"ans{i}"][1] != "unknown"]
    assert len(persons) == 2, f"doc {s['doc_id']}: {len(persons)} non-unknown answers"
    lls = [_ll(r) for r in s["resps"]]
    p12 = max(range(len(lls)), key=lambda i: lls[i])
    pred = p12 if p12 < 2 else 2
    texts = [d[f"ans{i}"] for i in persons]
    groups = [info[f"ans{i}"][1] for i in persons]
    biased_flags = [int(t in d["biased_answers"]) for t in texts]
    meta = d.get("additional_metadata") or {}
    stereo = meta.get("stereotyped_groups", [])
    if isinstance(stereo, (list, tuple)):
        stereo = "|".join(str(x) for x in stereo)
    target = int(s["target"])
    return {
        "doc_id": s["doc_id"], "category": d["category"], "context": d["context_condition"],
        "polarity": d["question_polarity"], "question_index": d.get("question_index"),
        "example_id": d.get("example_id"), "subcategory": meta.get("subcategory", ""),
        "stereotyped_groups": stereo, "text0": texts[0], "text1": texts[1],
        "group0": groups[0], "group1": groups[1], "biased0": biased_flags[0], "biased1": biased_flags[1],
        "target": target, "pred": pred, "correct": int(pred == target),
        "biased": int(pred < 2 and biased_flags[pred] == 1),
        "ll0": lls[0], "ll1": lls[1], "ll_unk": max(lls[2:]),
        "acc_lm_eval": int(round(float(s["acc"]))) if "acc" in s else "",
    }


def extract(log, out_csv):
    n = mism = 0
    tmp = out_csv.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s in iter_items(log):
            r = item_row(s)
            if r["acc_lm_eval"] != "" and r["acc_lm_eval"] != r["correct"]:
                mism += 1
            w.writerow(r)
            n += 1
    tmp.replace(out_csv)
    return n, mism


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit-dir", required=True, help="fair_pruning/results/e3/<unit>_<lang>")
    ap.add_argument("--logs-dir", help="folder of the per-item BBQ logs of the zeroed sets")
    ap.add_argument("--out", help="output folder (default <unit-dir>/items)")
    ap.add_argument("--sets", nargs="*", help="only these set ids")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-prefix", help="logs named <log-prefix>__<set_id>_bbq.json (E2: llama_3.2_3b)")
    ap.add_argument("--baseline-log", help="baseline BBQ log if the summaries do not record it (E2)")
    a = ap.parse_args()

    unit_dir = Path(a.unit_dir)
    fair = unit_dir.parents[2]                      # .../fair_pruning
    logs_dir = Path(a.logs_dir) if a.logs_dir else fair / "checkpoints_e3" / "results" / "lm_evals"
    out = Path(a.out) if a.out else unit_dir / "items"
    out.mkdir(parents=True, exist_ok=True)

    summaries = sorted((unit_dir / "per_set").glob("*_summary.json"))
    if not summaries:
        sys.exit(f"no summaries in {unit_dir / 'per_set'}")
    sets_path = out / "sets.json"
    meta = json.load(open(sets_path)) if sets_path.exists() else {"sets": {}}

    loaded = [(sp, json.load(open(sp, encoding="utf-8"))) for sp in summaries]
    if a.baseline_log and not any(sm["kind"] == "baseline" for _, sm in loaded):
        loaded.insert(0, (Path(a.baseline_log), {"set": "baseline", "kind": "baseline", "n_neurons": 0}))
    reals = {}                                       # category -> (k, set id) of the real set with the largest K
    for _, sm in loaded:
        cat, k = set_category_k(sm)
        if sm["kind"] == "real" and cat and (cat not in reals or (k or 0) > reals[cat][0]):
            reals[cat] = (k or 0, sm["set"])

    for sp, sm in loaded:
        sid = sm["set"]
        if a.sets and sid not in a.sets:
            continue
        unit_key = f"{sm['unit']}_{sm['language']}" if "unit" in sm else a.log_prefix
        if sm["kind"] == "baseline":
            log = Path(sm.get("baseline_bbq_log") or a.baseline_log or "")   # E1 log reused (gate S4), notebook 10, or --baseline-log
        elif a.log_prefix:
            log = logs_dir / f"{safe_name(a.log_prefix + '__' + sid)}_bbq.json"
        else:
            log = logs_dir / f"{safe_name(unit_key + '__' + sid)}_bbq.json"
        cat, k = set_category_k(sm)
        mimics = sm.get("mimics") or (reals.get(cat, (None, None))[1] if sm["kind"] == "random" else None)
        out_csv = out / f"{sid}.csv.gz"
        if out_csv.exists() and not a.overwrite and sid in meta["sets"]:
            print(f"[{sid}] already extracted — skipped")
            continue
        if not log.exists():
            print(f"[{sid}] log not found: {log} — skipped")
            continue
        t0 = dt.datetime.now()
        print(f"[{sid}] reading {log} ({log.stat().st_size / 1e6:.0f} MB)")
        n, mism = extract(log, out_csv)
        meta["sets"][sid] = {
            "set": sid, "kind": sm["kind"], "category": cat, "k": k,
            "mimics": mimics, "derived_from": sm.get("derived_from"), "seed": int(sid.rsplit("seed", 1)[1]) if "seed" in sid else None,
            "n_neurons": sm.get("n_neurons"), "neuron_indices": sm.get("neuron_indices"),
            "summary": str(sp), "source_log": str(log), "source_log_bytes": log.stat().st_size,
            "source_log_sha256": sha256(log), "table": out_csv.name, "n_items": n,
            "pred_mismatches_vs_lm_eval_acc": mism, "manifest_sha256": sm.get("manifest_sha256"),
            "summary_session_gpu": (sm.get("env") or {}).get("gpu"), "extracted_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        meta["unit"] = unit_key or meta.get("unit")
        meta["python"] = platform.python_version()
        json.dump(meta, open(sets_path, "w", encoding="utf-8"), indent=2)   # saved after every set: resumable
        flag = "" if (n == N_ITEMS and mism == 0) else "   ⚠️  CHECK"
        print(f"   {n} items, {mism} mismatches vs lm-eval acc, {dt.datetime.now() - t0} → {out_csv.name}{flag}")

    print(f"done: {len(meta['sets'])} sets in {sets_path}")


if __name__ == "__main__":
    main()
