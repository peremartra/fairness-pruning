#!/usr/bin/env python3
"""
E3 step 2: per-item analysis of one unit (for example Llama-3.1-8B EN) from the compact tables of extract_items.py.

Questions (descriptive only: no thresholds, no verdicts):
  1. direction  - where do the changed answers go (towards the stereotyped person, the other person, "unknown")?
  2. groups     - within a category, which groups (answer_info) carry the changes?
  3. overlap    - do different sets change the same items (Jaccard of the changed items)?
  4. margin     - are the changed items those where the unpruned model was undecided (small loglikelihood margin)?
  5. bootstrap  - item bootstrap intervals for the change in disambiguated bias and for the % of changed answers.

Usage (laptop; input = the `items/` folder written by step 1, a few MB per set):
    python analysis/e3/e3_item_analysis.py --items-dir "<Drive>/fair_pruning/results/e3/llama-3.1-8b_en/items"
Options: --out-dir, --bootstrap 1000, --seed 0, --target-categories Religion Race_ethnicity Gender_identity
Output: <items-dir>/../analysis/<timestamp>_items/ (report.md, CSV files, results_full.json, inputs.json).
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

CATS = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance", "Race_ethnicity",
        "Race_x_SES", "Race_x_gender", "Religion", "SES", "Sexual_orientation"]
STATES = ["S", "A", "U"]   # S = stereotyped person (in biased_answers), A = the other person, U = "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def state(df):
    return np.where(df["pred"].to_numpy() == 2, "U", np.where(df["biased"].to_numpy() == 1, "S", "A"))


def load(items_dir):
    meta = json.load(open(items_dir / "sets.json", encoding="utf-8"))["sets"]
    tables = {}
    for sid, m in meta.items():
        # keep_default_na=False: group labels such as "NA" / "None" must stay strings (otherwise NaN != NaN breaks the check)
        t = pd.read_csv(items_dir / m["table"], keep_default_na=False, na_values=[""]).set_index("doc_id").sort_index()
        t["state"] = state(t)
        lls = t[["ll0", "ll1", "ll_unk"]].to_numpy()
        srt = np.sort(lls, axis=1)
        t["margin"] = srt[:, -1] - srt[:, -2]
        tables[sid] = t
    base_ids = [s for s, m in meta.items() if m["kind"] == "baseline"]
    assert len(base_ids) == 1, f"expected one baseline, found {base_ids}"
    base = base_ids[0]
    for sid, t in tables.items():
        assert t.index.equals(tables[base].index), f"{sid}: items differ from the baseline"
        cols = ["category", "context", "group0", "group1", "target"]
        eq = (t[cols].fillna("") == tables[base][cols].fillna(""))   # NaN-safe (empty labels are NaN)
        bad = [c for c in cols if not eq[c].all()]
        assert not bad, f"{sid}: item metadata differs from the baseline in {bad}; first rows: {list(eq.index[~eq[bad[0]]][:3])}"
    order = [base] + sorted([s for s in meta if meta[s]["kind"] == "real"], key=lambda s: (-int(meta[s]["k"] or 0), s)) \
        + sorted([s for s in meta if meta[s]["kind"] == "random"], key=lambda s: (s.rsplit("_seed", 1)[0], meta[s]["seed"]))
    return meta, tables, base, order


def disamb_bias(df):
    d = df[df["context"] == "disambig"]
    nonunk = (d["pred"] < 2).sum()
    return 100 * (2 * d["biased"].sum() / nonunk - 1) if nonunk else np.nan


# ── 1. direction ────────────────────────────────────────────────────────────
def direction(meta, tables, base, order):
    b = tables[base]
    rows, trans = [], []
    for sid in order:
        if sid == base:
            continue
        t = tables[sid]
        for cat in CATS:
            for ctx in ("ambig", "disambig", "all"):
                m = (b["category"] == cat) & ((b["context"] == ctx) if ctx != "all" else True)
                bs, ts = b.loc[m, "state"], t.loc[m, "state"]
                n = int(m.sum())
                ch = bs != ts
                ct = pd.crosstab(bs, ts).reindex(index=STATES, columns=STATES, fill_value=0)
                for f in STATES:
                    for to in STATES:
                        trans.append({"set": sid, "kind": meta[sid]["kind"], "category": cat, "context": ctx,
                                      "from": f, "to": to, "n": int(ct.loc[f, to])})
                to_s = int((ch & (ts == "S")).sum()); to_a = int((ch & (ts == "A")).sum()); to_u = int((ch & (ts == "U")).sum())
                rows.append({"set": sid, "kind": meta[sid]["kind"], "category": cat, "context": ctx, "n_items": n,
                             "changed": int(ch.sum()), "changed_pct": 100 * ch.mean() if n else np.nan,
                             "to_stereotyped": to_s, "to_other_person": to_a, "to_unknown": to_u,
                             "S_to_A": int(ct.loc["S", "A"]), "A_to_S": int(ct.loc["A", "S"]),
                             "net_towards_stereotype_pct": 100 * (int(ct.loc["A", "S"]) - int(ct.loc["S", "A"])) / n if n else np.nan,
                             "correct_to_incorrect": int(((b.loc[m, "correct"] == 1) & (t.loc[m, "correct"] == 0)).sum()),
                             "incorrect_to_correct": int(((b.loc[m, "correct"] == 0) & (t.loc[m, "correct"] == 1)).sum())})
    return pd.DataFrame(rows), pd.DataFrame(trans)


# ── 2. groups ───────────────────────────────────────────────────────────────
def groups(meta, tables, base, order):
    b = tables[base]
    rows = []
    for cat in CATS:
        bc = b[b["category"] == cat]
        gs = sorted(set(bc["group0"]) | set(bc["group1"]))
        for g in gs:
            m_idx = bc.index[(bc["group0"] == g) | (bc["group1"] == g)]
            for sid in order:
                if sid == base:
                    continue
                ch = (tables[sid].loc[m_idx, "pred"] != b.loc[m_idx, "pred"])
                rows.append({"set": sid, "kind": meta[sid]["kind"], "category": cat, "group": g,
                             "n_items": len(m_idx), "changed": int(ch.sum()), "changed_pct": 100 * ch.mean()})
    return pd.DataFrame(rows)


# ── 3. overlap ──────────────────────────────────────────────────────────────
def overlap(tables, base, order, category=None):
    b = tables[base]
    m = (b["category"] == category) if category else pd.Series(True, index=b.index)
    changed = {sid: set(b.index[m & (tables[sid]["pred"] != b["pred"])]) for sid in order if sid != base}
    ids = list(changed)
    mat = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for x in ids:
        for y in ids:
            u = changed[x] | changed[y]
            mat.loc[x, y] = len(changed[x] & changed[y]) / len(u) if u else np.nan
    return mat


# ── 4. margin ───────────────────────────────────────────────────────────────
def margin(meta, tables, base, order):
    b = tables[base]
    dec = pd.qcut(b["margin"].rank(method="first"), 10, labels=False) + 1   # deciles of the baseline margin, 1 = smallest
    edges = b.groupby(dec)["margin"].agg(["min", "max"])
    rows, med = [], []
    for sid in order:
        if sid == base:
            continue
        ch = tables[sid]["pred"] != b["pred"]
        for cat in CATS + ["ALL"]:
            m = (b["category"] == cat) if cat != "ALL" else pd.Series(True, index=b.index)
            med.append({"set": sid, "kind": meta[sid]["kind"], "category": cat,
                        "median_margin_changed": b.loc[m & ch, "margin"].median(),
                        "median_margin_unchanged": b.loc[m & ~ch, "margin"].median(),
                        "n_changed": int((m & ch).sum())})
        for d in range(1, 11):
            m = dec == d
            rows.append({"set": sid, "kind": meta[sid]["kind"], "decile": d, "margin_min": edges.loc[d, "min"],
                         "margin_max": edges.loc[d, "max"], "n_items": int(m.sum()), "changed_pct": 100 * ch[m].mean()})
    return pd.DataFrame(med), pd.DataFrame(rows)


# ── 5. bootstrap ────────────────────────────────────────────────────────────
def bootstrap(meta, tables, base, order, n_boot, seed):
    b = tables[base]
    rng = np.random.default_rng(seed)
    rows = []
    for cat in CATS:
        bc = b[b["category"] == cat]
        dis = bc.index[bc["context"] == "disambig"]
        idx_d = rng.integers(0, len(dis), size=(n_boot, len(dis)), dtype=np.int32)    # same resamples for every set
        idx_a = rng.integers(0, len(bc), size=(n_boot, len(bc)), dtype=np.int32)
        bb, bn = b.loc[dis, "biased"].to_numpy(), (b.loc[dis, "pred"] < 2).to_numpy()
        base_bias = 100 * (2 * bb[idx_d].sum(1) / bn[idx_d].sum(1) - 1)
        for sid in order:
            if sid == base:
                continue
            t = tables[sid]
            tb, tn = t.loc[dis, "biased"].to_numpy(), (t.loc[dis, "pred"] < 2).to_numpy()
            delta = 100 * (2 * tb[idx_d].sum(1) / tn[idx_d].sum(1) - 1) - base_bias
            ch = (t.loc[bc.index, "pred"].to_numpy() != bc["pred"].to_numpy())
            chp = 100 * ch[idx_a].mean(1)
            rows.append({"set": sid, "kind": meta[sid]["kind"], "category": cat,
                         "disamb_bias_delta": disamb_bias(t[t["category"] == cat]) - disamb_bias(bc),
                         "disamb_bias_delta_ci_low": np.nanpercentile(delta, 2.5), "disamb_bias_delta_ci_high": np.nanpercentile(delta, 97.5),
                         "changed_pct": 100 * ch.mean(),
                         "changed_pct_ci_low": np.percentile(chp, 2.5), "changed_pct_ci_high": np.percentile(chp, 97.5)})
    return pd.DataFrame(rows)


# ── report ──────────────────────────────────────────────────────────────────
def md(df, nd=2):
    return df.to_markdown(floatfmt=f".{nd}f") if hasattr(df, "to_markdown") else df.round(nd).to_string()


def with_random_range(df, value_cols, keys):
    """Real sets as rows plus min / mean / max over the random sets of each random family."""
    real = df[df["kind"] == "real"].copy()
    rnd = df[df["kind"] == "random"].copy()
    rnd["family"] = rnd["set"].str.rsplit("_seed", n=1).str[0]
    agg = rnd.groupby(keys + ["family"])[value_cols].agg(["min", "mean", "max"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return real, agg.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-categories", nargs="*", default=["Religion", "Race_ethnicity", "Gender_identity"])
    a = ap.parse_args()

    items_dir = Path(a.items_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) if a.out_dir else items_dir.parent / "analysis" / f"{stamp}_items"
    out.mkdir(parents=True, exist_ok=True)

    meta, tables, base, order = load(items_dir)
    print(f"{len(tables)} sets ({base} + {len(order) - 1}); writing to {out}")

    dir_df, trans_df = direction(meta, tables, base, order)
    grp_df = groups(meta, tables, base, order)
    ov_all = overlap(tables, base, order)
    ov_cat = {c: overlap(tables, base, order, c) for c in a.target_categories}
    med_df, dec_df = margin(meta, tables, base, order)
    bs_df = bootstrap(meta, tables, base, order, a.bootstrap, a.seed)

    dir_df.to_csv(out / "direction.csv", index=False)
    trans_df.to_csv(out / "transitions.csv", index=False)
    grp_df.to_csv(out / "groups.csv", index=False)
    ov_all.to_csv(out / "overlap_changed_items_all.csv")
    for c, m in ov_cat.items():
        m.to_csv(out / f"overlap_changed_items_{c}.csv")
    med_df.to_csv(out / "margin_changed_vs_unchanged.csv", index=False)
    dec_df.to_csv(out / "margin_deciles.csv", index=False)
    bs_df.to_csv(out / "bootstrap.csv", index=False)

    # report
    R = [f"# E3 per-item analysis — {json.load(open(items_dir / 'sets.json'))['unit']}", "",
         f"Generated {dt.datetime.now().isoformat(timespec='seconds')}. Baseline: `{base}`. "
         f"Real sets: {sum(meta[s]['kind'] == 'real' for s in meta)}; random sets: {sum(meta[s]['kind'] == 'random' for s in meta)}. "
         "Descriptive only. Random families are summarised by min / mean / max.", ""]
    for cat in a.target_categories:
        R += [f"## {cat}", ""]
        d = dir_df[(dir_df["category"] == cat) & (dir_df["context"] != "all")]
        cols = ["changed_pct", "to_stereotyped", "to_other_person", "to_unknown", "net_towards_stereotype_pct",
                "correct_to_incorrect", "incorrect_to_correct"]
        real, rr = with_random_range(d, cols, ["context"])
        R += ["### 1. Direction of the changed answers (counts; net = (A→S − S→A) / items, %)", "",
              md(real.set_index(["set", "context"])[["n_items"] + cols]), "", "Random sets:", "",
              md(rr.set_index(["family", "context"])), ""]
        g = grp_df[grp_df["category"] == cat]
        piv = g[g["kind"] == "real"].pivot_table(index="group", columns="set", values="changed_pct")
        gr = g[g["kind"] == "random"].copy()
        gr["family"] = gr["set"].str.rsplit("_seed", n=1).str[0]
        rng_ = gr.groupby(["group", "family"])["changed_pct"].agg(["min", "max"]).unstack("family")
        rng_.columns = [f"{f}_{s}" for s, f in rng_.columns]
        n_items = g.drop_duplicates("group").set_index("group")["n_items"]
        R += ["### 2. % changed answers by group mentioned in the item", "",
              md(pd.concat([n_items.rename("n_items"), piv, rng_], axis=1)), ""]
        oc = ov_cat[cat]
        reals = [s for s in order if meta.get(s, {}).get("kind") == "real"]
        rand = [s for s in order if meta.get(s, {}).get("kind") == "random"]
        tab = oc.loc[reals, reals].copy()
        if rand:
            tab["random_mean"] = oc.loc[reals, rand].mean(axis=1)
            rr_pairs = [oc.loc[x, y] for i, x in enumerate(rand) for y in rand[i + 1:]]
            R += [f"### 3. Jaccard of the changed items (random–random pairs: mean {np.nanmean(rr_pairs):.3f}, "
                  f"max {np.nanmax(rr_pairs):.3f})", "", md(tab, 3), ""]
        else:
            R += ["### 3. Jaccard of the changed items", "", md(tab, 3), ""]
        mc = med_df[med_df["category"] == cat]
        R += ["### 4. Baseline loglikelihood margin (top − second of person0 / person1 / unknown)", "",
              md(mc.set_index("set")[["n_changed", "median_margin_changed", "median_margin_unchanged"]], 3), ""]
        bc = bs_df[bs_df["category"] == cat]
        R += [f"### 5. Bootstrap over items ({a.bootstrap} resamples, 95 % percentile interval)", "",
              md(bc.set_index("set").drop(columns=["category"])), ""]
    dm = dec_df.pivot_table(index="decile", columns="set", values="changed_pct")
    R += ["## Changed answers by decile of the baseline margin (all categories; decile 1 = most undecided)", "",
          md(dm[[c for c in order if c in dm.columns]]), ""]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")

    json.dump({"direction": dir_df.to_dict("records"), "groups": grp_df.to_dict("records"),
               "overlap_all": ov_all.to_dict(), "overlap_by_category": {c: m.to_dict() for c, m in ov_cat.items()},
               "margin_changed_vs_unchanged": med_df.to_dict("records"), "margin_deciles": dec_df.to_dict("records"),
               "bootstrap": bs_df.to_dict("records")}, open(out / "results_full.json", "w"), indent=1, default=float)
    files = [items_dir / "sets.json"] + [items_dir / m["table"] for m in meta.values()]
    json.dump({"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "args": vars(a),
               "script_sha256": sha256(__file__), "repo_commit": git_commit(Path(__file__).parent),
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "files": [{"path": str(f), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files],
               "sets": meta}, open(out / "inputs.json", "w"), indent=2)
    print(f"✅ report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
