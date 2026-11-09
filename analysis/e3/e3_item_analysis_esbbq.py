#!/usr/bin/env python3
"""
E3 step 2 for EsBBQ (Spanish unit, for example Salamandra-7B ES) from the compact tables of extract_items_esbbq.py.

Counterpart of e3_item_analysis.py (BBQ): same five questions, same tables, same report layout. Descriptive only:
no thresholds, no verdicts. Each set (neurons zeroed) is compared item by item with the baseline (unmodified model).
  1. direction  - where do the changed answers go (towards the pro-stereotype person, the other person, "unknown")?
  2. groups     - within a category, which groups (answer_info) and which topics (relevant_social_value) carry the changes?
  3. overlap    - do different sets change the same items (Jaccard of the changed items)?
  4. margin     - are the changed items those where the unpruned model was undecided (small loglikelihood margin)?
  5. bootstrap  - item bootstrap intervals for the % of changed answers and for the change in three bias measures:
       disamb_bias_delta       change of the BBQ-script measure 100*(2*biased/non-unknown - 1) on disambiguated items
                               (the one used in the BBQ analysis, for comparison across units);
       esbbq_disamb_bias_delta change of the EsBBQ measure acc(pro-stereo) - acc(anti-stereo), in points (as lm-eval reports);
       esbbq_amb_bias_delta    change of the EsBBQ ambiguous measure (incorrect pro-stereo - incorrect anti-stereo) / ambiguous items, in points.
Extra table (section 0): % of changed answers of every category for every set, so that the target category
can be read against the others. This unit has no random control sets.

States: S = the answer is the pro-stereotype person (see extract_items_esbbq.py), A = the other person, U = "unknown".

Usage (laptop or Colab CPU; input = the `items/` folder written by step 1):
    python analysis/e3/e3_item_analysis_esbbq.py --items-dir "<Drive>/fair_pruning/results/esbbq_zeroed_7b/items"
Options: --out-dir, --bootstrap 1000, --seed 0, --target-categories Religion Age Gender
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

EMPTY_LABEL = "(empty label)"   # group label missing in answer_info
STATES = ["S", "A", "U"]        # S = pro-stereotype person, A = the other person, U = "unknown"
N_TASKS = 10


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


def load(items_dir, allow_partial=False):
    sets_json = json.load(open(items_dir / "sets.json", encoding="utf-8"))
    assert sets_json.get("benchmark") == "esbbq", "sets.json is not from extract_items_esbbq.py"
    meta = sets_json["sets"]
    tables = {}
    for sid, m in meta.items():
        if m.get("problems"):
            print(f"⚠️  {sid}: the extraction reported problems: {m['problems']}")
        if len(m["tasks"]) != N_TASKS and not allow_partial:
            raise SystemExit(f"{sid}: {len(m['tasks'])} of {N_TASKS} tasks (use --allow-partial only for tests)")
        # keep_default_na=False: group labels such as "NA" / "None" must stay strings (otherwise NaN != NaN breaks the check)
        t = pd.read_csv(items_dir / m["table"], keep_default_na=False, na_values=[""]).set_index("item_id").sort_index()
        for c in ("group0", "group1"):
            t[c] = t[c].fillna(EMPTY_LABEL).astype(str)      # some items have an empty group label: keep them, visible
        t["relevant_social_value"] = t["relevant_social_value"].fillna(EMPTY_LABEL).astype(str)
        t["state"] = state(t)
        lls = t[["ll0", "ll1", "ll_unk"]].to_numpy()
        srt = np.sort(lls, axis=1)
        t["margin"] = srt[:, -1] - srt[:, -2]
        tables[sid] = t
    b0 = next(iter(tables.values()))
    for c in ("group0", "group1"):
        n_empty = b0.loc[b0[c] == EMPTY_LABEL, "category"].value_counts()
        if len(n_empty):
            print(f"note: items with empty {c}: " + ", ".join(f"{k}={v}" for k, v in n_empty.items()))
    base_ids = [s for s, m in meta.items() if m["kind"] == "baseline"]
    assert len(base_ids) == 1, f"expected one baseline, found {base_ids}"
    base = base_ids[0]
    for sid, t in tables.items():
        assert t.index.equals(tables[base].index), f"{sid}: items differ from the baseline"
        cols = ["category", "context", "question_type", "relevant_social_value", "group0", "group1", "target"]
        bad = [c for c in cols if not (t[c] == tables[base][c]).all()]
        assert not bad, f"{sid}: item metadata differs from the baseline in {bad}"
    order = [base] + [s for s in meta if s != base]       # order of the manifest (Religion, Age, Gender)
    return meta, tables, base, order


def cats_of(tables, base):
    return sorted(tables[base]["category"].unique())


def disamb_bias(df):
    """Measure of the BBQ analysis script (disambiguated items)."""
    d = df[df["context"] == "disambig"]
    nonunk = (d["pred"] < 2).sum()
    return 100 * (2 * d["biased"].sum() / nonunk - 1) if nonunk else np.nan


def esbbq_disamb_bias(df):
    """acc(pro-stereo) - acc(anti-stereo) on disambiguated items, in points (esbbq_utils.bias_score_disambig_agg)."""
    d = df[df["context"] == "disambig"]
    pro, anti = d["question_type"] == "pro-stereo", d["question_type"] == "anti-stereo"
    if pro.sum() == 0 or anti.sum() == 0:
        return np.nan
    return 100 * (d.loc[pro, "correct"].mean() - d.loc[anti, "correct"].mean())


def esbbq_amb_bias(df):
    """(incorrect pro-stereo - incorrect anti-stereo) / ambiguous items, in points (esbbq_utils.bias_score_ambig_agg)."""
    a = df[df["context"] == "ambig"]
    wrong = (a["correct"] == 0) & (a["pred"] < 2)
    return 100 * ((wrong & (a["biased"] == 1)).sum() - (wrong & (a["biased"] == 0)).sum()) / len(a) if len(a) else np.nan


# ── 0. changed answers in every category ────────────────────────────────────
def changed_by_category(meta, tables, base, order, cats):
    b = tables[base]
    rows = []
    for sid in order:
        if sid == base:
            continue
        ch = tables[sid]["pred"] != b["pred"]
        for cat in cats:
            m = b["category"] == cat
            rows.append({"set": sid, "kind": meta[sid]["kind"], "category": cat, "n_items": int(m.sum()),
                         "changed": int((ch & m).sum()), "changed_pct": 100 * ch[m].mean()})
    return pd.DataFrame(rows)


# ── 1. direction ────────────────────────────────────────────────────────────
def direction(meta, tables, base, order, cats):
    b = tables[base]
    rows, trans = [], []
    for sid in order:
        if sid == base:
            continue
        t = tables[sid]
        for cat in cats:
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
def groups(meta, tables, base, order, cats):
    b = tables[base]
    rows = []
    for cat in cats:
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


def topics(meta, tables, base, order, cats):
    """% of changed answers by relevant_social_value (topic of the item) within each category."""
    b = tables[base]
    rows = []
    for cat in cats:
        bc = b[b["category"] == cat]
        for v in sorted(bc["relevant_social_value"].unique()):
            m_idx = bc.index[bc["relevant_social_value"] == v]
            for sid in order:
                if sid == base:
                    continue
                ch = (tables[sid].loc[m_idx, "pred"] != b.loc[m_idx, "pred"])
                rows.append({"set": sid, "kind": meta[sid]["kind"], "category": cat, "topic": v,
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
def margin(meta, tables, base, order, cats):
    b = tables[base]
    dec = pd.qcut(b["margin"].rank(method="first"), 10, labels=False) + 1   # deciles of the baseline margin, 1 = smallest
    edges = b.groupby(dec)["margin"].agg(["min", "max"])
    rows, med = [], []
    for sid in order:
        if sid == base:
            continue
        ch = tables[sid]["pred"] != b["pred"]
        for cat in cats + ["ALL"]:
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
def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def bootstrap(meta, tables, base, order, cats, n_boot, seed):
    b = tables[base]
    rng = np.random.default_rng(seed)
    rows = []

    def arrays(df, idx):
        """Per-item arrays of one set restricted to the index `idx`."""
        d = df.loc[idx]
        return {"biased": d["biased"].to_numpy(), "nonunk": (d["pred"] < 2).to_numpy(), "correct": d["correct"].to_numpy(),
                "pro": (d["question_type"] == "pro-stereo").to_numpy(), "anti": (d["question_type"] == "anti-stereo").to_numpy(),
                "wrong_pro": ((d["correct"] == 0) & (d["pred"] < 2) & (d["biased"] == 1)).to_numpy(),
                "wrong_anti": ((d["correct"] == 0) & (d["pred"] < 2) & (d["biased"] == 0)).to_numpy()}

    def s_bbq(a, idx):          # BBQ-script measure, disambiguated items
        return 100 * (2 * _ratio(a["biased"][idx].sum(1), a["nonunk"][idx].sum(1)) - 1)

    def s_dis(a, idx):          # acc(pro) - acc(anti), points
        pro, anti = a["pro"][idx], a["anti"][idx]
        return 100 * (_ratio((a["correct"][idx] * pro).sum(1), pro.sum(1)) - _ratio((a["correct"][idx] * anti).sum(1), anti.sum(1)))

    def s_amb(a, idx):          # (incorrect pro - incorrect anti) / ambiguous items, points
        return 100 * (a["wrong_pro"][idx].sum(1) - a["wrong_anti"][idx].sum(1)) / idx.shape[1]

    for cat in cats:
        bc = b[b["category"] == cat]
        dis = bc.index[bc["context"] == "disambig"]
        amb = bc.index[bc["context"] == "ambig"]
        idx_d = rng.integers(0, len(dis), size=(n_boot, len(dis)), dtype=np.int32)    # same resamples for every set
        idx_m = rng.integers(0, len(amb), size=(n_boot, len(amb)), dtype=np.int32)
        idx_a = rng.integers(0, len(bc), size=(n_boot, len(bc)), dtype=np.int32)
        bd, bm = arrays(b, dis), arrays(b, amb)
        base_bbq, base_dis, base_amb = s_bbq(bd, idx_d), s_dis(bd, idx_d), s_amb(bm, idx_m)
        for sid in order:
            if sid == base:
                continue
            t = tables[sid]
            td, tm = arrays(t, dis), arrays(t, amb)
            d_bbq, d_dis, d_amb = s_bbq(td, idx_d) - base_bbq, s_dis(td, idx_d) - base_dis, s_amb(tm, idx_m) - base_amb
            ch = (t.loc[bc.index, "pred"].to_numpy() != bc["pred"].to_numpy())
            chp = 100 * ch[idx_a].mean(1)
            tc = t[t["category"] == cat]
            q = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5))
            rows.append({"set": sid, "kind": meta[sid]["kind"], "category": cat,
                         "disamb_bias_delta": disamb_bias(tc) - disamb_bias(bc),
                         "disamb_bias_delta_ci_low": q(d_bbq)[0], "disamb_bias_delta_ci_high": q(d_bbq)[1],
                         "esbbq_disamb_bias_delta": esbbq_disamb_bias(tc) - esbbq_disamb_bias(bc),
                         "esbbq_disamb_bias_delta_ci_low": q(d_dis)[0], "esbbq_disamb_bias_delta_ci_high": q(d_dis)[1],
                         "esbbq_amb_bias_delta": esbbq_amb_bias(tc) - esbbq_amb_bias(bc),
                         "esbbq_amb_bias_delta_ci_low": q(d_amb)[0], "esbbq_amb_bias_delta_ci_high": q(d_amb)[1],
                         "changed_pct": 100 * ch.mean(),
                         "changed_pct_ci_low": np.percentile(chp, 2.5), "changed_pct_ci_high": np.percentile(chp, 97.5)})
    return pd.DataFrame(rows)


# ── report ──────────────────────────────────────────────────────────────────
def md(df, nd=2):
    return df.to_markdown(floatfmt=f".{nd}f") if hasattr(df, "to_markdown") else df.round(nd).to_string()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-categories", nargs="*", default=["Religion", "Age", "Gender"])
    ap.add_argument("--allow-partial", action="store_true", help="tests only: tables with fewer than the ten tasks")
    a = ap.parse_args()

    items_dir = Path(a.items_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) if a.out_dir else items_dir.parent / "analysis" / f"{stamp}_items"
    out.mkdir(parents=True, exist_ok=True)

    meta, tables, base, order = load(items_dir, a.allow_partial)
    cats = cats_of(tables, base)
    targets = [c for c in a.target_categories if c in cats]
    unknown = [c for c in a.target_categories if c not in cats]
    if unknown:
        print(f"note: target categories not in the data (skipped): {unknown}; categories present: {cats}")
    print(f"{len(tables)} sets ({base} + {len(order) - 1}); {len(cats)} categories; writing to {out}")

    chg_df = changed_by_category(meta, tables, base, order, cats)
    dir_df, trans_df = direction(meta, tables, base, order, cats)
    grp_df = groups(meta, tables, base, order, cats)
    top_df = topics(meta, tables, base, order, cats)
    ov_all = overlap(tables, base, order)
    ov_cat = {c: overlap(tables, base, order, c) for c in targets}
    med_df, dec_df = margin(meta, tables, base, order, cats)
    bs_df = bootstrap(meta, tables, base, order, cats, a.bootstrap, a.seed)

    chg_df.to_csv(out / "changed_by_category.csv", index=False)
    dir_df.to_csv(out / "direction.csv", index=False)
    trans_df.to_csv(out / "transitions.csv", index=False)
    grp_df.to_csv(out / "groups.csv", index=False)
    top_df.to_csv(out / "topics.csv", index=False)
    ov_all.to_csv(out / "overlap_changed_items_all.csv")
    for c, m in ov_cat.items():
        m.to_csv(out / f"overlap_changed_items_{c}.csv")
    med_df.to_csv(out / "margin_changed_vs_unchanged.csv", index=False)
    dec_df.to_csv(out / "margin_deciles.csv", index=False)
    bs_df.to_csv(out / "bootstrap.csv", index=False)

    # report
    reals = [s for s in order if s != base]
    R = [f"# E3 per-item analysis — {json.load(open(items_dir / 'sets.json'))['unit']} (EsBBQ)", "",
         f"Generated {dt.datetime.now().isoformat(timespec='seconds')}. Baseline: `{base}`. Real sets: {len(reals)}. "
         "This unit has no random control sets, so no comparison with random neurons is possible. Descriptive only. "
         "The sets of one category come from the same ranking of neurons, so the smaller ones are contained in the larger ones "
         "and a high Jaccard between them is expected.", "",
         "Sets: " + "; ".join(f"`{s}` = {meta[s]['name']} ({meta[s]['n_neurons']} neurons)" for s in reals), ""]
    R += ["## 0. % of changed answers in every category (answer differs from the baseline)", "",
          md(pd.concat([chg_df.drop_duplicates("category").set_index("category")["n_items"],
                        chg_df.pivot_table(index="category", columns="set", values="changed_pct")[reals]], axis=1)), ""]
    for cat in targets:
        R += [f"## {cat}", ""]
        d = dir_df[(dir_df["category"] == cat) & (dir_df["context"] != "all")]
        cols = ["changed_pct", "to_stereotyped", "to_other_person", "to_unknown", "net_towards_stereotype_pct",
                "correct_to_incorrect", "incorrect_to_correct"]
        R += ["### 1. Direction of the changed answers (counts; net = (A→S − S→A) / items, %)", "",
              md(d.set_index(["set", "context"])[["n_items"] + cols]), ""]
        g = grp_df[grp_df["category"] == cat]
        piv = g.pivot_table(index="group", columns="set", values="changed_pct")[reals]
        n_items = g.drop_duplicates("group").set_index("group")["n_items"]
        if len(piv) > 2:
            R += ["### 2. % changed answers by group mentioned in the item", "", md(pd.concat([n_items.rename("n_items"), piv], axis=1)), ""]
        else:
            R += ["### 2. % changed answers by group mentioned in the item",
                  "", f"Not shown: every item of this category mentions the same {len(piv)} groups ({', '.join(piv.index)}), so the rows are identical to the category total.", ""]
        tp = top_df[top_df["category"] == cat]
        tpiv = tp.pivot_table(index="topic", columns="set", values="changed_pct")[reals]
        R += ["### 2b. % changed answers by topic of the item (relevant_social_value)", "",
              md(pd.concat([tp.drop_duplicates("topic").set_index("topic")["n_items"].rename("n_items"), tpiv], axis=1)), ""]
        R += ["### 3. Jaccard of the changed items", "", md(ov_cat[cat], 3), ""]
        mc = med_df[med_df["category"] == cat]
        mcols = ["n_changed", "median_margin_changed", "median_margin_unchanged"]
        R += ["### 4. Baseline loglikelihood margin (top − second of person0 / person1 / unknown)", "",
              md(mc.set_index("set")[mcols], 3), ""]
        bc = bs_df[bs_df["category"] == cat].set_index("set").drop(columns=["category", "kind"])
        R += [f"### 5. Bootstrap over items ({a.bootstrap} resamples, 95 % percentile interval)", "", md(bc), ""]
    dm = dec_df.pivot_table(index="decile", columns="set", values="changed_pct")[reals]
    R += ["## Changed answers by decile of the baseline margin (all categories; decile 1 = most undecided)", "", md(dm), ""]
    (out / "report.md").write_text("\n".join(R), encoding="utf-8")

    json.dump({"changed_by_category": chg_df.to_dict("records"), "direction": dir_df.to_dict("records"),
               "groups": grp_df.to_dict("records"), "topics": top_df.to_dict("records"), "overlap_all": ov_all.to_dict(),
               "overlap_by_category": {c: m.to_dict() for c, m in ov_cat.items()},
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
