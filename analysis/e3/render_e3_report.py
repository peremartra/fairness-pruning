#!/usr/bin/env python3
"""
Render clean, structured Markdown reports for E3 from the consolidated CSV tables in results/e3/tables/.

No Drive access needed: reads only the compact CSVs produced by build_e3_tables.py.
Every section and table has exactly one concise explanatory sentence; no prose.

Writes:
  - <out-dir>/E3_RESULTS.md          (Level 1 overview + Level 2 index + Level 3 diagnostics)
  - <out-dir>/units/<unit>.md        (Level 2 homogeneous dossier for each unit)
  - <out-dir>/README.md              (Directory guide, table schemas, and regeneration commands)

Usage:
    python analysis/e3/render_e3_report.py \
        --tables-dir results/e3/tables \
        --out-dir results/e3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fmt_num(v, nd=2):
    if pd.isna(v):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{float(v):.{nd}f}"


def fmt_int(v):
    if pd.isna(v):
        return "-"
    return f"{int(round(float(v))):,}"


def fmt_cnt_pct(ch, n, pct, nd=2):
    if pd.isna(pct):
        return "-"
    if pd.isna(ch) or pd.isna(n):
        return f"{float(pct):.{nd}f}%"
    return f"{int(round(float(ch))):,} / {int(round(float(n))):,} ({float(pct):.{nd}f}%)"


def fmt_ci(lo, hi, nd=2):
    if pd.isna(lo) or pd.isna(hi):
        return "-"
    return f"[{float(lo):.{nd}f}, {float(hi):.{nd}f}]"


def fmt_val_ci(val, lo, hi, nd=2):
    if pd.isna(val):
        return "-"
    s = f"{float(val):.{nd}f}"
    if pd.notna(lo) and pd.notna(hi):
        s += f" [{float(lo):.{nd}f}, {float(hi):.{nd}f}]"
    return s


def md_table(headers, rows, align=None):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    if align:
        sep = []
        for a in align:
            if a == "r":
                sep.append("---:")
            elif a == "c":
                sep.append(":---:")
            else:
                sep.append(":---")
        lines.append("| " + " | ".join(sep) + " |")
    else:
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def render_master(tables: dict, manifest: dict) -> str:
    inv = tables["inventory"]
    rvr = tables["real_vs_random"]
    loc = tables["localization"]
    abl = tables["ablation"]
    l31 = tables["l31_weights"]
    msum = tables["margin_summary"]
    mdec = tables["margin_deciles"]
    k3 = tables["gpu_noise_k3"]
    g3b = tables["gpu_noise_3b"]

    R = [
        "# E3 Results — Neuron Zeroing Interventions Across Models and Languages",
        "",
        f"Consolidated from `{len(manifest['source_files'])}` verified Drive files (`inputs_manifest.json`, generated `{manifest['generated_at_utc']}`); primary measure is changed answers (`changed / n_items`, `flips_pct`) against the unpruned baseline, while BBQ/EsBBQ bias scores and accuracy transitions are secondary.",
        "",
        "## Navigation",
        "",
        "- **Level 1 — Cross-Unit Overview:** [1.1 Inventory](#11-experiment-inventory) · [1.2 Real vs. Random Controls](#12-primary-result-selected-vs-layer-matched-random-neurons) · [1.3 Localization (All Items)](#13-target-category-concentration-all-items-pooled) · [1.4 Localization (Disambiguated Only)](#14-target-category-concentration-disambiguated-items-only)",
        "- **Level 2 — Unit Dossiers:** [2. Index of Units](#2-unit-dossiers-level-2)",
        "- **Level 3 — Diagnostics & Controls:** [3.1 Single-Neuron Ablation (`L31·6950`)](#31-single-neuron-ablation-llama-31-8b-en-l316950) · [3.2 Baseline Margin Mechanism](#32-baseline-indecision-mechanism-margin-deciles) · [3.3 Hardware Noise Controls](#33-hardware-noise-controls-l4-vs-a100) · [3.4 Provenance & Tables](#34-provenance-and-source-tables)",
        "",
        "---",
        "",
        "## 1. Cross-Unit Overview (Level 1)",
        "",
        "### 1.1 Experiment Inventory",
        "",
        "All eight E3 units covered in this report, with dataset size, number of intervened sets, tested K values, and GPU hardware.",
        "",
    ]

    rows = []
    for _, r in inv.iterrows():
        link = f"[`{r['unit']}`](units/{r['unit']}.md)"
        rnd_str = str(int(r["n_random_sets"]))
        if int(r["n_random_sensitivity_sets"]) > 0:
            rnd_str += f" (+{int(r['n_random_sensitivity_sets'])} L4)"
        rows.append([
            link,
            r["model"],
            r["language"],
            r["benchmark"],
            f"{int(r['n_items_total']):,} ({int(r['n_categories'])} cats)",
            int(r["n_real_sets"]),
            int(r["n_ablation_sets"]),
            rnd_str,
            r["k_values"],
            "Yes" if bool(r["has_per_item_analysis"]) else "Summary only",
            r["gpus_intervened"],
        ])
    R.append(md_table(
        ["Unit", "Model", "Lang", "Bench", "Items", "Real", "Abl.", "Random", "K", "Per-item", "Intervened GPUs"],
        rows,
        ["l", "l", "c", "c", "r", "r", "r", "r", "c", "c", "l"],
    ))
    R.append("")

    # 1.2 Primary Result: Real vs Random
    R += [
        "### 1.2 Primary Result: Selected vs. Layer-Matched Random Neurons",
        "",
        "Changed answers (`changed / n_items`, `%`) in the target category and across the remaining categories for selected neurons versus random neurons drawn uniformly from the exact same layers (primary comparisons first; sensitivity rows marked).",
        "",
    ]
    rows = []
    for _, r in rvr.sort_values(["is_primary", "benchmark", "unit", "comparison"], ascending=[False, True, True, True]).iterrows():
        tag = "" if bool(r["is_primary"]) else " *(sens.)*"
        rand_cnt = f"{int(r['rand_target_changed_min']):,}–{int(r['rand_target_changed_max']):,}"
        rand_pct = f"{r['rand_target_flips_min']:.2f}–{r['rand_target_flips_max']:.2f}% (mean {r['rand_target_flips_mean']:.2f}%)"
        rows.append([
            f"[`{r['unit']}`](units/{r['unit']}.md)",
            f"{r['comparison']}{tag}",
            r["target_category"],
            fmt_cnt_pct(r["real_target_changed"], r["target_n_items"], r["real_target_flips_pct"]),
            f"{rand_cnt} ({rand_pct})",
            f"{r['ratio_real_over_rand_mean']:.2f}×",
            r["rank_of_real"],
            f"{r['real_other_mean_flips_pct']:.2f}% vs. {r['rand_other_mean_flips_pct']:.2f}%",
            f"{r['real_specificity_ratio']:.2f}× vs. {r['rand_specificity_ratio_mean']:.2f}×",
            f"{r['real_gpu']} / {r['random_gpus']}",
        ])
    R.append(md_table(
        ["Unit", "Comparison", "Target Cat.", "Selected Flips (Target)", "Random Flips (Target, min–max)", "Real/Rand", "Rank", "Other Cats (Real vs. Rand)", "Ratio (Real vs. Rand)", "GPUs (Real / Rand)"],
        rows,
        ["l", "l", "l", "r", "r", "r", "c", "r", "r", "l"],
    ))
    R.append("")

    # 1.3 Target-Category Concentration (All Items Pooled)
    R += [
        "### 1.3 Target-Category Concentration (All Items Pooled)",
        "",
        "Changed answers for every real set (`K = 5`, `K = 20`, and EsBBQ grid sets) in its target category versus the unweighted mean of the other categories, with 95% item-bootstrap CI, concentration ratio, rank, and total benchmark flips.",
        "",
    ]
    loc_real = loc[loc["kind"] == "real"].copy()
    unit_order = {u: i for i, u in enumerate(inv["unit"].tolist())}
    loc_real["_uord"] = loc_real["unit"].map(unit_order)
    loc_real = loc_real.sort_values(["_uord", "k", "set_category", "set"], ascending=[True, False, True, True])
    rows = []
    for _, r in loc_real.iterrows():
        rows.append([
            f"[`{r['unit']}`](units/{r['unit']}.md)",
            f"`{r['set']}`",
            r["target_category"],
            int(r["k"]),
            f"{int(r['last_layer_neurons'])}/{int(r['n_neurons'])}",
            fmt_cnt_pct(r["target_changed"], r["target_n_items"], r["target_flips_pct"]),
            fmt_ci(r["target_flips_ci_low"], r["target_flips_ci_high"]),
            f"{r['other_mean_flips_pct']:.2f}%",
            f"{r['specificity_ratio']:.2f}×",
            f"{int(r['target_rank'])}/{int(r['n_categories'])}",
            fmt_cnt_pct(r["all_items_changed"], r["all_items_total"], r["all_items_flips_pct"]),
        ])
    R.append(md_table(
        ["Unit", "Set", "Target Cat.", "K", "Last Layer", "Target Flips", "95% CI (%)", "Other Mean", "Ratio", "Rank", "All Benchmark Flips"],
        rows,
        ["l", "l", "l", "r", "c", "r", "c", "r", "r", "c", "r"],
    ))
    R.append("")

    # 1.4 Disambiguated-only concentration
    R += [
        "### 1.4 Target-Category Concentration (Disambiguated Items Only)",
        "",
        "Same localization ratios and ranks restricted to disambiguated items (`context = disambig`, matching the EsBBQ summary view reported in `R1_exp.md`).",
        "",
        "<details>",
        "<summary>Expand disambiguated-only localization table (all real sets with per-item logs)</summary>",
        "",
    ]
    loc_dis = loc_real[loc_real["disambig_target_flips_pct"].notna()]
    rows = []
    for _, r in loc_dis.iterrows():
        rows.append([
            f"[`{r['unit']}`](units/{r['unit']}.md)",
            f"`{r['set']}`",
            r["target_category"],
            int(r["k"]),
            f"{r['disambig_target_flips_pct']:.2f}%",
            f"{r['disambig_other_mean_flips_pct']:.2f}%",
            f"{r['disambig_specificity_ratio']:.2f}×",
            f"{int(r['disambig_target_rank'])}/{int(r['n_categories'])}",
        ])
    R.append(md_table(
        ["Unit", "Set", "Target Cat.", "K", "Disambig Target (%)", "Disambig Other Mean (%)", "Ratio", "Rank"],
        rows,
        ["l", "l", "l", "r", "r", "r", "r", "c"],
    ))
    R += ["", "</details>", "", "---", ""]

    # Level 2 Index
    R += [
        "## 2. Unit Dossiers (Level 2)",
        "",
        "Each unit dossier follows the same layout: unit summary, real-vs-random controls, full `set × category` flip matrix, neuron/item Jaccard overlap, secondary BBQ/EsBBQ bias and accuracy metrics, baseline margin mechanism, and demographic group/topic breakdowns.",
        "",
    ]
    rows = []
    for _, r in inv.iterrows():
        u = r["unit"]
        scope = "Full per-item analysis (flips, overlap, secondary metrics, margins, groups"
        if r["benchmark"] == "EsBBQ":
            scope += ", topics)"
        elif bool(r["has_per_item_analysis"]):
            scope += ")"
        else:
            scope = "Summary-only (`per_set/*_summary.json`: flips by category, neuron overlap, disambig bias delta)"
        rows.append([
            f"[`units/{u}.md`](units/{u}.md)",
            r["model"],
            r["language"],
            r["benchmark"],
            f"{int(r['n_real_sets'])} real / {int(r['n_random_sets'])} rand / {int(r['n_ablation_sets'])} abl",
            scope,
        ])
    R.append(md_table(["Dossier", "Model", "Lang", "Bench", "Sets", "Contents"], rows, ["l", "l", "c", "c", "l", "l"]))
    R += ["", "---", ""]

    # Level 3 Diagnostics
    R += [
        "## 3. Diagnostics and Controls (Level 3)",
        "",
        "### 3.1 Single-Neuron Ablation (`Llama-3.1-8B EN`, `L31·6950`)",
        "",
        "Exploratory decomposition of `real_race5` and `real_religion5` into their shared neuron (`abl_shared1` = `L31·6950`) and the remaining four neurons of each set (`abl_race4`, `abl_religion4`), showing changed answers across contexts and Jaccard overlap with the original Top-5 sets.",
        "",
    ]
    abl_all = abl[abl["context"] == "all"].copy()
    abl_amb = abl[abl["context"] == "ambig"].set_index(["set", "category"])
    abl_dis = abl[abl["context"] == "disambig"].set_index(["set", "category"])
    piv_sets = ["real_race5", "real_religion5", "abl_shared1", "abl_race4", "abl_religion4"]
    rows = []
    for sid in piv_sets:
        sub = abl_all[abl_all["set"] == sid].set_index("category")
        if sub.empty:
            continue
        neurons = sub["neurons"].iloc[0]
        rows.append([
            f"`{sid}`",
            neurons,
            fmt_cnt_pct(sub.loc["Religion", "changed"], sub.loc["Religion", "n_items"], sub.loc["Religion", "flips_pct"]),
            f"{abl_amb.loc[(sid, 'Religion'), 'flips_pct']:.2f}% / {abl_dis.loc[(sid, 'Religion'), 'flips_pct']:.2f}%",
            fmt_cnt_pct(sub.loc["Race_ethnicity", "changed"], sub.loc["Race_ethnicity", "n_items"], sub.loc["Race_ethnicity", "flips_pct"]),
            f"{abl_amb.loc[(sid, 'Race_ethnicity'), 'flips_pct']:.2f}% / {abl_dis.loc[(sid, 'Race_ethnicity'), 'flips_pct']:.2f}%",
            fmt_cnt_pct(sub.loc["Gender_identity", "changed"], sub.loc["Gender_identity", "n_items"], sub.loc["Gender_identity", "flips_pct"]),
            f"{sub.loc['Religion', 'jaccard_vs_real_race5_allctx']:.2f} / {sub.loc['Race_ethnicity', 'jaccard_vs_real_race5_allctx']:.2f}",
            f"{sub.loc['Religion', 'jaccard_vs_real_religion5_allctx']:.2f} / {sub.loc['Race_ethnicity', 'jaccard_vs_real_religion5_allctx']:.2f}",
        ])
    R.append(md_table(
        ["Set", "Neurons (L31)", "Religion (All)", "Religion (Amb/Dis)", "Race (All)", "Race (Amb/Dis)", "Gender (All)", "Jacc vs. Race5 (Rel/Race)", "Jacc vs. Rel5 (Rel/Race)"],
        rows,
        ["l", "l", "r", "c", "r", "c", "r", "c", "c"],
    ))
    R += [
        "",
        "L2 norm of the `down_proj` column and within-layer percentiles in Layer 31 (14,336 neurons) for all nine neurons of `real_race5` and `real_religion5`.",
        "",
    ]
    rows = []
    for _, r in l31.iterrows():
        rows.append([
            f"`L31·{int(r['neuron'])}`",
            f"`{r['set']}`",
            f"{r['col_norm']:.4f}",
            f"{r['col_norm_pct']:.2f}%",
            f"{r['Religion_score']:.3f} ({r['Religion_score_pct']:.2f}%)",
            f"{r['Race_score']:.3f} ({r['Race_score_pct']:.2f}%)",
            f"{r['Gender_score']:.3f} ({r['Gender_score_pct']:.2f}%)",
        ])
    R.append(md_table(
        ["Neuron", "Ablation Subset", "`down_proj` Norm", "Norm Pct.", "Religion Score (Pct.)", "Race Score (Pct.)", "Gender Score (Pct.)"],
        rows,
        ["l", "l", "r", "r", "r", "r", "r"],
    ))
    R.append("")

    # 3.2 Baseline Margin Mechanism
    R += [
        "### 3.2 Baseline Indecision Mechanism (Margin Deciles)",
        "",
        "Median baseline log-likelihood margin (`top − second` across person0/person1/unknown) of changed versus unchanged items across all categories, alongside flip rate in decile 1 (most undecided) versus decile 10 (most confident) for `K = 20` real sets (and `Gender·5` in Salamandra-7B ES).",
        "",
    ]
    msum_all = msum[(msum["category"] == "ALL") & (msum["kind"] == "real")]
    d1 = mdec[(mdec["decile"] == 1) & (mdec["kind"] == "real")].set_index(["unit", "set"])["changed_pct"]
    d5 = mdec[(mdec["decile"] == 5) & (mdec["kind"] == "real")].set_index(["unit", "set"])["changed_pct"]
    d10 = mdec[(mdec["decile"] == 10) & (mdec["kind"] == "real")].set_index(["unit", "set"])["changed_pct"]
    key_sets = {"real_gender20", "real_religion20", "real_race20", "exp10", "exp11", "exp13", "exp14", "exp5"}
    rows = []
    for _, r in msum_all[msum_all["set"].isin(key_sets)].iterrows():
        u, s = r["unit"], r["set"]
        rows.append([
            f"[`{u}`](units/{u}.md)",
            f"`{s}`",
            fmt_int(r["n_changed"]),
            fmt_num(r["median_margin_changed"], 3),
            fmt_num(r["median_margin_unchanged"], 3),
            fmt_num(d1.get((u, s), np.nan), 2) + "%",
            fmt_num(d5.get((u, s), np.nan), 2) + "%",
            fmt_num(d10.get((u, s), np.nan), 2) + "%",
        ])
    R.append(md_table(
        ["Unit", "Set", "Changed Items (All)", "Median Margin (Changed)", "Median Margin (Unchanged)", "Decile 1 Flips", "Decile 5 Flips", "Decile 10 Flips"],
        rows,
        ["l", "l", "r", "r", "r", "r", "r", "r"],
    ))
    R.append("")

    # 3.3 Hardware Noise Controls
    R += [
        "### 3.3 Hardware Noise Controls (L4 vs. A100)",
        "",
        "K3 baseline rerun comparison for `Llama-3.1-8B ES` (unpruned model evaluated on A100 vs. L4 across all 27,320 EsBBQ items), establishing the cross-GPU hardware flip floor.",
        "",
    ]
    rows = []
    for _, r in k3.iterrows():
        rows.append([
            f"**{r['category']}**" if r["category"] == "TOTAL" else r["category"],
            fmt_cnt_pct(r["ambig_changed"], r["ambig_n"], r["ambig_changed_pct"]),
            fmt_cnt_pct(r["disambig_changed"], r["disambig_n"], r["disambig_changed_pct"]),
            fmt_cnt_pct(r["total_changed"], r["total_n"], r["total_changed_pct"]),
        ])
    R.append(md_table(
        ["EsBBQ Category", "Ambiguous Flips (L4 vs. A100)", "Disambiguated Flips (L4 vs. A100)", "Total Flips (L4 vs. A100)"],
        rows,
        ["l", "r", "r", "r"],
    ))
    R += [
        "",
        "Sensitivity comparison of the identical random set `rand_gender20_seed0` in `Llama-3.2-3B EN` evaluated on A100-SXM4-80GB versus L4 against the A100-SXM4-40GB baseline.",
        "",
    ]
    rows = []
    for _, r in g3b.iterrows():
        rows.append([
            r["category"],
            fmt_cnt_pct(r["a100_changed"], r["n_items"], r["a100_flips_pct"]),
            fmt_cnt_pct(r["l4_changed"], r["n_items"], r["l4_flips_pct"]),
            f"{r['diff_l4_minus_a100_pp']:+.2f} pp",
        ])
    R.append(md_table(
        ["BBQ Category", "A100-80GB Flips", "L4 Flips", "L4 − A100 (pp)"],
        rows,
        ["l", "r", "r", "r"],
    ))
    R.append("")

    # 3.4 Provenance
    R += [
        "### 3.4 Provenance and Source Tables",
        "",
        "Every number above is generated deterministically from the consolidated CSV files in `results/e3/tables/`, each traced by SHA-256 to its source Drive file in `tables/inputs_manifest.json`.",
        "",
    ]
    rows = []
    for fname, info in sorted(manifest["outputs"].items()):
        rows.append([f"[`tables/{fname}`](tables/{fname})", f"{info['rows']:,}", info["cols"], f"{info['bytes']:,} B"])
    R.append(md_table(["CSV Table", "Rows", "Columns", "Size"], rows, ["l", "r", "r", "r"]))
    R.append("")
    return "\n".join(R)


def render_unit(unit: str, tables: dict) -> str:
    inv = tables["inventory"]
    uinfo = inv[inv["unit"] == unit].iloc[0]
    loc = tables["localization"]
    rvr = tables["real_vs_random"]
    flips = tables["flips_by_category"]
    ov = tables["overlap"]
    sec = tables["secondary"]
    msum = tables["margin_summary"]
    mdec = tables["margin_deciles"]
    grp = tables["groups"]
    top = tables["topics"]
    prov = tables["provenance"]

    u_loc = loc[loc["unit"] == unit].copy()
    u_rvr = rvr[rvr["unit"] == unit].copy()
    u_flips = flips[flips["unit"] == unit].copy()
    u_ov = ov[ov["unit"] == unit].copy()
    u_sec = sec[sec["unit"] == unit].copy()
    u_msum = msum[msum["unit"] == unit].copy()
    u_mdec = mdec[mdec["unit"] == unit].copy()
    u_grp = grp[grp["unit"] == unit].copy()
    u_top = top[top["unit"] == unit].copy()
    u_prov = prov[prov["unit"] == unit].copy()

    R = [
        f"# E3 Unit Dossier — `{unit}` ({uinfo['model']}, {uinfo['language']}, {uinfo['benchmark']})",
        "",
        f"[← Back to E3 Overview](../E3_RESULTS.md) · Benchmark: **{uinfo['benchmark']}** ({int(uinfo['n_items_total']):,} items, {int(uinfo['n_categories'])} categories) · Source: `{uinfo['analysis_dir']}`.",
        "",
    ]

    sec_num = 1
    # 1. Localization summary
    R += [
        f"## {sec_num}. Set Summary and Target-Category Localization",
        "",
        "Changed answers (`pred != base_pred`, `changed / n_items`, `%`) for each real and ablation set in its target category versus the mean of the other categories, with layer distribution and GPU.",
        "",
    ]
    sec_num += 1
    prov_idx = u_prov.set_index("set")
    rows = []
    for _, r in u_loc.sort_values(["kind", "k", "set"], ascending=[False, False, True]).iterrows():
        lprof = prov_idx.loc[r["set"], "layer_profile"] if r["set"] in prov_idx.index else ""
        rows.append([
            f"`{r['set']}`",
            r["kind"],
            r["target_category"] if pd.notna(r["target_category"]) else "-",
            fmt_int(r["k"]),
            fmt_cnt_pct(r["target_changed"], r["target_n_items"], r["target_flips_pct"]),
            fmt_ci(r["target_flips_ci_low"], r["target_flips_ci_high"]),
            (f"{r['other_mean_flips_pct']:.2f}%" if pd.notna(r["other_mean_flips_pct"]) else "-"),
            (f"{r['specificity_ratio']:.2f}×" if pd.notna(r["specificity_ratio"]) else "-"),
            (f"{int(r['target_rank'])}/{int(r['n_categories'])}" if pd.notna(r["target_rank"]) else "-"),
            fmt_cnt_pct(r["all_items_changed"], r["all_items_total"], r["all_items_flips_pct"]),
            r["gpu"],
            lprof,
        ])
    R.append(md_table(
        ["Set", "Kind", "Target Cat.", "K", "Target Flips", "95% CI (%)", "Other Mean", "Ratio", "Rank", "All Benchmark Flips", "GPU", "Layer Profile"],
        rows,
        ["l", "c", "l", "r", "r", "c", "r", "r", "c", "r", "l", "l"],
    ))
    R.append("")

    # 2. Real vs Random (if any)
    if not u_rvr.empty:
        R += [
            f"## {sec_num}. Selected vs. Layer-Matched Random Neurons",
            "",
            "Direct contrast of selected sets against uniform random neuron sets drawn from the exact same layers.",
            "",
        ]
        sec_num += 1
        rows = []
        for _, r in u_rvr.iterrows():
            rand_str = f"{int(r['rand_target_changed_min']):,}–{int(r['rand_target_changed_max']):,} ({r['rand_target_flips_min']:.2f}–{r['rand_target_flips_max']:.2f}%, mean {r['rand_target_flips_mean']:.2f}%)"
            rows.append([
                r["comparison"],
                r["target_category"],
                fmt_cnt_pct(r["real_target_changed"], r["target_n_items"], r["real_target_flips_pct"]),
                rand_str,
                f"{r['ratio_real_over_rand_mean']:.2f}×",
                r["rank_of_real"],
                f"{r['real_other_mean_flips_pct']:.2f}% vs. {r['rand_other_mean_flips_pct']:.2f}%",
                f"{r['real_specificity_ratio']:.2f}× vs. {r['rand_specificity_ratio_mean']:.2f}×",
                f"{r['real_gpu']} / {r['random_gpus']}",
            ])
        R.append(md_table(
            ["Comparison", "Target Cat.", "Selected Flips", "Random Flips (min–max, mean)", "Real/Rand", "Rank", "Other Cats (Real vs. Rand)", "Ratio (Real vs. Rand)", "GPUs"],
            rows,
            ["l", "l", "r", "r", "r", "c", "r", "r", "l"],
        ))
        R.append("")

    # 3. Full Set x Category Matrix (all items pooled + ambig/disambig in details)
    R += [
        f"## {sec_num}. Changed Answers Across All Categories (`Set × Category`)",
        "",
        "Full matrix of changed answers (`pred != base_pred`, `changed (flips_pct)`) across every benchmark category (all items pooled; random families summarized by mean).",
        "",
    ]
    sec_num += 1
    f_all = u_flips[u_flips["context"] == "all"].copy()
    cats = sorted(f_all["category"].unique().tolist())
    cat_n = f_all.drop_duplicates("category").set_index("category")["n_items"].to_dict()
    nonrand_sets = u_loc["set"].tolist()
    # Build columns: nonrand sets + random families
    rnd_f = f_all[f_all["kind"] == "random"].copy()
    fams = []
    if not rnd_f.empty:
        rnd_f["family"] = rnd_f["set"].str.rsplit("_seed", n=1).str[0]
        fams = sorted(rnd_f["family"].unique().tolist())

    headers = ["Category", "Items"] + [f"`{s}`" for s in nonrand_sets] + [f"`{f}` (mean)" for f in fams]
    align = ["l", "r"] + ["r"] * (len(nonrand_sets) + len(fams))
    f_idx = f_all.set_index(["set", "category"])
    rows = []
    for c in cats:
        row = [c, f"{int(cat_n[c]):,}"]
        for s in nonrand_sets:
            if (s, c) in f_idx.index:
                x = f_idx.loc[(s, c)]
                row.append(f"{int(x['changed']):,} ({x['flips_pct']:.2f}%)")
            else:
                row.append("-")
        for f in fams:
            sub = rnd_f[(rnd_f["family"] == f) & (rnd_f["category"] == c)]
            row.append(f"{sub['flips_pct'].mean():.2f}% [{sub['flips_pct'].min():.2f}–{sub['flips_pct'].max():.2f}]")
        rows.append(row)
    R.append(md_table(headers, rows, align))
    R.append("")

    # Ambig / Disambig breakdown in details if present
    f_ctx = u_flips[u_flips["context"].isin(["ambig", "disambig"]) & u_flips["kind"].isin(["real", "ablation"])]
    if not f_ctx.empty:
        R += [
            "<details>",
            "<summary>Expand ambiguous vs. disambiguated flips by category (real & ablation sets)</summary>",
            "",
            "Per-category flip percentage split by `ambig / disambig` context for each real and ablation set.",
            "",
        ]
        fc_idx = f_ctx.set_index(["set", "category", "context"])["flips_pct"]
        rows = []
        for c in cats:
            row = [c]
            for s in nonrand_sets:
                a = fc_idx.get((s, c, "ambig"), np.nan)
                d = fc_idx.get((s, c, "disambig"), np.nan)
                row.append(f"{fmt_num(a, 2)}% / {fmt_num(d, 2)}%")
            rows.append(row)
        R.append(md_table(["Category"] + [f"`{s}`" for s in nonrand_sets], rows, ["l"] + ["c"] * len(nonrand_sets)))
        R += ["", "</details>", ""]

    # 4. Overlap
    if not u_ov.empty:
        R += [
            f"## {sec_num}. Neuron and Changed-Item Overlap Between Sets",
            "",
            "Pairwise count of shared zeroed neurons, neuron Jaccard, and changed-item Jaccard (all items and within each set's target category).",
            "",
        ]
        sec_num += 1
        rows = []
        for _, r in u_ov.sort_values(["n_shared_neurons", "items_jaccard_all"], ascending=[False, False]).iterrows():
            rows.append([
                f"`{r['set_a']}` vs. `{r['set_b']}`",
                f"{int(r['n_shared_neurons'])} / {int(r['n_neurons_a'])},{int(r['n_neurons_b'])}",
                fmt_num(r["neuron_jaccard"], 3),
                fmt_num(r["items_jaccard_all"], 3),
                f"{r['target_cat_a']}: {fmt_num(r['items_jaccard_target_a'], 3)}",
                f"{r['target_cat_b']}: {fmt_num(r['items_jaccard_target_b'], 3)}",
                r["shared_neurons"] if pd.notna(r["shared_neurons"]) and str(r["shared_neurons"]) else "-",
            ])
        R.append(md_table(
            ["Pair", "Shared Neurons", "Neuron Jaccard", "Items Jacc (All)", "Items Jacc (Target A)", "Items Jacc (Target B)", "Shared Neuron IDs"],
            rows,
            ["l", "c", "r", "r", "r", "r", "l"],
        ))
        R.append("")

    # 5. Secondary metrics
    if not u_sec.empty:
        R += [
            f"## {sec_num}. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)",
            "",
            "Secondary benchmark metrics for target categories (`disambig` and `ambig` contexts, state-based S/A/U transitions); descriptive and not primary localization measures.",
            "",
        ]
        sec_num += 1
        is_es = (uinfo["benchmark"] == "EsBBQ")
        u_sec_ctx = u_sec[u_sec["context"].isin(["ambig", "disambig"])].sort_values(["category", "set", "context"])
        rows = []
        for _, r in u_sec_ctx.iterrows():
            if is_es:
                bias_str = (
                    fmt_val_ci(r["esbbq_disamb_bias_delta"], r["esbbq_disamb_bias_delta_ci_low"], r["esbbq_disamb_bias_delta_ci_high"])
                    if r["context"] == "disambig"
                    else fmt_val_ci(r["esbbq_amb_bias_delta"], r["esbbq_amb_bias_delta_ci_low"], r["esbbq_amb_bias_delta_ci_high"])
                )
                bbq_eq = fmt_val_ci(r["disamb_bias_delta"], r["disamb_bias_delta_ci_low"], r["disamb_bias_delta_ci_high"]) if r["context"] == "disambig" else "-"
                rows.append([
                    r["category"],
                    f"`{r['set']}`",
                    r["context"],
                    fmt_cnt_pct(r["changed"], r["n_items"], r["changed_pct"]),
                    f"{fmt_int(r['A_to_S'])} / {fmt_int(r['S_to_A'])}",
                    fmt_num(r["net_towards_stereotype_pct"], 2) + "%",
                    f"{fmt_int(r['correct_to_incorrect'])} / {fmt_int(r['incorrect_to_correct'])}",
                    bias_str,
                    bbq_eq,
                ])
            else:
                bias_str = (
                    fmt_val_ci(r["disamb_bias_delta"], r["disamb_bias_delta_ci_low"], r["disamb_bias_delta_ci_high"])
                    if r["context"] == "disambig"
                    else "-"
                )
                rows.append([
                    r["category"],
                    f"`{r['set']}`",
                    r["context"],
                    fmt_cnt_pct(r["changed"], r["n_items"], r["changed_pct"]),
                    f"{fmt_int(r['A_to_S'])} / {fmt_int(r['S_to_A'])}",
                    (fmt_num(r["net_towards_stereotype_pct"], 2) + "%") if pd.notna(r["net_towards_stereotype_pct"]) else "-",
                    f"{fmt_int(r['correct_to_incorrect'])} / {fmt_int(r['incorrect_to_correct'])}",
                    bias_str,
                ])
        if is_es:
            R.append(md_table(
                ["Category", "Set", "Context", "Flips", "A→S / S→A", "Net→Stereo", "Corr→Inc / Inc→Corr", "EsBBQ Bias Δ (95% CI)", "BBQ-Formula Disamb Δ (95% CI)"],
                rows,
                ["l", "l", "c", "r", "c", "r", "c", "r", "r"],
            ))
        else:
            R.append(md_table(
                ["Category", "Set", "Context", "Flips", "A→S / S→A", "Net→Stereo", "Corr→Inc / Inc→Corr", "Disamb Bias Δ (95% CI)"],
                rows,
                ["l", "l", "c", "r", "c", "r", "c", "r"],
            ))
        R.append("")

    # 6. Margin mechanism
    if not u_msum.empty and not u_mdec.empty:
        R += [
            f"## {sec_num}. Baseline Log-Likelihood Margin Mechanism",
            "",
            "Median baseline margin (`top − second`) of changed versus unchanged items (`ALL` categories) and flip rate across baseline-margin deciles 1–10 (decile 1 = most undecided).",
            "",
        ]
        sec_num += 1
        rows = []
        for _, r in u_msum[u_msum["category"] == "ALL"].iterrows():
            rows.append([
                f"`{r['set']}`",
                r["kind"],
                fmt_num(r["n_changed"], 1) if r["kind"] == "random_family_mean" else fmt_int(r["n_changed"]),
                fmt_num(r["median_margin_changed"], 3),
                fmt_num(r["median_margin_unchanged"], 3),
            ])
        R.append(md_table(
            ["Set", "Kind", "Changed Items (All)", "Median Margin (Changed)", "Median Margin (Unchanged)"],
            rows,
            ["l", "c", "r", "r", "r"],
        ))
        R.append("")

        piv_dec = u_mdec.pivot_table(index="decile", columns="set", values="changed_pct")
        dec_cols = [c for c in nonrand_sets if c in piv_dec.columns] + [c for c in piv_dec.columns if c not in nonrand_sets]
        edges = u_mdec.drop_duplicates("decile").set_index("decile")
        rows = []
        for d in range(1, 11):
            row = [str(d), f"[{edges.loc[d, 'margin_min']:.3f}, {edges.loc[d, 'margin_max']:.3f}]"]
            for c in dec_cols:
                row.append(f"{piv_dec.loc[d, c]:.2f}%" if pd.notna(piv_dec.loc[d, c]) else "-")
            rows.append(row)
        R.append(md_table(
            ["Decile", "Margin Range"] + [f"`{c}`" for c in dec_cols],
            rows,
            ["c", "c"] + ["r"] * len(dec_cols),
        ))
        R.append("")

    # 7. Granular breakdowns (groups & topics) in collapsible blocks
    if not u_grp.empty:
        R += [
            f"## {sec_num}. Granular Breakdowns (Demographic Groups and Topics)",
            "",
            "Flip rates disaggregated by mentioned demographic group (`answer_info`) and, for EsBBQ, by item topic (`relevant_social_value`).",
            "",
            "<details>",
            "<summary>Expand demographic group breakdown (target categories)</summary>",
            "",
        ]
        sec_num += 1
        for cat in sorted(u_grp["category"].unique()):
            sub = u_grp[u_grp["category"] == cat]
            piv = sub.pivot_table(index="group", columns="set", values="changed_pct")
            n_col = sub.drop_duplicates("group").set_index("group")["n_items"]
            cols = [c for c in nonrand_sets if c in piv.columns]
            R += [f"### {cat} — Groups", ""]
            rows = []
            for g in piv.index:
                rows.append([g, f"{int(n_col.loc[g]):,}"] + [f"{piv.loc[g, c]:.2f}%" if pd.notna(piv.loc[g, c]) else "-" for c in cols])
            R.append(md_table(["Group", "Items"] + [f"`{c}`" for c in cols], rows, ["l", "r"] + ["r"] * len(cols)))
            R.append("")
        R += ["</details>", ""]

    if not u_top.empty:
        R += [
            "<details>",
            "<summary>Expand EsBBQ topic breakdown (`relevant_social_value`)</summary>",
            "",
        ]
        for cat in sorted(u_top["category"].unique()):
            sub = u_top[u_top["category"] == cat]
            piv = sub.pivot_table(index="topic", columns="set", values="changed_pct")
            n_col = sub.drop_duplicates("topic").set_index("topic")["n_items"]
            cols = [c for c in nonrand_sets if c in piv.columns]
            R += [f"### {cat} — Topics", ""]
            rows = []
            for t in piv.index:
                rows.append([t, f"{int(n_col.loc[t]):,}"] + [f"{piv.loc[t, c]:.2f}%" if pd.notna(piv.loc[t, c]) else "-" for c in cols])
            R.append(md_table(["Topic", "Items"] + [f"`{c}`" for c in cols], rows, ["l", "r"] + ["r"] * len(cols)))
            R.append("")
        R += ["</details>", ""]

    return "\n".join(R)


def render_readme(manifest: dict) -> str:
    R = [
        "# results/e3 — Consolidated E3 Experiment Results",
        "",
        "Structured, reproducible summary of all E3 neuron-zeroing interventions across models and languages, built directly from the verified per-item analysis outputs on Google Drive.",
        "",
        "## Entry Points",
        "",
        "- **[`E3_RESULTS.md`](E3_RESULTS.md)** — Master navigable report (Level 1 cross-unit overview, Level 2 unit index, Level 3 diagnostics & hardware noise controls).",
        "- **[`units/`](units/)** — Eight homogeneous unit dossiers (one per model × language).",
        "- **[`tables/`](tables/)** — Compact consolidated CSVs (single source of truth for both the Markdown reports and paper tables/figures) plus [`inputs_manifest.json`](tables/inputs_manifest.json) with SHA-256 hashes of every source file read on Drive.",
        "",
        "## Measurement Conventions",
        "",
        "- **Primary measure:** changed answers against the unpruned baseline (`changed / n_items`, `flips_pct` over 3-way predictions `{person0, person1, unknown}`, pooled across `ambig + disambig` items per category), contrasting selected neurons with layer-matched random neurons and measuring target-category concentration (`ratio = target / mean(other categories)`, `rank`).",
        "- **Secondary measures:** BBQ/EsBBQ bias score deltas (with 95% item-bootstrap CIs), stereotype transitions (`A→S`, `S→A`), and accuracy transitions (`correct↔incorrect`).",
        "",
        "## How to Regenerate",
        "",
        "1. Rebuild CSV tables from Google Drive (Colab CPU runtime with Drive mounted; read-only on Drive):",
        "```bash",
        "python analysis/e3/build_e3_tables.py \\",
        "  --drive-root /content/drive/MyDrive/fair_pruning \\",
        "  --out-dir results/e3/tables",
        "```",
        "",
        "2. Re-render all Markdown files locally from `results/e3/tables/` (no Drive required):",
        "```bash",
        "python analysis/e3/render_e3_report.py \\",
        "  --tables-dir results/e3/tables \\",
        "  --out-dir results/e3",
        "```",
        "",
        f"Last built from Drive: `{manifest['generated_at_utc']}` (`{len(manifest['source_files'])}` source files verified).",
        "",
    ]
    return "\n".join(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", default="results/e3/tables")
    ap.add_argument("--out-dir", default="results/e3")
    args = ap.parse_args()

    tdir = Path(args.tables_dir)
    out = Path(args.out_dir)
    units_dir = out / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(tdir / "inputs_manifest.json", encoding="utf-8"))
    tables = {
        "inventory": pd.read_csv(tdir / "e3_inventory.csv"),
        "localization": pd.read_csv(tdir / "e3_localization_summary.csv"),
        "real_vs_random": pd.read_csv(tdir / "e3_real_vs_random.csv"),
        "flips_by_category": pd.read_csv(tdir / "e3_flips_by_category.csv"),
        "overlap": pd.read_csv(tdir / "e3_neuron_overlap.csv"),
        "secondary": pd.read_csv(tdir / "e3_secondary_metrics.csv"),
        "ablation": pd.read_csv(tdir / "e3_ablation_l31_6950.csv"),
        "l31_weights": pd.read_csv(tdir / "e3_l31_weights.csv"),
        "margin_summary": pd.read_csv(tdir / "e3_margin_summary.csv"),
        "margin_deciles": pd.read_csv(tdir / "e3_margin_deciles.csv"),
        "groups": pd.read_csv(tdir / "e3_groups.csv"),
        "topics": pd.read_csv(tdir / "e3_esbbq_topics.csv"),
        "gpu_noise_k3": pd.read_csv(tdir / "e3_gpu_noise_k3.csv"),
        "gpu_noise_3b": pd.read_csv(tdir / "e3_gpu_noise_3b.csv"),
        "provenance": pd.read_csv(tdir / "e3_provenance.csv"),
    }

    (out / "E3_RESULTS.md").write_text(render_master(tables, manifest), encoding="utf-8")
    (out / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    for u in tables["inventory"]["unit"].tolist():
        (units_dir / f"{u}.md").write_text(render_unit(u, tables), encoding="utf-8")

    print(f"Rendered {out / 'E3_RESULTS.md'}, {out / 'README.md'}, and {len(tables['inventory'])} unit dossiers in {units_dir}")


if __name__ == "__main__":
    main()
