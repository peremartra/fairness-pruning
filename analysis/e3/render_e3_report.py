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
        "**Why this table:** summarizes the full experimental scope across all eight E3 units (models, languages, benchmarks, set counts, tested K values, and GPU hardware) so every downstream table maps to its exact evaluation setup.",
        "",
    ]

    rows = []
    for _, r in inv.iterrows():
        link = f"[`{r['unit']}`](units/{r['unit']}.md)"
        rnd_str = str(int(r["n_random_sets"]))
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
    R += [
        "",
        "*Legend — **Real**: selected top-K neuron sets; **Abl.**: exploratory ablation subsets derived from real sets (see 3.1); **Random**: layer-matched uniform random control sets; **K**: number of zeroed neurons per set; **Per-item**: whether full item-level logs and bootstrap analysis are available (`Summary only` = aggregated per-category summary JSONs only).*",
        "",
    ]

    # 1.2 Primary Result: Real vs Random
    R += [
        "### 1.2 Primary Result: Selected vs. Layer-Matched Random Neurons",
        "",
        "**Why this check:** tests the paper's core localization claim—whether zeroing BiasScore-selected neurons changes substantially more target-category answers than zeroing the same number of random neurons drawn uniformly from the exact same MLP layers (all primary comparisons are hardware-matched on A100).",
        "",
    ]
    rows = []
    rvr_primary = rvr[rvr["is_primary"]].sort_values(["benchmark", "unit", "comparison"], ascending=[True, True, True])
    for _, r in rvr_primary.iterrows():
        rand_cnt = f"{int(r['rand_target_changed_min']):,}–{int(r['rand_target_changed_max']):,}"
        rand_pct = f"{r['rand_target_flips_min']:.2f}–{r['rand_target_flips_max']:.2f}% (mean {r['rand_target_flips_mean']:.2f}%)"
        rows.append([
            f"[`{r['unit']}`](units/{r['unit']}.md)",
            r["comparison"],
            r["target_category"],
            fmt_cnt_pct(r["real_target_changed"], r["target_n_items"], r["real_target_flips_pct"]),
            f"{rand_cnt} ({rand_pct})",
            f"{r['ratio_real_over_rand_mean']:.2f}×",
            r["rank_of_real"],
            f"{r['real_gpu']} / {r['random_gpus']}",
        ])
    R.append(md_table(
        ["Unit", "Comparison", "Target Cat.", "Selected Flips (Target)", "Random Flips (Target, min–max)", "Real/Rand", "Rank", "GPUs (Real / Rand)"],
        rows,
        ["l", "l", "l", "r", "r", "r", "c", "l"],
    ))
    R += [
        "",
        "*Legend — **Selected Flips (Target)**: target-category items whose 3-way prediction (`person0`/`person1`/`unknown`) differs from baseline (`changed / n_items`, `%`); **Random Flips (Target, min–max)**: range across the layer-matched random seeds reported as `min–max changed items (min–max% (mean %))` over the same `n_items` target denominator (e.g., `41–76` = lowest and highest seed counts, `0.72–1.34%` = corresponding percentages, `mean 0.93%` = average across seeds); **Real/Rand**: `Selected % / Random mean %`; **Rank**: ordinal position of the selected set among selected + random (`1` = highest).*",
        "",
        "**Key takeaways:**",
        "- **Selected sets exceed all random controls (9/9):** across all 9 primary comparisons, none of the 55 layer-matched A100 random seeds reaches the selected set's flip rate (`Rank 1` in every case), with `Real/Rand` ratios ranging from **6.10× to 32.77×**.",
        "- **Layer-matched random controls flip few items:** random-seed means remain between **0.62% and 1.46%** (per-seed range `0.25%–3.17%`), indicating that the layer profile and `K` alone account for only a small share of the observed changes.",
        "- **Differences by model scale and category:** the largest flip rates and ratios appear in `Gender` for the 7–8B models (`Llama-3.1-8B EN`: `24.00%` / `25.68×`; `Salamandra-7B EN`: `16.36%` / `17.58×`; `Salamandra-7B ES` at `K = 5`: `20.34%` / `32.77×`), whereas `Llama-3.2-3B EN` and `Qwen3-4B-Base EN` show smaller absolute rates (`6.58%–11.08%`, ratios `6.10×–9.43×`).",
        "",
    ]

    # 1.3 Target-Category Concentration (All Items Pooled)
    R += [
        "### 1.3 Target-Category Concentration (All Items Pooled)",
        "",
        "**Why this check:** tests whether the answer changes caused by selected neurons concentrate in their target demographic category rather than moving all benchmark categories equally, comparing `Target Flips` against `Other Mean` across every real set (`K = 5`, `K = 20`, and EsBBQ grid sets).",
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
            fmt_cnt_pct(r["all_items_changed"], r["all_items_total"], r["all_items_flips_pct"]),
        ])
    R.append(md_table(
        ["Unit", "Set", "Target Cat.", "K", "Last Layer", "Target Flips", "95% CI (%)", "Other Mean", "Ratio", "All Benchmark Flips"],
        rows,
        ["l", "l", "l", "r", "c", "r", "c", "r", "r", "r"],
    ))
    R += [
        "",
        "*Legend — **Last Layer**: zeroed neurons located in the final MLP layer out of `K`; **Target Flips**: changed answers (`pred != base_pred`) in the target category; **95% CI (%)**: paired item-bootstrap percentile interval (`B = 1,000`); **Other Mean**: unweighted mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **All Benchmark Flips**: total changed answers across the full benchmark.*",
        "",
        "**Key takeaways:**",
        "- **`Gender·20` changes much more in Gender than in the other categories (7 of 8 units):** across every Llama and Salamandra unit (EN and ES), the flip rate in Gender is **2.13× to 3.82×** higher than the mean of the other categories (EN: `Llama-3.1-8B` `3.82×`, `Salamandra-7B` `3.79×`, `Llama-3.2-1B` `3.04×`, `Llama-3.2-3B` `2.13×`; ES: `Salamandra-7B` `3.38×`, `Llama-3.2-3B` `2.51×`, `Llama-3.1-8B` `2.44×`).",
        "- **`Religion` shows a smaller gap over other categories, and `Race` shows very little:** at `K = 20` outside Qwen, `Religion` changes **1.15×–2.70×** more than the other-category mean (highest in `Salamandra-7B EN` `2.48×` and `Llama-3.2-1B EN` `2.70×`), while `Race` stays at **1.04×–1.82×** (`1.04×–1.17×` in ES).",
        "- **`Qwen3-4B-Base EN` changes all categories roughly equally:** across all 6 sets, target-category flip rates are close to the other-category mean (`Ratio` **0.83×–1.05×**); although selected neurons flip more items than random controls (Table 1.2), the effect is not focused on the target category.",
        "- **`K = 5` is enough in some models but too small in others:** in `Llama-3.1-8B EN`, `Salamandra-7B EN`, and `Salamandra-7B ES`, `Gender·5` already produces nearly the same Gender flip rate as `Gender·20` (`23.33%` vs. `24.00%`, `16.10%` vs. `16.36%`, `20.34%` vs. `19.60%`) with less spillover (`Ratio` `4.83×–5.45×`). By contrast, in `Llama-3.2-3B EN/ES`, `Llama-3.2-1B EN`, and `Llama-3.1-8B ES`, `Gender·5` barely moves Gender (`1.32%–2.34%`, `Ratio < 1×`) and the gap only emerges at `K = 20`.",
        "- **Last-layer predominance (especially in `Race` and `Religion`):** most selected neurons in `Race` and `Religion` come from the final MLP layer (often `17/20–20/20`), whereas `Gender` draws more neurons from earlier layers (`6/20–17/20` in the last layer).",
        "",
    ]

    # 1.4 Disambiguated-only concentration
    R += [
        "### 1.4 Target-Category Concentration (Disambiguated Items Only)",
        "",
        "**Why this check:** verifies that target-category concentration still holds when restricted to disambiguated items alone (`context = disambig`, where the prompt supplies a clear factual answer), confirming the effect is not confined to ambiguous items.",
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
        ])
    R.append(md_table(
        ["Unit", "Set", "Target Cat.", "K", "Disambig Target (%)", "Disambig Other Mean (%)", "Ratio"],
        rows,
        ["l", "l", "l", "r", "r", "r", "r"],
    ))
    R += [
        "",
        "*Legend — **Disambig Target / Other Mean**: changed-answer `%` restricted to disambiguated items (`direction.csv`, S/A/U state changes); **Ratio**: `Disambig Target / Disambig Other Mean`.*",
        "",
        "</details>",
        "",
        "**Key takeaways:**",
        "- **Lower absolute flip rates than pooled, but changes persist on disambiguated items:** restricting to items with a clear factual answer reduces absolute flip rates compared with Table 1.3 (e.g., `Llama-3.1-8B EN Gender·20`: `14.67%` vs. `24.00%`; `Salamandra-7B EN Gender·20`: `8.43%` vs. `16.36%`; `Salamandra-7B ES Gender·5`: `16.47%` vs. `20.34%`), yet substantial changes remain.",
        "- **Cross-category and cross-model patterns match Table 1.3:** `Gender·20` keeps ratios of **2.00×–3.91×** across Llama and Salamandra (EN and ES), and `Gender·5` reaches **4.17×–5.10×** in the three 7–8B units where `K = 5` is active; `Qwen3-4B-Base EN` stays near or below `1×` (**0.46×–1.03×**).",
        "- **Slightly higher ratios for `Religion` on disambiguated items:** excluding ambiguous items raises the `Religion` ratio in several units (`Llama-3.1-8B EN`: `1.54× → 1.96×` at `K = 20`, `2.77× → 3.63×` at `K = 5`; `Salamandra-7B EN`: `2.48× → 3.36×`, `2.76× → 3.94×`; `Llama-3.1-8B ES`: `1.32× → 1.78×`, `1.53× → 2.01×`).",
        "",
        "---",
        "",
    ]

    # Level 2 Index
    R += [
        "## 2. Unit Dossiers (Level 2)",
        "",
        "**Why this table:** indexes the eight homogeneous per-unit dossiers where full `set × category` matrices, neuron/item overlaps, secondary bias/accuracy metrics, margin deciles, and demographic group/topic breakdowns are stored.",
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
    R += [
        "",
        "*Legend — **Sets**: number of selected (`real`), layer-matched random (`rand`), and exploratory ablation (`abl`) sets in each unit dossier. The `3 abl` sets in `llama-3.1-8b_en` decompose `real_race5` and `real_religion5` around their single shared neuron `L31·6950`: `abl_shared1` (`L31·6950` alone, 1 neuron), `abl_race4` (`real_race5` without `L31·6950`, 4 neurons), and `abl_religion4` (`real_religion5` without `L31·6950`, 4 neurons); see [Section 3.1](#31-single-neuron-ablation-llama-31-8b-en-l316950).*",
        "",
        "---",
        "",
    ]

    # Level 3 Diagnostics
    R += [
        "## 3. Diagnostics and Controls (Level 3)",
        "",
        "### 3.1 Single-Neuron Ablation (`Llama-3.1-8B EN`, `L31·6950`)",
        "",
        "**Why this check:** tests whether the shared effect of `real_race5` and `real_religion5` in `Llama-3.1-8B EN` is spread across all five neurons or concentrated in their single shared neuron (`abl_shared1` = `L31·6950`), comparing changed answers and item-level overlap (`In` = share of row flips present in the original Top-5; `Reprod.` = share of original Top-5 flips reproduced) across Religion and Race.",
        "",
    ]
    abl_all = abl[abl["context"] == "all"].copy()
    abl_counts = abl_all.set_index(["set", "category"])["changed"].to_dict()

    def _overlap_pcts(sub_df, sid: str, ref_sid: str, j_col: str):
        if sid == ref_sid:
            return "—", "—"
        in_vals, rep_vals = [], []
        for cat in ("Religion", "Race_ethnicity"):
            n_s = int(sub_df.loc[cat, "changed"])
            n_ref = int(abl_counts[(ref_sid, cat)])
            j = float(sub_df.loc[cat, j_col])
            inter = round((j / (1.0 + j)) * (n_s + n_ref)) if j > 0 else 0
            denom = n_s + n_ref - inter
            if denom > 0:
                assert abs((inter / denom) - j) < 1e-10, f"Overlap recovery mismatch for {ref_sid} {cat}"
            in_vals.append(f"{100.0 * inter / n_s:.1f}%" if n_s else "-")
            rep_vals.append(f"{100.0 * inter / n_ref:.1f}%" if n_ref else "-")
        return " / ".join(in_vals), " / ".join(rep_vals)

    piv_sets = ["real_race5", "real_religion5", "abl_shared1", "abl_race4", "abl_religion4"]
    rows = []
    for sid in piv_sets:
        sub = abl_all[abl_all["set"] == sid].set_index("category")
        if sub.empty:
            continue
        neurons = sub["neurons"].iloc[0]
        in_r5, rep_r5 = _overlap_pcts(sub, sid, "real_race5", "jaccard_vs_real_race5_allctx")
        in_rel5, rep_rel5 = _overlap_pcts(sub, sid, "real_religion5", "jaccard_vs_real_religion5_allctx")
        rows.append([
            f"`{sid}`",
            neurons,
            fmt_cnt_pct(sub.loc["Religion", "changed"], sub.loc["Religion", "n_items"], sub.loc["Religion", "flips_pct"]),
            fmt_cnt_pct(sub.loc["Race_ethnicity", "changed"], sub.loc["Race_ethnicity", "n_items"], sub.loc["Race_ethnicity", "flips_pct"]),
            in_r5,
            rep_r5,
            in_rel5,
            rep_rel5,
        ])
    R.append(md_table(
        ["Set", "Neurons (L31)", "Religion (All)", "Race (All)", "In Race5 (Rel/Race)", "Reprod. Race5 (Rel/Race)", "In Rel5 (Rel/Race)", "Reprod. Rel5 (Rel/Race)"],
        rows,
        ["l", "l", "r", "r", "c", "c", "c", "c"],
    ))
    R += [
        "",
        "*Legend — **(All)**: pooled changed answers (`changed / n_items`, `%`, `pred != base_pred`); **In Race5 / In Rel5 (Rel/Race)**: percentage of the row's changed answers that also changed in `real_race5` / `real_religion5` (`shared / row_changed`, ordered `Religion / Race`); **Reprod. Race5 / Reprod. Rel5 (Rel/Race)**: percentage of `real_race5` / `real_religion5` changed answers reproduced by the row (`shared / original_changed`, ordered `Religion / Race`); `—`: self-comparison.*",
        "",
        "**Key takeaways:**",
        "- **One neuron (`L31·6950`) accounts for most of `real_race5` and `real_religion5` with few new flips:** against `real_race5`, **95.7%** (`111/116`) of `L31·6950`'s Religion flips and **86.6%** (`348/402`) of its Race flips were already changed by the full 5-neuron set (only `4.3%` and `13.4%` new), reproducing **91.0%** and **83.9%** of the original flips; against `real_religion5`, **75.0% / 83.8%** were in the original and **73.7% / 79.1%** are reproduced.",
        "- **The other four Race neurons (`abl_race4`) have little effect:** although `63.2%` (Religion) and `65.2%` (Race) of their flips overlap with `real_race5`, the absolute counts are small (`1.58%` and `2.25%`) and reproduce only **9.8% / 24.3%** of `real_race5`.",
        "- **The other four Religion neurons (`abl_religion4`) show partial, less aligned changes:** they flip `6.17%` in Religion and `2.54%` in Race, with roughly half not present in `real_religion5` (`In`: **54.1% / 61.1%**) and reproducing **33.9% / 25.1%** of `real_religion5`. *(Exploratory single-model check; no dedicated 1-neuron random control.)*",
        "",
        "**Why this check:** tests whether `L31·6950`'s disproportionate effect is merely due to abnormally large output weights (`down_proj` column norm) rather than its differential activation (`BiasScore`). Below are the `down_proj` L2 norms, BiasScores, and within-layer percentiles across all 14,336 neurons of Layer 31 for the nine neurons in `real_race5` and `real_religion5` (all nine have below-median weight norms, percentiles 0.34%–25.96%, ruling out a weight-magnitude artifact).",
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
    R += [
        "",
        "*Legend — **`down_proj` Norm**: L2 norm of the neuron's output column in Layer 31; **Norm Pct.**: weight-norm percentile within Layer 31's 14,336 neurons (`100%` = largest, `0%` = smallest, so `8.20%` means 91.8% of neurons have larger weights); **Score (Pct.)**: differential-activation BiasScore and BiasScore percentile within Layer 31 (`100%` = highest) for each category.*",
        "",
        "**Key takeaways:**",
        "- **Small output weights (`down_proj`):** `L31·6950`'s impact is not explained by large output weights; its **weight-norm percentile** (`8.20%`, norm `0.5622`) sits well below the Layer 31 median, and all nine neurons fall in the bottom quarter by weight size (**weight-norm percentiles `0.34%–25.96%`**).",
        "- **A high `BiasScore` alone does not guarantee large flip counts:** the four neurons in `abl_race4` reach nearly the same **Race `BiasScore` percentiles** as `L31·6950` (**`99.97%–99.99%`** vs. `99.99%`), yet together produce far fewer flips than `L31·6950` alone.",
        "- **`L31·6950` scores near the top in all three categories:** its **`BiasScore` percentiles** are `99.99%` (Religion), `99.99%` (Race), and `99.71%` (Gender), unlike narrower neurons such as `L31·1495` (`99.99%` in Race vs. `58.80%` in Religion). One possible hypothesis (not directly measured on BBQ activations) is that this cross-category breadth causes `L31·6950` to fire on more evaluation items than the other Top-5 neurons.",
        "",
    ]

    # 3.2 Baseline Margin Mechanism
    R += [
        "### 3.2 Baseline Indecision Mechanism (Margin Deciles)",
        "",
        "**Why this check:** tests whether neuron zeroing disrupts confident predictions at random or acts selectively at the decision boundary by flipping only items where the unpruned model was already undecided. Below are the median baseline log-likelihood margins (`top − second` across `person0`/`person1`/`unknown`) of changed versus unchanged items, and the flip rate in Decile 1 (most undecided 10%), Decile 5, and Decile 10 (most confident 10%, `0.00%` flips in 22 of 23 rows, max `0.07%`) for `K = 20` real sets (plus `Gender·5` in `Salamandra-7B ES`).",
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
    R += [
        "",
        "*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) for changed vs. unchanged items; **Decile 1 / 5 / 10 Flips**: flip `%` in baseline-margin deciles (`1` = smallest margin / most undecided, `10` = largest margin / most confident).*",
        "",
        "**Key takeaways:**",
        "- **Flips occur on items where the baseline model was already undecided:** across all 23 rows (7 units), the median baseline margin (`top − second`) of changed items is **0.094–0.375**, versus **1.000–2.750** for unchanged items (~4× to ~11× smaller).",
        "- **High-confidence baseline predictions barely move:** flip rates are **19.88%–42.64%** in Decile 1 (`32%–43%` for all `K = 20` sets), drop to **0.11%–10.43%** in Decile 5, and reach **0.00% in 22 of 23 rows** in Decile 10 (max `0.07%` in `llama-3.2-3b_es exp11`).",
        "- **Consistent mechanism across families and languages:** the same pattern holds on both BBQ (EN) and EsBBQ (ES), including `Qwen3-4B-Base` (which has larger baseline margins, `~2.7`, yet still flips `38%–42%` in Decile 1 vs. `0.00%` in Decile 10).",
        "",
    ]

    # 3.3 Hardware Noise Controls
    R += [
        "### 3.3 Hardware Noise Controls (L4 vs. A100)",
        "",
        "**Why this check:** quantifies the pure cross-GPU hardware flip floor when running the exact same unpruned model (`Llama-3.1-8B ES`, K3 rerun across all 27,320 EsBBQ items) on A100 versus L4, showing why primary comparisons must stay on matching GPU hardware.",
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
        "*Legend — **Flips (L4 vs. A100)**: unpruned baseline items whose 3-way prediction changes solely due to running on L4 versus A100 hardware (`changed / n_items`, `%`).*",
        "",
        "**Key takeaways:**",
        "- **Switching GPU hardware alone changes `1.85%` of answers:** evaluating the exact same unpruned model on A100 versus L4 alters **506 / 27,320** EsBBQ items (`2.41%` ambiguous, `1.59%` disambiguated), ranging from **1.40% (`SES`) to 2.65% (`LGBTQIA`)** across categories.",
        "- **Comparable in size to small-set and random effects:** this `1.85%` hardware floor is on the same scale as layer-matched random controls (`0.6%–1.5%`) and close to small `K = 5` sets in `Llama-3.1-8B ES` (`2.34%–3.70%`), motivating matched GPU hardware for baseline and intervened runs.",
        "",
        "**Why this check:** measures how much L4-vs-A100 hardware noise inflates a random control set's flip rate when evaluated on L4 against an A100 baseline (identical `rand_gender20_seed0` set in `Llama-3.2-3B EN` run on A100-SXM4-80GB vs. L4).",
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
    R += [
        "",
        "*Legend — **A100-80GB / L4 Flips**: changed answers against the A100-40GB baseline for the exact same random neuron set on each GPU; **L4 − A100 (pp)**: percentage-point hardware difference.*",
        "",
        "**Key takeaways:**",
        "- **Running on L4 against an A100 baseline inflates flips in 10 of 11 categories:** the identical random set shows **+0.13 pp to +0.63 pp** more flips on L4 than on A100 (`Gender_identity`: `1.23% → 1.60%`, `+0.37 pp`; `Physical_appearance`: `1.59% → 2.22%`, `+0.63 pp`; sole exception `Religion`: `-0.25 pp`).",
        "- **Why L4 random controls were rerun on A100:** rerunning the L4 random sets on A100 avoids inflating the random baseline in primary comparisons.",
        "",
    ]

    # 3.4 Provenance
    R += [
        "### 3.4 Provenance and Source Tables",
        "",
        "**Why this table:** lists every consolidated CSV table in `results/e3/tables/` from which this report is deterministically generated, each traced by SHA-256 to its source Drive file in `tables/inputs_manifest.json`.",
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
        "**Why this check:** summarizes how strongly each selected (and ablation) neuron set concentrates answer changes (`pred != base_pred`) in its target category versus the mean of the other categories, together with its layer profile and GPU.",
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
            fmt_cnt_pct(r["all_items_changed"], r["all_items_total"], r["all_items_flips_pct"]),
            r["gpu"],
            lprof,
        ])
    R.append(md_table(
        ["Set", "Kind", "Target Cat.", "K", "Target Flips", "95% CI (%)", "Other Mean", "Ratio", "All Benchmark Flips", "GPU", "Layer Profile"],
        rows,
        ["l", "c", "l", "r", "r", "c", "r", "r", "r", "l", "l"],
    ))
    R += [
        "",
        "*Legend — **Kind**: `real` (selected top-K) or `ablation` (exploratory subset); **Target Flips**: changed answers (`pred != base_pred`) in target category; **95% CI (%)**: paired item-bootstrap interval (`B = 1,000`); **Other Mean**: mean flip `%` across non-target categories; **Ratio**: `Target % / Other Mean %`; **Layer Profile**: `L<layer>:<count>` of zeroed neurons.*",
        "",
    ]

    # 2. Real vs Random (if any)
    if not u_rvr.empty:
        R += [
            f"## {sec_num}. Selected vs. Layer-Matched Random Neurons",
            "",
            "**Why this check:** tests whether the BiasScore-selected neurons change substantially more target-category answers than uniform random neuron sets drawn from the exact same MLP layers.",
            "",
        ]
        sec_num += 1
        rows = []
        for _, r in u_rvr[u_rvr["is_primary"]].iterrows():
            rand_str = f"{int(r['rand_target_changed_min']):,}–{int(r['rand_target_changed_max']):,} ({r['rand_target_flips_min']:.2f}–{r['rand_target_flips_max']:.2f}%, mean {r['rand_target_flips_mean']:.2f}%)"
            rows.append([
                r["comparison"],
                r["target_category"],
                fmt_cnt_pct(r["real_target_changed"], r["target_n_items"], r["real_target_flips_pct"]),
                rand_str,
                f"{r['ratio_real_over_rand_mean']:.2f}×",
                r["rank_of_real"],
                f"{r['real_gpu']} / {r['random_gpus']}",
            ])
        R.append(md_table(
            ["Comparison", "Target Cat.", "Selected Flips", "Random Flips (min–max, mean)", "Real/Rand", "Rank", "GPUs"],
            rows,
            ["l", "l", "r", "r", "r", "c", "l"],
        ))
        R += [
            "",
            "*Legend — **Selected Flips**: target-category changed answers (`changed / n_items`, `%`, `pred != base_pred`); **Random Flips (min–max, mean)**: range across the layer-matched random seeds reported as `min–max changed items (min–max%, mean %)` over the same `n_items` target denominator; **Real/Rand**: `Selected % / Random mean %`; **Rank**: position of selected set among selected + random (`1` = highest).*",
            "",
        ]

    # 3. Full Set x Category Matrix (all items pooled + ambig/disambig in details)
    R += [
        f"## {sec_num}. Changed Answers Across All Categories (`Set × Category`)",
        "",
        "**Why this table:** exposes the full `set × category` flip matrix (`pred != base_pred`, all items pooled; random families summarized by mean) so off-target spillover across individual categories can be inspected directly.",
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
    R += [
        "",
        "*Legend — Cells show `changed (flips_pct)` (`pred != base_pred`, pooled `ambig + disambig`); random-family columns show `mean% [min–max]` across seeds.*",
        "",
    ]

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
        R += [
            "",
            "*Legend — Cells show `ambig% / disambig%` (S/A/U state changes from `direction.csv`).*",
            "",
            "</details>",
            "",
        ]

    # 4. Overlap
    if not u_ov.empty:
        R += [
            f"## {sec_num}. Neuron and Changed-Item Overlap Between Sets",
            "",
            "**Why this check:** tests whether distinct target categories share the same high-BiasScore neurons and whether different interventions flip the same underlying items (pairwise shared neurons, neuron Jaccard, and changed-item Jaccard).",
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
        R += [
            "",
            "*Legend — **Shared Neurons**: identical `(layer, index)` neurons (`shared / size_A, size_B`); **Neuron Jaccard**: `|A ∩ B| / |A ∪ B|` over zeroed neurons; **Items Jacc (All / Target A / Target B)**: Jaccard overlap of changed items across all items or restricted to each set's target category.*",
            "",
        ]

    # 5. Secondary metrics
    if not u_sec.empty:
        R += [
            f"## {sec_num}. Secondary Metrics (Bias Score Deltas, Stereotype Transitions, Accuracy Transitions)",
            "",
            "**Why this table:** reports secondary directional metrics (BBQ/EsBBQ bias-score deltas, stereotype transitions `A→S`/`S→A`, and accuracy transitions) for descriptive completeness; these are secondary to answer-change localization.",
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
        R += [
            "",
            "*Legend — **A→S / S→A**: flips from anti-stereotype to stereotyped person and vice versa; **Net→Stereo**: `(A→S − S→A) / n_items` (`%`); **Corr→Inc / Inc→Corr**: accuracy transitions; **Bias Δ (95% CI)**: change in benchmark bias score vs. baseline with paired item-bootstrap CI.*",
            "",
        ]

    # 6. Margin mechanism
    if not u_msum.empty and not u_mdec.empty:
        R += [
            f"## {sec_num}. Baseline Log-Likelihood Margin Mechanism",
            "",
            "**Why this check:** verifies whether changed answers come selectively from baseline-undecided items (low `top − second` margin) rather than confident predictions, comparing median margins and flip rates across deciles 1–10.",
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
        R += [
            "",
            "*Legend — **Median Margin**: median baseline log-likelihood gap (`top − second` among `person0`/`person1`/`unknown`) of changed vs. unchanged items; **Decile**: baseline-margin decile (`1` = smallest margin / most undecided, `10` = largest margin).*",
            "",
        ]

    # 7. Granular breakdowns (groups & topics) in collapsible blocks
    if not u_grp.empty:
        R += [
            f"## {sec_num}. Granular Breakdowns (Demographic Groups and Topics)",
            "",
            "**Why this table:** checks whether answer changes concentrate in specific demographic subgroups (`answer_info`) or social topics (`relevant_social_value`, EsBBQ) within each target category.",
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
        "# results/e3 — Canonical Reference E3 Experiment Results",
        "",
        "**Status: Canonical Reference Dataset (updated 2026-11-24).**",
        "Use this directory for all paper tables, figures, and follow-up analyses. Every primary comparison is **100% hardware-matched** (all 5 EN units + `Salamandra-7B ES` + `Llama-3.1-8B ES` on A100; `Llama-3.2-3B ES` homogeneously on L4; zero mixed L4-vs-A100 comparisons). Earlier L4-mixed Drive analysis folders (such as `llama-3.2-3b_en/analysis/20261123_200457_items_3b_random/` and `esbbq_zeroed_llama8b/analysis/20261112_081632_items/`) are superseded and retained on Drive for provenance/sensitivity only.",
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
